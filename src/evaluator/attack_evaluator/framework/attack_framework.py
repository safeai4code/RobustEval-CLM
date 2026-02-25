import json
import os
from typing import Any, Dict, List, Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from tqdm import tqdm

from src.core.datasets.dataset_wrapper import AdversarialDatasetWrapper
from src.core.models.base_model import BaseModel
from src.evaluator.attack_evaluator.attack_registry import AttackRegistry
from src.evaluator.attack_evaluator.attacks.base_attack import BaseAttack
from src.evaluator.utils.evaluation import evaluator


class AttackFramework:
    """Orchestrates adversarial attack generation and model evaluation."""

    def __init__(
        self,
        model: BaseModel,
        attack_method: str = "synonym",
        attack_config: Dict[str, Any] = None,
        dataset: str = "humaneval",
        mini: bool = False,
        is_vllm: bool = False,
        attacker: Optional[BaseAttack] = None,
    ):
        """
        Initialize attack framework.

        Args:
            model: Model to attack.
            attack_method: Name of the attack to use.
            attack_config: Attack configuration dictionary.
            dataset: Dataset to use ("humaneval" or "mbpp").
            mini: Whether to use the mini version of the dataset.
            is_vllm: Whether the model is a VLLM model (enables batch processing).
            attacker: Pre-instantiated attack object.  When provided, *attack_method*
                and *attack_config* are only used for book-keeping; no new attacker is
                created.  Pass this when the attacker must be initialised *before* the
                main model (e.g. translation attacks that load a HuggingFace model on
                the same GPU).
        """
        self.model = model
        self.attack_method = attack_method
        self.attack_config = attack_config or {}
        self.dataset = dataset.lower()
        self.mini = mini
        self.is_vllm = is_vllm

        # Load dataset and derive input_type
        if self.dataset == "humaneval":
            self.problems = get_human_eval_plus(mini=mini)
            self.attack_config["input_type"] = "code"
            self.concat_prompt = True
        elif self.dataset == "mbpp":
            self.problems = get_mbpp_plus(mini=mini)
            self.attack_config["input_type"] = "prompt"
            self.concat_prompt = False
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Choose 'humaneval' or 'mbpp'")

        # Use the pre-created attacker when one is supplied; otherwise build from registry.
        if attacker is not None:
            self.attacker = attacker
        else:
            attack_class = AttackRegistry.get(attack_method)
            self.attacker = attack_class(self.attack_config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_noise_to_model(self):
        """Apply noise attack to the model and return the modified model."""
        return self.attacker.apply_noise(self.model)

    def _build_adversarial_prompts(self, problems: list) -> dict:
        """Build adversarial prompts for each problem."""
        adversarial_prompts = {}
        for task_id, problem in problems:
            prompt = problem["prompt"]
            adversarial_prompts[task_id] = self.attacker.generate_adversarial_example(prompt)
        return adversarial_prompts

    def _build_llm_adversarial_prompts(self, problems: list, generator) -> dict:
        """Build adversarial prompts via an LLM wrapper."""
        index_dict = {}
        prompts = []
        for task_id, problem in problems:
            prompts.append(problem["prompt"])
            index_dict[problem["prompt"]] = task_id

        adversarial_generation = generator.generate_dataset(prompts, self.attack_config["attack_type"])
        adversarial_prompts = {}
        for prompt, adv in adversarial_generation:
            adversarial_prompts[index_dict[prompt]] = adv
        return adversarial_prompts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_attack(
        self,
        sample_indices: Optional[List[int]] = None,
        save_prompts: str = None,
        save_results: str = None,
        gen_ori: bool = False,
    ):
        """
        Run the full attack pipeline on the selected problems.

        Args:
            sample_indices: Optional list of problem indices to attack.
                            If None, all problems are used.
            save_prompts: Directory path for saving generated prompts.
            save_results: Directory path for saving evaluation results.
            gen_ori: When True, also generate outputs for the original (unperturbed) inputs.

        Returns:
            Tuple ``(original_results, adversarial_results)`` when *gen_ori* is True,
            otherwise just ``adversarial_results``.
        """
        problems_to_attack = (
            list(self.problems.items())
            if sample_indices is None
            else [(k, v) for i, (k, v) in enumerate(self.problems.items()) if i in sample_indices]
        )

        original_generations_dict: Dict[str, Any] = {}
        adversarial_generations_dict: Dict[str, Any] = {}

        if save_prompts:
            os.makedirs(save_prompts, exist_ok=True)
        adv_prompt_file = os.path.join(save_prompts, "adversarial_prompts.jsonl") if save_prompts else None
        ori_prompt_file = os.path.join(save_prompts, "original_prompts.jsonl") if save_prompts else None

        if ori_prompt_file and os.path.exists(ori_prompt_file):
            try:
                with open(ori_prompt_file, "r") as fh:
                    for line in fh:
                        try:
                            data = json.loads(line)
                            if "task_id" in data:
                                original_generations_dict[data["task_id"]] = data
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(original_generations_dict)} existing original generations")
            except OSError as exc:
                print(f"Error reading {ori_prompt_file}: {exc}")

        if adv_prompt_file and os.path.exists(adv_prompt_file):
            try:
                with open(adv_prompt_file, "r") as fh:
                    for line in fh:
                        try:
                            data = json.loads(line)
                            if "task_id" in data:
                                adversarial_generations_dict[data["task_id"]] = data
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(adversarial_generations_dict)} existing adversarial generations")
            except OSError as exc:
                print(f"Error reading {adv_prompt_file}: {exc}")

        # Build adversarial prompts
        if self.attack_method != "llm_attack":
            adversarial_prompts = self._build_adversarial_prompts(problems_to_attack)
        else:
            attack_wrapper = AdversarialDatasetWrapper(attack_model=self.attacker)
            adversarial_prompts = self._build_llm_adversarial_prompts(problems_to_attack, attack_wrapper)
            assert len(adversarial_prompts) == len(problems_to_attack), (
                "Adversarial prompts not generated correctly"
            )

        ori_prompt_f = None
        adv_prompt_f = None
        original_results = None

        try:
            if save_prompts:
                adv_prompt_f = open(adv_prompt_file, "a")
                if gen_ori:
                    ori_prompt_f = open(ori_prompt_file, "a")

            original_generations: List[Dict[str, Any]] = []
            adversarial_generations: List[Dict[str, Any]] = []

            skipped_orig = skipped_adv = new_orig = new_adv = 0

            # Original (unperturbed) generations
            if gen_ori:
                if self.is_vllm:
                    tasks_to_generate = []
                    task_ids_to_generate = []

                    for task_id, problem in problems_to_attack:
                        if task_id in original_generations_dict:
                            original_generations.append(original_generations_dict[task_id])
                            skipped_orig += 1
                        else:
                            tasks_to_generate.append(problem["prompt"])
                            task_ids_to_generate.append(task_id)

                    if tasks_to_generate:
                        print(f"Generating {len(tasks_to_generate)} original outputs in batch...")
                        original_outputs = self.model.batch_generate(
                            tasks_to_generate, concat_prompt=self.concat_prompt
                        )
                        for task_id, prompt, output in zip(
                            task_ids_to_generate, tasks_to_generate, original_outputs
                        ):
                            entry = {"task_id": task_id, "solution": output, "prompt": prompt}
                            new_orig += 1
                            if ori_prompt_f:
                                ori_prompt_f.write(json.dumps(entry) + "\n")
                                ori_prompt_f.flush()
                            original_generations.append(entry)
                else:
                    for task_id, problem in tqdm(problems_to_attack, desc="Processing original tasks"):
                        prompt = problem["prompt"]
                        if task_id in original_generations_dict:
                            original_generations.append(original_generations_dict[task_id])
                            skipped_orig += 1
                        else:
                            output = self.model.generate(prompt, concat_prompt=self.concat_prompt)
                            entry = {"task_id": task_id, "solution": output, "prompt": prompt}
                            new_orig += 1
                            if ori_prompt_f:
                                ori_prompt_f.write(json.dumps(entry) + "\n")
                                ori_prompt_f.flush()
                            original_generations.append(entry)

            # Apply noise to model weights (noise attack only)
            if self.attack_method == "noise":
                self.model = self.attacker.apply_noise(self.model)

            # Adversarial generations
            if self.is_vllm:
                prompts_to_generate = []
                task_ids_to_generate = []

                for task_id, _ in problems_to_attack:
                    if task_id in adversarial_generations_dict:
                        adversarial_generations.append(adversarial_generations_dict[task_id])
                        skipped_adv += 1
                    else:
                        prompts_to_generate.append(adversarial_prompts[task_id])
                        task_ids_to_generate.append(task_id)

                if prompts_to_generate:
                    print(f"Generating {len(prompts_to_generate)} adversarial outputs in batch...")
                    adversarial_outputs = self.model.batch_generate(
                        prompts_to_generate, concat_prompt=self.concat_prompt
                    )
                    for task_id, prompt, output in zip(
                        task_ids_to_generate, prompts_to_generate, adversarial_outputs
                    ):
                        entry = {"task_id": task_id, "solution": output, "prompt": prompt}
                        new_adv += 1
                        if adv_prompt_f:
                            adv_prompt_f.write(json.dumps(entry) + "\n")
                            adv_prompt_f.flush()
                        adversarial_generations.append(entry)
            else:
                for task_id, _ in tqdm(problems_to_attack, desc="Processing attack tasks"):
                    adversarial_prompt = adversarial_prompts[task_id]
                    if task_id in adversarial_generations_dict:
                        adversarial_generations.append(adversarial_generations_dict[task_id])
                        skipped_adv += 1
                    else:
                        output = self.model.generate(adversarial_prompt, concat_prompt=self.concat_prompt)
                        entry = {"task_id": task_id, "solution": output, "prompt": adversarial_prompt}
                        new_adv += 1
                        if adv_prompt_f:
                            adv_prompt_f.write(json.dumps(entry) + "\n")
                            adv_prompt_f.flush()
                        adversarial_generations.append(entry)

            if gen_ori:
                print(f"Original outputs: {new_orig} newly generated, {skipped_orig} reused")
            print(f"Adversarial outputs: {new_adv} newly generated, {skipped_adv} reused")

            # Evaluate
            if gen_ori and save_results:
                original_results = evaluator(self.dataset, original_generations)
                os.makedirs(save_results, exist_ok=True)
                with open(os.path.join(save_results, "original_results.json"), "w") as fh:
                    json.dump(original_results, fh)

            adversarial_results = evaluator(self.dataset, adversarial_generations)
            if save_results:
                os.makedirs(save_results, exist_ok=True)
                with open(os.path.join(save_results, "adversarial_results.json"), "w") as fh:
                    json.dump(adversarial_results, fh)

        finally:
            if ori_prompt_f:
                ori_prompt_f.close()
            if adv_prompt_f:
                adv_prompt_f.close()

        return (original_results, adversarial_results) if gen_ori else adversarial_results


if __name__ == "__main__":
    from src.models import CodeLLaMAModel

    model = CodeLLaMAModel(
        model_path="deepseek-ai/deepseek-coder-1.3b-base",
    )
    attack_config = {
        "attack_model": 'gpt-3.5-turbo',
        "temperature": 0.7,
        "max_tokens": 150,
        "api_path": "/home/sfang9/workshop/project_test/openai/openai-key",
        "attack_type": 'paraphrase',
        "input_type": None,
    }
    attack_framework = AttackFramework(
        model=model, attack_method="llm_attack", attack_config=attack_config, dataset="mbpp")
    save_path = "/home/sfang9/workshop/project_test/test-results"
    _, _ = attack_framework.run_attack(save_prompts=save_path, save_results=save_path)

