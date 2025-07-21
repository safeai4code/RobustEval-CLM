import json
import os
from typing import Any, Dict, List, Optional, Type

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm import tqdm

from src.core.datasets.dataset_wrapper import AdversarialDatasetWrapper
from src.evaluator.attack_evaluator.attacks.base_attack import BaseAttack
from src.core.models.base_model import BaseModel
from src.evaluator.utils.evaluation import evaluator


class AttackRegistry:
    """Registry for attack classes to reduce coupling."""
    
    _attacks: Dict[str, Type[BaseAttack]] = {}
    
    @classmethod
    def register(cls, name: str, attack_class: Type[BaseAttack]):
        """Register an attack class."""
        cls._attacks[name] = attack_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseAttack]:
        """Get an attack class by name."""
        if name not in cls._attacks:
            raise ValueError(f"Unknown attack method: {name}")
        return cls._attacks[name]
    
    @classmethod
    def list_attacks(cls) -> List[str]:
        """List all registered attack names."""
        return list(cls._attacks.keys())
    
    @classmethod
    def auto_discover(cls):
        """Auto-discover and register all attack classes."""
        # Import all attack classes to trigger registration
        try:
            from src.evaluator.attack_evaluator.attacks import (
                SynonymAttack, CharacterCaseAttack, TranslationAttack,
                ChatGPTAttack, NoiseAttack, NaturalNoiseAttack,
                SemanticAttack, StructuralAttack
            )
            
            # Register all attacks
            cls.register("synonym", SynonymAttack)
            cls.register("char", CharacterCaseAttack)
            cls.register("translation", TranslationAttack)
            cls.register("translate", TranslationAttack)  # alias
            cls.register("chatgpt", ChatGPTAttack)
            cls.register("llm_attack", ChatGPTAttack)  # alias
            cls.register("noise", NoiseAttack)
            cls.register("natural_noise", NaturalNoiseAttack)
            cls.register("semantic", SemanticAttack)
            cls.register("structural", StructuralAttack)
            
        except ImportError as e:
            print(f"Warning: Could not import some attack classes: {e}")


# Auto-discover attacks on module load
AttackRegistry.auto_discover()


class AttackFramework:
    def __init__(
        self,
        model: BaseModel,
        attack_method: str = "synonym",
        attack_config: Dict[str, Any] = None,
        dataset: str = "humaneval",
        mini: bool = False
    ):
        """
        Initialize attack framework.

        Args:
            model: Model to attack
            attack_method: Type of attack to use
            attack_config: Attack configuration
            dataset: Dataset to use ("humaneval" or "mbpp")
            mini: Whether to use mini version of dataset
        """
        self.model = model
        self.attack_method = attack_method
        self.attack_config = attack_config or {}
        self.dataset = dataset.lower()
        self.mini = mini
        
        # Load appropriate dataset
        if self.dataset == "humaneval":
            self.problems = get_human_eval_plus(mini=mini)
            self.attack_config['input_type'] = 'code'
        elif self.dataset == "mbpp":
            self.problems = get_mbpp_plus(mini=mini)
            self.attack_config['input_type'] = 'prompt'
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Choose 'humaneval' or 'mbpp'")
        
        # Get attack class from registry
        attack_class = AttackRegistry.get(attack_method)
        self.attacker = attack_class(self.attack_config)
        
        # For noise attacks, apply noise to the model
        if self.attack_method == "noise":
            self.model = self._apply_noise_to_model()

    def _apply_noise_to_model(self):
        """Apply noise attack to the model and return the modified model."""
        return self.attacker.apply_noise(self.model)

    def _build_adversarial_prompts(self, problems: list) -> dict:
        """Build adversarial prompts for the given list of prompts."""
        adversarial_prompts = {}
        for task_id, problem in problems:
            prompt = problem["prompt"]
            adversarial_prompt = self.attacker.generate_adversarial_example(prompt)
            adversarial_prompts[task_id] = adversarial_prompt
        return adversarial_prompts

    def _build_llm_adversarial_prompts(self, problems: list, generator) -> dict:
        """Build adversarial prompts for the given list of prompts."""
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
    
    def run_attack(
        self, sample_indices: Optional[List[int]] = None,
        save_prompts: str = None, save_results: str = None,
        gen_ori: bool = False
    ):
        """
        Run attack pipeline on selected problems.
        
        Args:
            sample_indices: Optional list of problem indices to attack.
                            If None, all problems will be used.
            save_prompts: Optional path to save prompts.
            save_results: Optional path to save results.
            gen_ori: Whether to generate outputs for original inputs.
                     If False, will attempt to load existing outputs.
        
        Returns:
            Dictionary containing attack results and evaluation metrics
        """
        
        # Track generations
        original_generations = []
        adversarial_generations = []
        
        # Process each problem
        problems_to_attack = (
            list(self.problems.items()) if sample_indices is None 
            else [(k, v) for i, (k, v) in enumerate(self.problems.items()) 
                  if i in sample_indices]
        )
        
        # Load existing prompts
        original_generations_dict = {}
        adversarial_generations_dict = {}
        
        if save_prompts:
            os.makedirs(save_prompts, exist_ok=True)
            adv_prompt_file = os.path.join(save_prompts, "adversarial_prompts.jsonl")
            ori_prompt_file = os.path.join(save_prompts, "original_prompts.jsonl")
        
        if os.path.exists(ori_prompt_file):
            try:
                with open(ori_prompt_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "task_id" in data:
                                original_generations_dict[data["task_id"]] = data
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(original_generations_dict)} existing original generations")
            except Exception as e:
                print(f"Error reading {ori_prompt_file}: {e}")
                
        if os.path.exists(adv_prompt_file):
            try:
                with open(adv_prompt_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "task_id" in data:
                                adversarial_generations_dict[data["task_id"]] = data
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(adversarial_generations_dict)} existing adversarial generations")
            except Exception as e:
                print(f"Error reading {adv_prompt_file}: {e}")


        # Build Adversarial Prompts
        if self.attack_method != "llm_attack":
            adversarial_prompts = self._build_adversarial_prompts(problems_to_attack)
        else:
            attack_wrapper = AdversarialDatasetWrapper(
                attack_model=self.attacker,
            )
            adversarial_prompts = self._build_llm_adversarial_prompts(problems_to_attack, attack_wrapper)
            assert len(adversarial_prompts) == len(problems_to_attack), "Adversarial prompts not generated correctly"
            
        ori_prompt_f = None
        adv_prompt_f = None
        try:
            if save_prompts:
                adv_prompt_f = open(adv_prompt_file, 'a')
                if gen_ori:
                    ori_prompt_f = open(ori_prompt_file, 'a')
            
            original_generations = []
            adversarial_generations = []
            
            # count skipped and newly generated prompts
            skipped_orig = 0
            skipped_adv = 0
            new_orig = 0
            new_adv = 0
            
            for task_id, problem in tqdm(problems_to_attack, desc="Processing tasks"):
                prompt = problem["prompt"]
                adversarial_prompt = adversarial_prompts[task_id]
                
                if gen_ori:
                    if task_id in original_generations_dict:
                        original_gen = original_generations_dict[task_id]
                        skipped_orig += 1
                    else:
                        original_output = self.model.generate(prompt)
                        original_gen = {
                            "task_id": task_id,
                            "solution": original_output,
                            "prompt": prompt,
                        }
                        new_orig += 1
                        
                        if save_prompts and ori_prompt_f:
                            ori_prompt_f.write(json.dumps(original_gen) + '\n')
                            ori_prompt_f.flush()
                    
                    original_generations.append(original_gen)
                
                if task_id in adversarial_generations_dict:
                    adversarial_gen = adversarial_generations_dict[task_id]
                    skipped_adv += 1
                else:
                    adversarial_output = self.model.generate(adversarial_prompt)
                    adversarial_gen = {
                        "task_id": task_id,
                        "solution": adversarial_output,
                        "prompt": adversarial_prompt,
                    }
                    new_adv += 1
                    
                    if save_prompts and adv_prompt_f:
                        adv_prompt_f.write(json.dumps(adversarial_gen) + '\n')
                        adv_prompt_f.flush()  # 确保立即写入磁盘
                
                adversarial_generations.append(adversarial_gen)
            
            # 显示处理统计信息
            if gen_ori:
                print(f"Original outputs: {new_orig} newly generated, {skipped_orig} reused from existing file")
            print(f"Adversarial outputs: {new_adv} newly generated, {skipped_adv} reused from existing file")
            
            # Evaluate original and adversarial outputs
            if gen_ori and save_results:
                original_results = evaluator(self.dataset, original_generations)
                os.makedirs(save_results, exist_ok=True)
                with open(os.path.join(save_results, "original_results.json"), 'w') as f:
                    json.dump(original_results, f)
            
            adversarial_results = evaluator(self.dataset, adversarial_generations)
            if save_results:
                os.makedirs(save_results, exist_ok=True)
                with open(os.path.join(save_results, "adversarial_results.json"), 'w') as f:
                    json.dump(adversarial_results, f)
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
