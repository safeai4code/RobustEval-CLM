import random
from typing import Any, Dict, Optional

import torch

from .base_attack import BaseAttack

# Import VLLM model classes for isinstance check
from src.core.models.model_implementations import VLLMModel, VLLMQuantizedModel



class NoiseAttack(BaseAttack):
    """Noise attack that adds random noise to model parameters.
    
    This attack supports both standard PyTorch models and VLLM models:
    - For standard models: Adds noise directly to parameters using PyTorch
    - For VLLM models: Uses collective RPC to add noise across all workers
    
    The VLLM noise injection requires the add_noise method to be injected
    into the Worker class before VLLM is imported (handled in model_implementations.py).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()
        # Initialize random seed if provided
        self.seed = config.get('seed')
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)

    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        required = ['noise_type', 'noise_level']
        if not all(key in self.config for key in required):
            raise ValueError(f"Config must contain: {required}")
        
        valid_noise_types = ["uniform", "gaussian"]
        if self.config['noise_type'] not in valid_noise_types:
            raise ValueError(f"noise_type must be one of {valid_noise_types}")
        
        if not isinstance(self.config['noise_level'], (int, float)) or self.config['noise_level'] < 0:
            raise ValueError("noise_level must be a non-negative number")
        
        if 'seed' in self.config and not isinstance(self.config['seed'], (int, type(None))):
            raise ValueError("seed must be an integer or None")

    def generate_adversarial_example(self, input_text: str, target_label: Optional[Any] = None) -> str:
        """For noise attacks, the prompt remains unchanged since we modify the model instead."""
        return input_text

    def apply_noise(self, model):
        """Apply noise to model parameters and return the modified model.
        
        Args:
            model: Model to apply noise to. Can be either:
                - Standard PyTorch model (CodeLLaMAModel, StaticQuantizedModel, etc.)
                - VLLM model (VLLMModel, VLLMQuantizedModel)
        
        Returns:
            Modified model with noise added to parameters
        """
        print("why this function is called twice?")
        # Check if this is a VLLM model by isinstance check
        if isinstance(model, (VLLMModel, VLLMQuantizedModel)):
            return self._apply_noise_vllm(model)
        else:
            return self._apply_noise_hf(model)
    
    def _apply_noise_vllm(self, model):
        """Apply noise to VLLM model using collective RPC."""
        print(f"Applying {self.config['noise_type']} noise to VLLM model with level {self.config['noise_level']}")
        
        # Get the executor from the VLLM model
        executor = model.model.llm_engine.model_executor
        
        # Use the injected add_noise method via collective_rpc
        seed = self.seed if self.seed is not None else 42
        results = executor.collective_rpc("add_noise", args=(self.config['noise_type'], self.config['noise_level'], seed))
        
        # Print results from each worker
        for i, result in enumerate(results):
            print(f"Worker {i}: {result}")
        
        return model
    
    def _apply_noise_hf(self, model):
        """Apply noise to standard PyTorch models."""
        if self.seed is not None:
            # Set all random seeds for complete reproducibility
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
        
        # Handle both wrapper models and direct PyTorch models
        pytorch_model = getattr(model, 'model', model)
        
        with torch.no_grad():
            for param in pytorch_model.parameters():
                if param.requires_grad:
                    if self.config['noise_type'] == 'gaussian':
                        noise = torch.randn_like(param) * self.config['noise_level']
                    else:  # uniform
                        noise = (torch.rand_like(param) * 2 - 1) * self.config['noise_level']
                    param.add_(noise)
        
        return model


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Create a simple test model
    test_model = nn.Linear(10, 5)
    original_weight = test_model.weight.clone()
    
    # Test configuration
    attack = NoiseAttack(config={
        'noise_type': 'gaussian',
        'noise_level': 0.01,
        'seed': 42
    })
    attack.validate_config()
    
    # Test prompt (should remain unchanged)
    prompt = "Write a function to calculate factorial."
    adversarial_prompt = attack.generate_adversarial_example(prompt)
    assert adversarial_prompt == prompt, "Prompt should remain unchanged for noise attacks"
    print("✓ Prompt unchanged test passed")
    
    # Test noise application on direct PyTorch model
    noisy_model = attack.apply_noise(test_model)
    modified_weight = noisy_model.weight
    
    # Verify that weights have been modified
    assert not torch.equal(original_weight, modified_weight), "Model weights should be modified"
    print("✓ Direct PyTorch model noise test passed")
    
    # Test with wrapper model (simulating CodeLLaMAModel structure)
    class MockWrapperModel:
        def __init__(self, pytorch_model):
            self.model = pytorch_model  # Store PyTorch model in .model attribute
    
    wrapper_model = MockWrapperModel(nn.Linear(10, 5))
    wrapper_model.model.weight.data = original_weight.clone()
    
    attack3 = NoiseAttack(config={'noise_type': 'gaussian', 'noise_level': 0.01, 'seed': 42})
    noisy_wrapper = attack3.apply_noise(wrapper_model)
    
    # Should modify the underlying PyTorch model
    assert not torch.equal(original_weight, noisy_wrapper.model.weight), "Wrapper model should be modified"
    print("✓ Wrapper model noise test passed")
    
    # Test reproducibility with seed
    test_model2 = nn.Linear(10, 5)
    test_model2.weight.data = original_weight.clone()
    
    attack2 = NoiseAttack(config={
        'noise_type': 'gaussian', 
        'noise_level': 0.01,
        'seed': 42
    })
    noisy_model2 = attack2.apply_noise(test_model2)
    
    # With same seed, noise should be reproducible
    assert torch.allclose(modified_weight, noisy_model2.weight, atol=1e-6), \
        "Noise should be reproducible with same seed"
    print("✓ Reproducibility test passed")
    
    # Test different noise types
    attack_uniform = NoiseAttack(config={
        'noise_type': 'uniform',
        'noise_level': 0.01,
        'seed': 42
    })
    test_model3 = nn.Linear(10, 5)
    test_model3.weight.data = original_weight.clone()
    noisy_model3 = attack_uniform.apply_noise(test_model3)
    
    # Uniform and gaussian noise should be different
    assert not torch.allclose(modified_weight, noisy_model3.weight, atol=1e-3), \
        "Different noise types should produce different results"
    print("✓ Different noise types test passed")
    
    print("All tests passed! NoiseAttack implementation is working correctly.")
