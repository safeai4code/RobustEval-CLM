import torch
import random
from typing import Dict, Any
from src.framework.base_attack import BaseAttack


class NoiseAttack(BaseAttack):
    """
    Implementation of noise-based attacks on model parameters.
    This attack adds controlled noise to model parameters instead of modifying inputs.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize noise attack with configuration.
        
        Args:
            config: Dictionary containing configuration parameters:
                - noise_type: Type of noise to add ('uniform' or 'gaussian')
                - noise_level: Level of noise to add
        """
        super().__init__(config)
        self.noise_type = config.get('noise_type', 'gaussian')
        self.noise_level = config.get('noise_level', 1e-3)
        self.seed = config.get('seed', 44)

    def validate_config(self) -> None:
        """Validate the noise configuration parameters."""
        # valid_noise_types = ["uniform", "gaussian"]
        # if self.noise_type not in valid_noise_types:
        #     raise ValueError(f"Noise type must be one of {valid_noise_types}")
        # if self.noise_level < 0:
        #     raise ValueError("Noise level must be positive")
    
    def add_noise_to_model(self, model):
        """
        Add noise to model parameters based on configuration.
        
        Args:
            model: The model to add noise to
            
        Returns:
            The model with added noise
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        
        with torch.no_grad():
            for param in model.model.parameters():
                if param.requires_grad:
                    if self.noise_type == 'gaussian':
                        noise = torch.randn_like(param) * self.noise_level
                    else:  # uniform
                        noise = (torch.rand_like(param) * 2 - 1) * self.noise_level
                    param.add_(noise)
        
        return model
    
    def generate_adversarial_example(self, prompt: str) -> str:
        """
        For noise attacks, the prompt doesn't change - we change the model instead.
        This method is implemented to maintain compatibility with the BaseAttack interface.
        
        Args:
            prompt: The original prompt
            
        Returns:
            The same prompt unchanged
        """
        # For noise attacks, we don't modify the input
        return prompt
