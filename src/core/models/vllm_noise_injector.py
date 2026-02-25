"""vLLM Worker noise injection utilities.

This module provides utilities to inject noise into vLLM models by adding
custom methods to the Worker class before vLLM is imported.
"""

import torch


def _worker_add_noise(self, noise_type="gaussian", noise_scale=0.01, seed=42):
    """Add noise to model parameters on each worker's GPU.
    
    This method is injected into the vLLM Worker class and can be called
    via executor.collective_rpc("add_noise", args=(noise_type, noise_scale, seed)).
    
    Args:
        noise_type: Type of noise to add (gaussian or uniform)
        noise_scale: Scale of the noise to add (standard deviation for Gaussian)
        seed: Random seed for reproducibility
        
    Returns:
        Status message indicating noise was added
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    model = self.model_runner.model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_floating_point():
                if noise_type == "gaussian":
                    noise = torch.randn_like(param) * noise_scale
                elif noise_type == "uniform":
                    noise = (torch.rand_like(param) * 2 - 1) * noise_scale
                else:
                    raise ValueError(f"Invalid noise type: {noise_type}")
                param.add_(noise)
    return f"Worker rank {self.rank}: noise added to {sum(1 for _ in model.named_parameters())} params"


def inject_noise_method():
    """Inject add_noise method to vLLM Worker class.
    
    This must be called BEFORE importing vLLM in model_implementations.py.
    The injection is done only once to avoid issues with multiple imports.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker

        # Only inject if not already present
        if not hasattr(Worker, 'add_noise'):
            Worker.add_noise = _worker_add_noise
            print("Successfully injected add_noise method to vLLM Worker class")
        
    except ImportError:
        # vLLM not installed or v1 worker not available
        print("Warning: Could not inject noise method - vLLM v1 worker not available")
