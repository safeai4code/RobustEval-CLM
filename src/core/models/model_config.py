"""Model configuration utilities."""

import torch
from typing import Optional, Dict, Any


class ModelConfigManager:
    """Utility class for creating and managing model configurations."""
    
    @staticmethod
    def create_model_config(
        model_type: str,
        quantized_type: Optional[str],
        quant_params: dict,
        gen_params: dict
    ) -> dict:
        """Create model configuration including quantization and generation settings"""
        model_config = {}
        
        # Add generation config if provided
        if gen_params:
            model_config["generation_config"] = {
                "num_return_sequences": gen_params.get("num_return_sequences", 1),
                "max_length": gen_params.get("max_length", 512),
                "temperature": gen_params.get("temperature", 0.7),
                "top_p": gen_params.get("top_p", 0.95),
                "num_beams": gen_params.get("num_beams", 10),
                "use_beam_search": gen_params.get("use_beam_search", False)
            }

        # Add quantization config if using quantization
        if quantized_type == "static":
            model_config["quant_config"] = {
                "method": quant_params.get("method", "bnb"),
                "bits": quant_params.get("bits", 8),
                "compute_dtype": torch.float16,
                "quant_type": quant_params.get("quant_type", "nf4"),
                "dataset": quant_params.get("dataset", "c4")
            }
        elif quantized_type == "dynamic":
            model_config["quant_config"] = {
                "bits": quant_params.get("bits", 8),
                "quantize_embeddings": quant_params.get("quantize_embeddings", False)
            }

        return model_config 