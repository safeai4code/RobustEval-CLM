"""Attack configuration module."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AttackConfig:
    """Configuration for attack parameters"""
    # synonym replacement attack parameters
    replacement_probability: float = 0.15
    max_synonyms: int = 3
    # character case attack parameters
    char_change_probability: float = 0.5
    max_char_changes: int = 5
    # translation attack parameters
    translation_model: str = "facebook/mbart-large-50-many-to-many-mmt"
    # LLM-based attack parameters
    attack_model: str = "gpt-4o"
    attack_type: str = "paraphrase"
    adv_temperature: float = 0.7
    adv_max_tokens: int = 150
    api_path: str = ""
    # General attack parameters
    input_type: str = "prompt"
    noise_type: str = "gaussian"
    noise_level: float = 1e-3
    seed: Optional[int] = None


@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters"""
    method: Literal["bnb", "gptq", "awq"] = "bnb"
    bits: Literal[4, 8] = 8
    quant_type: Literal["nf4", "fp4"] = "nf4"
    quantize_embeddings: bool = False


@dataclass
class GenerationConfig:
    """Configuration for generation parameters"""
    num_return_sequences: int = 1
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 10
    use_beam_search: bool = False


@dataclass
class EvaluationConfig:
    """Main configuration for evaluation"""
    # Model parameters
    model_path: str
    model_type: str = "codellama"
    quantized_type: Optional[str] = None
    
    # Dataset and attack parameters
    dataset: str = "mbpp"
    attack_method: str = "synonym"
    
    # Output parameters
    save_prompts: Optional[str] = None
    save_results: Optional[str] = None
    visualization: bool = False
    
    # Other parameters
    gen_ori: bool = False
    original_results: Optional[str] = None
    
    # Nested configurations
    attack_config: AttackConfig = None
    quantization_config: QuantizationConfig = None
    generation_config: GenerationConfig = None
    
    def __post_init__(self):
        """Initialize nested configs with defaults if not provided"""
        if self.attack_config is None:
            self.attack_config = AttackConfig()
        if self.quantization_config is None:
            self.quantization_config = QuantizationConfig()
        if self.generation_config is None:
            self.generation_config = GenerationConfig()
