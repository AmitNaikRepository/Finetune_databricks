"""
LoRA Configuration Module

Demonstrates Parameter-Efficient Fine-Tuning (PEFT) configuration
for enterprise LLM training.

LoRA (Low-Rank Adaptation) Benefits:
- Reduces trainable parameters by ~99%
- Faster training and lower memory usage
- Easy to swap adapters for different tasks
- Maintains base model quality
"""

from dataclasses import dataclass, field
from typing import List, Optional
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel
import torch


@dataclass
class LoRAConfigBuilder:
    """
    Builder for LoRA configuration with enterprise best practices

    LoRA Hyperparameter Guidelines:
    - Rank (r): 4-16 for most tasks, 32-64 for complex tasks
    - Alpha: Typically 2x rank (α/r ratio controls adaptation strength)
    - Dropout: 0.05-0.1 for regularization
    - Target modules: Query+Value projection for efficiency, all for quality
    """

    # Core LoRA parameters
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1

    # Target modules (model-specific)
    # GPT-2/GPT-Neo: ["c_attn", "c_proj"]
    # BLOOM: ["query_key_value"]
    # LLaMA: ["q_proj", "v_proj"]
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])

    # Advanced options
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: TaskType = TaskType.CAUSAL_LM
    inference_mode: bool = False
    modules_to_save: Optional[List[str]] = None

    # Quantization (for large models)
    use_8bit: bool = False
    use_4bit: bool = False

    def build(self) -> LoraConfig:
        """
        Build PEFT LoraConfig object

        Returns:
            LoraConfig ready for model training
        """
        config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
            inference_mode=self.inference_mode,
            modules_to_save=self.modules_to_save,
        )
        return config

    @classmethod
    def from_yaml(cls, config_dict: dict) -> "LoRAConfigBuilder":
        """
        Create LoRAConfigBuilder from YAML config

        Args:
            config_dict: Dictionary from YAML config file

        Returns:
            Configured LoRAConfigBuilder
        """
        lora_config = config_dict.get("lora", {})

        # Map string task type to enum
        task_type_str = lora_config.get("task_type", "CAUSAL_LM")
        task_type = getattr(TaskType, task_type_str)

        return cls(
            rank=lora_config.get("rank", 8),
            alpha=lora_config.get("alpha", 16),
            dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", ["c_attn", "c_proj"]),
            bias=lora_config.get("bias", "none"),
            task_type=task_type,
        )

    def calculate_trainable_params(self, base_model: PreTrainedModel) -> dict:
        """
        Calculate parameter efficiency metrics

        Args:
            base_model: The base model before LoRA

        Returns:
            Dict with parameter counts and efficiency metrics
        """
        # Get base model parameters
        base_params = sum(p.numel() for p in base_model.parameters())

        # Estimate LoRA parameters
        # For each target module: rank * (d_model_in + d_model_out)
        # This is an approximation
        lora_params = 0
        for name, module in base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if hasattr(module, "weight"):
                    weight_shape = module.weight.shape
                    # LoRA adds two matrices: (d, r) and (r, d)
                    lora_params += self.rank * (weight_shape[0] + weight_shape[1])

        efficiency = (lora_params / base_params) * 100 if base_params > 0 else 0

        return {
            "base_parameters": base_params,
            "lora_parameters": lora_params,
            "total_parameters": base_params + lora_params,
            "trainable_percentage": efficiency,
            "memory_savings": f"{(1 - efficiency/100) * 100:.1f}%"
        }


def apply_lora(model: PreTrainedModel, lora_config: LoraConfig) -> PreTrainedModel:
    """
    Apply LoRA to a pretrained model

    Args:
        model: Base pretrained model
        lora_config: LoRA configuration

    Returns:
        Model with LoRA adapters applied

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> lora_config = LoRAConfigBuilder(rank=8).build()
        >>> model = apply_lora(model, lora_config)
        >>> # Now model has trainable LoRA adapters
    """
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable%: {100 * trainable_params / all_param:.2f}%")

    return model


# Preset configurations for common use cases
LORA_PRESETS = {
    "minimal": {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
        "description": "Minimal LoRA for simple tasks, fastest training"
    },
    "balanced": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "description": "Balanced configuration for most tasks (recommended)"
    },
    "quality": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.1,
        "description": "Higher quality for complex tasks, slower training"
    },
    "maximum": {
        "rank": 32,
        "alpha": 64,
        "dropout": 0.1,
        "description": "Maximum capacity for very complex tasks"
    }
}


def get_preset_config(preset: str = "balanced") -> LoRAConfigBuilder:
    """
    Get a preset LoRA configuration

    Args:
        preset: One of "minimal", "balanced", "quality", "maximum"

    Returns:
        LoRAConfigBuilder with preset values

    Example:
        >>> config = get_preset_config("quality")
        >>> lora_config = config.build()
    """
    if preset not in LORA_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(LORA_PRESETS.keys())}")

    preset_values = LORA_PRESETS[preset]
    print(f"Using preset '{preset}': {preset_values['description']}")

    return LoRAConfigBuilder(
        rank=preset_values["rank"],
        alpha=preset_values["alpha"],
        dropout=preset_values["dropout"]
    )


# Example usage
if __name__ == "__main__":
    """
    Test LoRA configuration

    Run: python training/lora_config.py
    """
    print("\n=== LoRA Configuration Examples ===\n")

    # Example 1: Basic configuration
    print("1. Basic LoRA Configuration:")
    basic_config = LoRAConfigBuilder(rank=8, alpha=16)
    print(f"   Rank: {basic_config.rank}")
    print(f"   Alpha: {basic_config.alpha}")
    print(f"   Target modules: {basic_config.target_modules}\n")

    # Example 2: Preset configurations
    print("2. Available Presets:")
    for name, values in LORA_PRESETS.items():
        print(f"   {name:10s}: r={values['rank']:2d}, α={values['alpha']:2d} - {values['description']}")

    # Example 3: Build config for different models
    print("\n3. Model-Specific Configurations:")

    configs = {
        "GPT-2": LoRAConfigBuilder(target_modules=["c_attn", "c_proj"]),
        "LLaMA": LoRAConfigBuilder(target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]),
        "BLOOM": LoRAConfigBuilder(target_modules=["query_key_value"]),
    }

    for model_name, config in configs.items():
        print(f"   {model_name:10s}: {config.target_modules}")

    print("\n✓ LoRA configuration module ready!\n")
