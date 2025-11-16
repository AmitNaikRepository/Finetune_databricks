# Model Checkpoints

This directory stores fine-tuned model checkpoints.

## Structure

Each training run creates a timestamped directory:

```
checkpoints/
├── customer-support-v1_20241116_143022/
│   ├── adapter_config.json       # LoRA configuration
│   ├── adapter_model.bin         # LoRA weights
│   ├── tokenizer_config.json     # Tokenizer config
│   ├── special_tokens_map.json   # Special tokens
│   └── training_args.bin         # Training arguments
└── customer-support-v2_20241117_091545/
    └── ...
```

## LoRA vs Full Fine-Tuning

This project uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning:

- **Advantages:**
  - 99% reduction in trainable parameters
  - Faster training on CPU
  - Smaller checkpoint files (~1-10MB vs 500MB+)
  - Easy to swap adapters for different tasks

- **Structure:**
  - Base model: Downloaded from HuggingFace (not stored here)
  - LoRA adapters: Stored in this directory
  - At inference: Base model + adapters loaded together

## Usage

### Loading a Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "models/checkpoints/customer-support-v1_20241116_143022"
)

# Merge for inference (optional, faster)
model = model.merge_and_unload()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "models/checkpoints/customer-support-v1_20241116_143022"
)
```

### Using with API

The inference API automatically loads models from this directory:

```bash
# Load specific checkpoint
POST /api/v1/predict
{
  "query": "How do I reset my password?",
  "model_version": "customer-support-v1"
}
```

## Checkpoint Management

Models are automatically registered in the model registry with:
- Version number
- Training metrics
- Deployment status (staging/production/archived)

See `models/registry.py` for details.

## Size Considerations

**LoRA checkpoints**: ~5-10 MB each
**Full model checkpoints**: ~500-1000 MB each

This is why we use LoRA - you can store many model versions without using excessive disk space.

## Git Ignore

Model checkpoint files are git-ignored to keep the repository small:
- `*.bin` - Model weights
- `*.safetensors` - Alternative weight format
- `*.pt`, `*.pth` - PyTorch checkpoints

Only configuration files (JSON) are committed to show structure.
