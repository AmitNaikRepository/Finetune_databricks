"""
Model Inference Module

Handles loading and running inference with fine-tuned models.
Supports:
- Model caching for fast inference
- Batch inference
- Multiple model versions
- Performance monitoring
"""

import time
from typing import Dict, List, Optional
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from rich.console import Console

console = Console()


class ModelInferenceEngine:
    """
    Production-ready inference engine

    Features:
    - Model caching (don't reload for every request)
    - Version management
    - Performance tracking
    - Batch support
    - Error handling
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize inference engine

        Args:
            device: Device to run inference on
        """
        self.device = device
        self.loaded_models: Dict[str, tuple] = {}  # Cache loaded models
        self.inference_count = 0
        self.total_latency = 0.0

        console.print(f"[green]Inference engine initialized on {device}[/green]")

    def load_model(self, model_path: str, model_version: str = "default") -> tuple:
        """
        Load model and tokenizer with caching

        Args:
            model_path: Path to model checkpoint
            model_version: Version identifier for caching

        Returns:
            (model, tokenizer)
        """
        # Check cache
        if model_version in self.loaded_models:
            console.print(f"[cyan]Using cached model: {model_version}[/cyan]")
            return self.loaded_models[model_version]

        console.print(f"[cyan]Loading model: {model_path}[/cyan]")

        # Check if LoRA model
        adapter_config_path = Path(model_path) / "adapter_config.json"

        if adapter_config_path.exists():
            # Load LoRA model
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config.get('base_model_name_or_path', 'distilgpt2')

            # Load base + adapters
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()  # Merge for faster inference

        else:
            # Load full model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Move to device
        model.to(self.device)
        model.eval()

        # Cache
        self.loaded_models[model_version] = (model, tokenizer)

        console.print(f"[green]✓ Model loaded and cached: {model_version}[/green]")
        return model, tokenizer

    def predict(
        self,
        query: str,
        model_version: str = "default",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> Dict:
        """
        Generate response for a single query

        Args:
            query: Input query/instruction
            model_version: Model version to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Dict with response and metadata
        """
        if model_version not in self.loaded_models:
            raise ValueError(f"Model {model_version} not loaded. Call load_model() first.")

        model, tokenizer = self.loaded_models[model_version]

        # Format query (assumes Alpaca format was used in training)
        formatted_query = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{query}\n\n### Response:\n"
        )

        # Tokenize
        inputs = tokenizer(
            formatted_query,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode response (skip input)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Update stats
        self.inference_count += 1
        self.total_latency += latency_ms

        return {
            "response": response,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 2),
            "tokens_generated": len(generated_tokens),
            "metadata": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens
            }
        }

    def predict_batch(
        self,
        queries: List[str],
        model_version: str = "default",
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> List[Dict]:
        """
        Batch inference for multiple queries

        Args:
            queries: List of input queries
            model_version: Model version to use
            max_new_tokens: Maximum tokens per response
            **generation_kwargs: Additional generation parameters

        Returns:
            List of prediction dicts
        """
        results = []

        for query in queries:
            result = self.predict(
                query,
                model_version=model_version,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Get inference statistics"""
        return {
            "total_inferences": self.inference_count,
            "average_latency_ms": round(self.total_latency / self.inference_count, 2) if self.inference_count > 0 else 0,
            "loaded_models": list(self.loaded_models.keys()),
            "device": self.device
        }

    def unload_model(self, model_version: str):
        """Unload a model from cache to free memory"""
        if model_version in self.loaded_models:
            del self.loaded_models[model_version]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            console.print(f"[yellow]Model unloaded: {model_version}[/yellow]")


# Global inference engine instance
_inference_engine: Optional[ModelInferenceEngine] = None


def get_inference_engine(device: str = "cpu") -> ModelInferenceEngine:
    """
    Get singleton inference engine

    This pattern ensures we only have one engine instance
    and models stay loaded in memory between requests
    """
    global _inference_engine

    if _inference_engine is None:
        _inference_engine = ModelInferenceEngine(device=device)

    return _inference_engine


# Example usage
if __name__ == "__main__":
    """
    Test inference engine

    Run: python api/inference.py
    """
    console.print("\n[bold cyan]═══ Inference Engine Demo ═══[/bold cyan]\n")

    # Initialize engine
    engine = ModelInferenceEngine(device="cpu")

    console.print("[yellow]Note: For actual inference, you need a trained model[/yellow]")
    console.print("[yellow]This demo shows the inference engine structure[/yellow]\n")

    # Example: Load base model for testing
    console.print("[bold]Loading base model (distilgpt2)...[/bold]")
    model, tokenizer = engine.load_model("distilgpt2", model_version="base")

    # Test queries
    test_queries = [
        "How do I reset my password?",
        "Where is my order?",
        "I need help with my account"
    ]

    console.print("\n[bold]Running test predictions...[/bold]\n")

    for i, query in enumerate(test_queries, 1):
        console.print(f"[cyan]Query {i}:[/cyan] {query}")

        result = engine.predict(
            query,
            model_version="base",
            max_new_tokens=50,
            temperature=0.7
        )

        console.print(f"[green]Response:[/green] {result['response'][:100]}...")
        console.print(f"[dim]Latency: {result['latency_ms']}ms | Tokens: {result['tokens_generated']}[/dim]\n")

    # Show stats
    console.print("[bold]Inference Statistics:[/bold]")
    stats = engine.get_stats()
    for key, value in stats.items():
        console.print(f"  {key}: {value}")

    console.print("\n[green]✓ Inference engine demo complete![/green]\n")
