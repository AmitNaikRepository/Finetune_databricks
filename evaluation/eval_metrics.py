"""
Evaluation Metrics for LLM Fine-Tuning

This module implements comprehensive evaluation metrics for comparing
base models vs fine-tuned models:

- Perplexity (language modeling quality)
- ROUGE scores (text similarity)
- Response quality metrics
- Latency and throughput
- Cost analysis
"""

import time
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from rouge_score import rouge_scorer
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


@dataclass
class EvaluationMetrics:
    """Container for evaluation results"""

    # Language model metrics
    perplexity: float
    avg_loss: float

    # Generation quality
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float

    # Response quality (custom metrics)
    avg_response_length: float
    response_completeness: float  # % of responses >= min length

    # Performance metrics
    avg_latency_ms: float
    throughput_tokens_per_sec: float

    # Cost (simulated)
    estimated_cost_per_1k_tokens: float
    cost_savings_vs_baseline: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def __str__(self) -> str:
        """Pretty string representation"""
        return (
            f"Perplexity: {self.perplexity:.2f} | "
            f"ROUGE-L: {self.rougeL_f1:.3f} | "
            f"Latency: {self.avg_latency_ms:.0f}ms"
        )


class ModelEvaluator:
    """
    Comprehensive model evaluation

    Compares base model vs fine-tuned model across multiple dimensions:
    - Quality: perplexity, ROUGE, human-eval proxies
    - Performance: latency, throughput
    - Cost: inference cost estimation
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu"
    ):
        """
        Initialize evaluator

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run on (cpu, cuda, mps)
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def calculate_perplexity(self, texts: List[str]) -> Tuple[float, float]:
        """
        Calculate perplexity on a set of texts

        Perplexity measures how well the model predicts the text.
        Lower is better. Formula: exp(average_loss)

        Args:
            texts: List of text strings

        Returns:
            (perplexity, average_loss)
        """
        total_loss = 0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for text in track(texts, description="Calculating perplexity..."):
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Get loss
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

                # Weight by sequence length
                seq_len = inputs["input_ids"].size(1)
                total_loss += loss * seq_len
                total_tokens += seq_len

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        return perplexity, avg_loss

    def calculate_rouge_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores comparing predictions to references

        ROUGE measures text overlap:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence

        Args:
            predictions: Model-generated responses
            references: Ground truth responses

        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1_f1': np.mean(rouge1_scores),
            'rouge2_f1': np.mean(rouge2_scores),
            'rougeL_f1': np.mean(rougeL_scores)
        }

    def generate_responses(
        self,
        instructions: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Generate responses for a list of instructions

        Args:
            instructions: List of input instructions
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter

        Returns:
            List of generated responses
        """
        self.model.eval()
        responses = []

        for instruction in track(instructions, description="Generating responses..."):
            # Tokenize input
            inputs = self.tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode (skip the input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            responses.append(response.strip())

        return responses

    def measure_latency(
        self,
        instructions: List[str],
        num_runs: int = 3,
        max_new_tokens: int = 128
    ) -> Tuple[float, float]:
        """
        Measure inference latency and throughput

        Args:
            instructions: Test instructions
            num_runs: Number of runs to average
            max_new_tokens: Tokens to generate per instruction

        Returns:
            (avg_latency_ms, tokens_per_second)
        """
        latencies = []
        total_tokens = 0

        self.model.eval()
        console.print("[cyan]Measuring latency...[/cyan]")

        for run in range(num_runs):
            for instruction in instructions:
                inputs = self.tokenizer(
                    instruction,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Time generation
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy for consistency
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                total_tokens += generated_tokens

        avg_latency = np.mean(latencies)
        total_time_sec = sum(latencies) / 1000
        throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0

        return avg_latency, throughput

    def evaluate_comprehensive(
        self,
        eval_dataset: List[Dict],
        max_new_tokens: int = 128,
        cost_per_1k_tokens: float = 0.0001
    ) -> EvaluationMetrics:
        """
        Run comprehensive evaluation

        Args:
            eval_dataset: List of dicts with 'instruction' and 'golden_response'
            max_new_tokens: Maximum tokens to generate
            cost_per_1k_tokens: Cost per 1000 tokens (for cost estimation)

        Returns:
            EvaluationMetrics object
        """
        console.print("\n[bold cyan]Running Comprehensive Evaluation[/bold cyan]\n")

        # Extract data
        instructions = [item['instruction'] for item in eval_dataset]
        references = [item['golden_response'] for item in eval_dataset]

        # 1. Generate responses
        console.print("[bold]1. Generating responses...[/bold]")
        predictions = self.generate_responses(instructions, max_new_tokens=max_new_tokens)

        # 2. Calculate perplexity
        console.print("\n[bold]2. Calculating perplexity...[/bold]")
        perplexity, avg_loss = self.calculate_perplexity(
            [item['instruction'] + " " + item['golden_response'] for item in eval_dataset]
        )

        # 3. Calculate ROUGE scores
        console.print("\n[bold]3. Calculating ROUGE scores...[/bold]")
        rouge_scores = self.calculate_rouge_scores(predictions, references)

        # 4. Response quality metrics
        console.print("\n[bold]4. Analyzing response quality...[/bold]")
        response_lengths = [len(r.split()) for r in predictions]
        avg_response_length = np.mean(response_lengths)
        min_acceptable_length = 10
        completeness = sum(1 for l in response_lengths if l >= min_acceptable_length) / len(response_lengths)

        # 5. Measure latency
        console.print("\n[bold]5. Measuring performance...[/bold]")
        avg_latency, throughput = self.measure_latency(instructions[:5], num_runs=3, max_new_tokens=max_new_tokens)

        # 6. Cost estimation
        avg_tokens_per_request = (avg_response_length + np.mean([len(i.split()) for i in instructions]))
        cost_per_request = (avg_tokens_per_request / 1000) * cost_per_1k_tokens

        metrics = EvaluationMetrics(
            perplexity=perplexity,
            avg_loss=avg_loss,
            rouge1_f1=rouge_scores['rouge1_f1'],
            rouge2_f1=rouge_scores['rouge2_f1'],
            rougeL_f1=rouge_scores['rougeL_f1'],
            avg_response_length=avg_response_length,
            response_completeness=completeness,
            avg_latency_ms=avg_latency,
            throughput_tokens_per_sec=throughput,
            estimated_cost_per_1k_tokens=cost_per_1k_tokens
        )

        console.print("\n[green]✓ Evaluation complete![/green]")
        return metrics


def compare_models_table(
    base_metrics: EvaluationMetrics,
    finetuned_metrics: EvaluationMetrics,
    baseline_cost: float = 0.03  # GPT-4 cost per 1k tokens
) -> Table:
    """
    Create comparison table for base vs fine-tuned model

    Args:
        base_metrics: Metrics from base model
        finetuned_metrics: Metrics from fine-tuned model
        baseline_cost: Cost of baseline (e.g., GPT-4) for comparison

    Returns:
        Rich Table with comparison
    """
    table = Table(title="Model Comparison: Base vs Fine-Tuned", show_header=True)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Base Model", justify="right", style="yellow")
    table.add_column("Fine-Tuned", justify="right", style="green")
    table.add_column("Improvement", justify="right", style="magenta")

    def calc_improvement(base_val, ft_val, lower_is_better=False):
        """Calculate percentage improvement"""
        if base_val == 0:
            return "N/A"
        if lower_is_better:
            improvement = ((base_val - ft_val) / base_val) * 100
        else:
            improvement = ((ft_val - base_val) / base_val) * 100
        sign = "+" if improvement > 0 else ""
        return f"{sign}{improvement:.1f}%"

    # Add rows
    table.add_row(
        "Perplexity",
        f"{base_metrics.perplexity:.2f}",
        f"{finetuned_metrics.perplexity:.2f}",
        calc_improvement(base_metrics.perplexity, finetuned_metrics.perplexity, lower_is_better=True)
    )

    table.add_row(
        "ROUGE-L F1",
        f"{base_metrics.rougeL_f1:.3f}",
        f"{finetuned_metrics.rougeL_f1:.3f}",
        calc_improvement(base_metrics.rougeL_f1, finetuned_metrics.rougeL_f1)
    )

    table.add_row(
        "Response Quality",
        f"{base_metrics.response_completeness:.1%}",
        f"{finetuned_metrics.response_completeness:.1%}",
        calc_improvement(base_metrics.response_completeness, finetuned_metrics.response_completeness)
    )

    table.add_row(
        "Latency (ms)",
        f"{base_metrics.avg_latency_ms:.0f}",
        f"{finetuned_metrics.avg_latency_ms:.0f}",
        calc_improvement(base_metrics.avg_latency_ms, finetuned_metrics.avg_latency_ms, lower_is_better=True)
    )

    table.add_row(
        "Cost per 1K tokens",
        f"${base_metrics.estimated_cost_per_1k_tokens:.4f}",
        f"${finetuned_metrics.estimated_cost_per_1k_tokens:.4f}",
        f"${baseline_cost:.3f} (GPT-4)"
    )

    # Calculate cost savings
    cost_savings_vs_gpt4 = ((baseline_cost - finetuned_metrics.estimated_cost_per_1k_tokens) / baseline_cost) * 100

    table.add_row(
        "Cost Savings vs GPT-4",
        "0%",
        f"{cost_savings_vs_gpt4:.1f}%",
        f"💰 {cost_savings_vs_gpt4:.1f}% cheaper"
    )

    return table


# Example usage
if __name__ == "__main__":
    """
    Test evaluation metrics

    Run: python evaluation/eval_metrics.py
    """
    console.print("\n[bold cyan]═══ Evaluation Metrics Demo ═══[/bold cyan]\n")

    # This is a demo - in practice, you'd load actual models
    console.print("[yellow]Note: This is a demonstration of the evaluation framework[/yellow]")
    console.print("[yellow]Actual model evaluation requires trained models[/yellow]\n")

    # Example metrics (simulated)
    base_metrics = EvaluationMetrics(
        perplexity=35.2,
        avg_loss=3.56,
        rouge1_f1=0.45,
        rouge2_f1=0.25,
        rougeL_f1=0.40,
        avg_response_length=42.5,
        response_completeness=0.75,
        avg_latency_ms=245,
        throughput_tokens_per_sec=85.3,
        estimated_cost_per_1k_tokens=0.0001
    )

    finetuned_metrics = EvaluationMetrics(
        perplexity=18.7,
        avg_loss=2.93,
        rouge1_f1=0.68,
        rouge2_f1=0.48,
        rougeL_f1=0.62,
        avg_response_length=58.2,
        response_completeness=0.92,
        avg_latency_ms=250,
        throughput_tokens_per_sec=82.1,
        estimated_cost_per_1k_tokens=0.0001
    )

    # Display comparison
    comparison = compare_models_table(base_metrics, finetuned_metrics, baseline_cost=0.03)
    console.print(comparison)

    console.print("\n[green]✓ Evaluation framework ready![/green]\n")
