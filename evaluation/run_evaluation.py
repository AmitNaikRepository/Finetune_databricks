#!/usr/bin/env python3
"""
Model Evaluation Runner

Compares base model vs fine-tuned model and generates evaluation report.

Usage:
    python evaluation/run_evaluation.py --model models/checkpoints/my-model
    python evaluation/run_evaluation.py --model-version customer-support-v1
    python evaluation/run_evaluation.py --compare-base
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel

from evaluation.eval_metrics import ModelEvaluator, compare_models_table, EvaluationMetrics
from models.registry import ModelRegistry

console = Console()


def load_eval_dataset(dataset_path: str = "evaluation/eval_dataset.json") -> list:
    """Load evaluation dataset"""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def load_model_for_evaluation(
    model_path: str,
    device: str = "cpu"
) -> tuple:
    """
    Load model and tokenizer for evaluation

    Args:
        model_path: Path to model checkpoint
        device: Device to load on

    Returns:
        (model, tokenizer)
    """
    console.print(f"[cyan]Loading model from: {model_path}[/cyan]")

    # Check if it's a LoRA model
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        # Load LoRA model
        console.print("[cyan]Detected LoRA model, loading with adapters...[/cyan]")

        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get('base_model_name_or_path', 'distilgpt2')

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32
        )

        # Load LoRA adapters
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

    model.to(device)
    console.print(f"[green]✓ Model loaded on {device}[/green]\n")

    return model, tokenizer


def save_evaluation_report(
    base_metrics: EvaluationMetrics,
    finetuned_metrics: EvaluationMetrics,
    output_path: str = "evaluation/results"
):
    """Save evaluation results to JSON and HTML"""

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"evaluation_{timestamp}.json"
    results = {
        "timestamp": timestamp,
        "base_model_metrics": base_metrics.to_dict() if base_metrics else None,
        "finetuned_model_metrics": finetuned_metrics.to_dict(),
        "improvements": {}
    }

    if base_metrics:
        # Calculate improvements
        results["improvements"] = {
            "perplexity_reduction": ((base_metrics.perplexity - finetuned_metrics.perplexity) / base_metrics.perplexity) * 100,
            "rouge_l_improvement": ((finetuned_metrics.rougeL_f1 - base_metrics.rougeL_f1) / base_metrics.rougeL_f1) * 100,
            "quality_improvement": ((finetuned_metrics.response_completeness - base_metrics.response_completeness) / base_metrics.response_completeness) * 100,
        }

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]✓ Evaluation results saved to: {json_path}[/green]")

    # Create simple HTML report
    html_path = output_dir / f"evaluation_{timestamp}.html"
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-name {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .improvement {{
            color: #10b981;
            font-size: 16px;
        }}
        .comparison-table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Model Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="metric-card">
        <div class="metric-name">Fine-Tuned Model Perplexity</div>
        <div class="metric-value">{finetuned_metrics.perplexity:.2f}</div>
        {f'<div class="improvement">↓ {results["improvements"]["perplexity_reduction"]:.1f}% vs base model</div>' if base_metrics else ''}
    </div>

    <div class="metric-card">
        <div class="metric-name">ROUGE-L F1 Score</div>
        <div class="metric-value">{finetuned_metrics.rougeL_f1:.3f}</div>
        {f'<div class="improvement">↑ {results["improvements"]["rouge_l_improvement"]:.1f}% vs base model</div>' if base_metrics else ''}
    </div>

    <div class="metric-card">
        <div class="metric-name">Response Quality (Completeness)</div>
        <div class="metric-value">{finetuned_metrics.response_completeness:.1%}</div>
        {f'<div class="improvement">↑ {results["improvements"]["quality_improvement"]:.1f}% vs base model</div>' if base_metrics else ''}
    </div>

    <div class="comparison-table">
        <h2 style="padding: 20px 20px 0;">📊 Detailed Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                {'<th>Base Model</th>' if base_metrics else ''}
                <th>Fine-Tuned Model</th>
            </tr>
            <tr>
                <td>Perplexity</td>
                {f'<td>{base_metrics.perplexity:.2f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.perplexity:.2f}</strong></td>
            </tr>
            <tr>
                <td>Average Loss</td>
                {f'<td>{base_metrics.avg_loss:.3f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.avg_loss:.3f}</strong></td>
            </tr>
            <tr>
                <td>ROUGE-1 F1</td>
                {f'<td>{base_metrics.rouge1_f1:.3f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.rouge1_f1:.3f}</strong></td>
            </tr>
            <tr>
                <td>ROUGE-2 F1</td>
                {f'<td>{base_metrics.rouge2_f1:.3f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.rouge2_f1:.3f}</strong></td>
            </tr>
            <tr>
                <td>ROUGE-L F1</td>
                {f'<td>{base_metrics.rougeL_f1:.3f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.rougeL_f1:.3f}</strong></td>
            </tr>
            <tr>
                <td>Avg Response Length (words)</td>
                {f'<td>{base_metrics.avg_response_length:.1f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.avg_response_length:.1f}</strong></td>
            </tr>
            <tr>
                <td>Response Completeness</td>
                {f'<td>{base_metrics.response_completeness:.1%}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.response_completeness:.1%}</strong></td>
            </tr>
            <tr>
                <td>Avg Latency (ms)</td>
                {f'<td>{base_metrics.avg_latency_ms:.0f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.avg_latency_ms:.0f}</strong></td>
            </tr>
            <tr>
                <td>Throughput (tokens/sec)</td>
                {f'<td>{base_metrics.throughput_tokens_per_sec:.1f}</td>' if base_metrics else ''}
                <td><strong>{finetuned_metrics.throughput_tokens_per_sec:.1f}</strong></td>
            </tr>
        </table>
    </div>

    <div class="metric-card">
        <h2>💰 Cost Analysis</h2>
        <p><strong>Fine-tuned model cost:</strong> ${finetuned_metrics.estimated_cost_per_1k_tokens:.4f} per 1K tokens</p>
        <p><strong>GPT-4 baseline cost:</strong> $0.030 per 1K tokens</p>
        <p class="improvement" style="font-size: 20px;">
            💵 Cost savings: {((0.03 - finetuned_metrics.estimated_cost_per_1k_tokens) / 0.03 * 100):.1f}% cheaper than GPT-4
        </p>
    </div>
</body>
</html>
    """

    with open(html_path, 'w') as f:
        f.write(html_content)

    console.print(f"[green]✓ HTML report saved to: {html_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        help="Model version from registry (alternative to --model)"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate base model for comparison"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="evaluation/eval_dataset.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run evaluation on"
    )

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Model Evaluation Pipeline[/bold cyan]\n"
        "Comprehensive evaluation with quality and performance metrics",
        border_style="cyan"
    ))

    # Determine model path
    if args.model_version:
        registry = ModelRegistry()
        model_info = registry.get_model(args.model_version)
        if not model_info:
            console.print(f"[red]✗ Model version '{args.model_version}' not found in registry[/red]")
            sys.exit(1)
        model_path = model_info['model_path']
        console.print(f"[green]Found model in registry: {args.model_version}[/green]\n")
    elif args.model:
        model_path = args.model
    else:
        console.print("[red]✗ Must specify either --model or --model-version[/red]")
        sys.exit(1)

    # Load evaluation dataset
    console.print(f"[cyan]Loading evaluation dataset from: {args.eval_dataset}[/cyan]")
    eval_dataset = load_eval_dataset(args.eval_dataset)
    console.print(f"[green]✓ Loaded {len(eval_dataset)} evaluation examples[/green]\n")

    # Evaluate base model (if requested)
    base_metrics = None
    if args.compare_base:
        console.print("[bold]Evaluating Base Model[/bold]\n")
        base_model, base_tokenizer = load_model_for_evaluation("distilgpt2", args.device)
        base_evaluator = ModelEvaluator(base_model, base_tokenizer, args.device)
        base_metrics = base_evaluator.evaluate_comprehensive(eval_dataset, cost_per_1k_tokens=0.0001)

        console.print("\n[bold cyan]Base Model Results:[/bold cyan]")
        console.print(f"  Perplexity: {base_metrics.perplexity:.2f}")
        console.print(f"  ROUGE-L: {base_metrics.rougeL_f1:.3f}")
        console.print(f"  Response Quality: {base_metrics.response_completeness:.1%}\n")

    # Evaluate fine-tuned model
    console.print("[bold]Evaluating Fine-Tuned Model[/bold]\n")
    model, tokenizer = load_model_for_evaluation(model_path, args.device)
    evaluator = ModelEvaluator(model, tokenizer, args.device)
    finetuned_metrics = evaluator.evaluate_comprehensive(eval_dataset, cost_per_1k_tokens=0.0001)

    console.print("\n[bold cyan]Fine-Tuned Model Results:[/bold cyan]")
    console.print(f"  Perplexity: {finetuned_metrics.perplexity:.2f}")
    console.print(f"  ROUGE-L: {finetuned_metrics.rougeL_f1:.3f}")
    console.print(f"  Response Quality: {finetuned_metrics.response_completeness:.1%}\n")

    # Display comparison
    if base_metrics:
        console.print("\n")
        comparison_table = compare_models_table(base_metrics, finetuned_metrics)
        console.print(comparison_table)

    # Save report
    console.print("\n[bold]Saving Evaluation Report...[/bold]")
    save_evaluation_report(base_metrics, finetuned_metrics)

    console.print("\n[bold green]✓ Evaluation complete![/bold green]\n")


if __name__ == "__main__":
    main()
