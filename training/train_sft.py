#!/usr/bin/env python3
"""
Enterprise Supervised Fine-Tuning (SFT) Pipeline

This script demonstrates production-grade LLM fine-tuning with:
- Parameter-efficient fine-tuning (LoRA)
- Experiment tracking (MLflow/W&B)
- Model versioning and registry
- Comprehensive logging and monitoring
- Best practices for reproducibility

Usage:
    python training/train_sft.py --config training/config.yaml
    python training/train_sft.py --config training/config.yaml --wandb
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import PeftModel, PeftConfig
import mlflow
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

from data.data_loader import DatabricksConnector, DataPreprocessor
from training.lora_config import LoRAConfigBuilder, apply_lora
from models.registry import ModelRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
console = Console()


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer with Enterprise Features

    Features:
    - Automatic experiment tracking
    - Model registry integration
    - Checkpoint management
    - Performance monitoring
    - Cost tracking
    """

    def __init__(self, config_path: str, use_wandb: bool = False):
        """
        Initialize SFT trainer

        Args:
            config_path: Path to YAML configuration file
            use_wandb: Whether to use Weights & Biases instead of MLflow
        """
        self.config_path = config_path
        self.use_wandb = use_wandb
        self.config = self._load_config()
        self.device = self._setup_device()
        self.model_registry = ModelRegistry()

        # Set random seeds for reproducibility
        self._set_seeds()

        console.print(Panel.fit(
            "[bold cyan]Enterprise LLM Fine-Tuning Pipeline[/bold cyan]\n"
            f"Model: {self.config['model']['base_model']}\n"
            f"Device: {self.device}\n"
            f"Experiment: {self.config['experiment']['experiment_name']}",
            border_style="cyan"
        ))

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_device(self) -> str:
        """Setup computation device (CPU/GPU)"""
        if torch.cuda.is_available():
            device = "cuda"
            console.print(f"[green]✓ GPU available: {torch.cuda.get_device_name(0)}[/green]")
        elif torch.backends.mps.is_available():
            device = "mps"
            console.print("[green]✓ Apple Silicon GPU available[/green]")
        else:
            device = "cpu"
            console.print("[yellow]⚠ Running on CPU (training will be slow)[/yellow]")

        return device

    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['training']['seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self):
        """
        Load and preprocess training data

        In production, this connects to Databricks and loads from Delta Lake.
        For demo, loads simulated data.
        """
        console.print("\n[bold]Step 1: Loading Training Data[/bold]")

        # Connect to data source
        connector = DatabricksConnector(use_simulation=True)

        # Load dataset
        dataset = connector.load_training_data(
            limit=self.config['data'].get('max_samples')
        )

        # Preprocess and format
        preprocessor = DataPreprocessor(
            format=self.config['data']['instruction_format']
        )

        prepared_data = preprocessor.prepare_dataset(
            dataset,
            train_split=self.config['data']['train_split']
        )

        self.train_dataset = prepared_data['train']
        self.eval_dataset = prepared_data['validation']

        # Display sample
        console.print("\n[bold]Sample Training Example:[/bold]")
        console.print(Panel(
            self.train_dataset[0]['text'][:400] + "...",
            border_style="blue",
            title="Formatted Instruction"
        ))

        return prepared_data

    def setup_model_and_tokenizer(self):
        """
        Load base model and tokenizer, apply LoRA

        This demonstrates:
        1. Loading pretrained models from HuggingFace
        2. Configuring tokenizer with special tokens
        3. Applying parameter-efficient fine-tuning (LoRA)
        """
        console.print("\n[bold]Step 2: Setting Up Model & Tokenizer[/bold]")

        model_name = self.config['model']['base_model']

        # Load tokenizer
        console.print(f"[cyan]Loading tokenizer: {model_name}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 doesn't have a pad token, so we add one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model
        console.print(f"[cyan]Loading base model: {model_name}[/cyan]")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float16 for GPU
            device_map=None,  # Don't auto-assign to GPU yet
            trust_remote_code=self.config['model'].get('trust_remote_code', False)
        )

        # Display model info
        total_params = sum(p.numel() for p in self.base_model.parameters())
        console.print(f"[green]✓ Base model loaded: {total_params:,} parameters[/green]")

        # Apply LoRA
        console.print("\n[cyan]Applying LoRA (Parameter-Efficient Fine-Tuning)...[/cyan]")
        lora_config_builder = LoRAConfigBuilder.from_yaml(self.config)
        lora_config = lora_config_builder.build()

        self.model = apply_lora(self.base_model, lora_config)

        # Show parameter efficiency
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        console.print(f"[green]✓ LoRA applied: {trainable_params:,} trainable parameters[/green]")
        console.print(f"[green]  Efficiency: {100 * trainable_params / total_params:.2f}% of original[/green]")

    def tokenize_dataset(self):
        """
        Tokenize datasets for training

        This prepares the text data for model consumption
        """
        console.print("\n[bold]Step 3: Tokenizing Datasets[/bold]")

        max_length = self.config['model']['model_max_length']

        def tokenize_function(examples):
            """Tokenize text and prepare labels"""
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )

        # Tokenize
        console.print("[cyan]Tokenizing training data...[/cyan]")
        self.tokenized_train = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train dataset"
        )

        console.print("[cyan]Tokenizing validation data...[/cyan]")
        self.tokenized_eval = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )

        console.print(f"[green]✓ Tokenization complete[/green]")
        console.print(f"  Train samples: {len(self.tokenized_train)}")
        console.print(f"  Eval samples: {len(self.tokenized_eval)}")

    def setup_training_args(self) -> TrainingArguments:
        """
        Configure training arguments

        This shows enterprise best practices for training configuration
        """
        train_config = self.config['training']
        exp_config = self.config['experiment']

        # Determine output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/checkpoints/{exp_config['run_name']}_{timestamp}"

        # Setup experiment tracking
        if self.use_wandb:
            report_to = ["wandb"]
            os.environ["WANDB_PROJECT"] = exp_config['project_name']
        else:
            report_to = ["mlflow"]
            # MLflow will be initialized separately

        training_args = TrainingArguments(
            # Output
            output_dir=output_dir,
            overwrite_output_dir=True,

            # Training hyperparameters
            num_train_epochs=train_config['num_train_epochs'],
            per_device_train_batch_size=train_config['per_device_train_batch_size'],
            per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],

            # Optimization
            learning_rate=train_config['learning_rate'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            warmup_steps=train_config['warmup_steps'],
            optim=train_config['optim'],
            weight_decay=train_config['weight_decay'],
            max_grad_norm=train_config['max_grad_norm'],

            # Evaluation
            evaluation_strategy=train_config['evaluation_strategy'],
            eval_steps=train_config['eval_steps'],
            save_strategy=train_config['save_strategy'],
            save_steps=train_config['save_steps'],
            save_total_limit=train_config['save_total_limit'],
            load_best_model_at_end=train_config['load_best_model_at_end'],
            metric_for_best_model=train_config['metric_for_best_model'],

            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=train_config['logging_steps'],
            logging_first_step=train_config['logging_first_step'],
            report_to=report_to,

            # Performance
            dataloader_num_workers=train_config.get('dataloader_num_workers', 0),
            fp16=train_config.get('fp16', False),
            bf16=train_config.get('bf16', False),

            # Reproducibility
            seed=train_config['seed'],
            data_seed=train_config.get('data_seed', 42),

            # Misc
            remove_unused_columns=True,
            push_to_hub=False,
        )

        self.output_dir = output_dir
        return training_args

    def train(self):
        """
        Execute training loop with experiment tracking

        This is the main training execution with full monitoring
        """
        console.print("\n[bold]Step 4: Training Model[/bold]")

        # Setup training arguments
        training_args = self.setup_training_args()

        # Initialize experiment tracking
        if not self.use_wandb:
            mlflow.set_tracking_uri("sqlite:///monitoring/mlflow.db")
            mlflow.set_experiment(self.config['experiment']['experiment_name'])

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_eval,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Start training
        console.print("\n[bold green]Starting training...[/bold green]")

        with mlflow.start_run(run_name=self.config['experiment']['run_name']) as run:
            # Log configuration
            mlflow.log_params({
                "model_name": self.config['model']['base_model'],
                "lora_rank": self.config['lora']['rank'],
                "lora_alpha": self.config['lora']['alpha'],
                "learning_rate": self.config['training']['learning_rate'],
                "batch_size": self.config['training']['per_device_train_batch_size'],
                "num_epochs": self.config['training']['num_train_epochs'],
            })

            # Train
            train_result = trainer.train()

            # Log results
            metrics = train_result.metrics
            mlflow.log_metrics({
                "train_loss": metrics['train_loss'],
                "train_runtime": metrics['train_runtime'],
                "train_samples_per_second": metrics['train_samples_per_second'],
            })

            # Save model
            console.print("\n[cyan]Saving fine-tuned model...[/cyan]")
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

            # Register in model registry
            model_version = self.model_registry.register_model(
                model_path=self.output_dir,
                model_name=self.config['experiment']['run_name'],
                metrics={
                    "train_loss": metrics['train_loss'],
                    "eval_loss": trainer.evaluate()['eval_loss']
                },
                metadata={
                    "base_model": self.config['model']['base_model'],
                    "lora_config": self.config['lora'],
                    "training_samples": len(self.train_dataset),
                }
            )

            console.print(f"[green]✓ Model saved to: {self.output_dir}[/green]")
            console.print(f"[green]✓ Registered as version: {model_version}[/green]")

        self.display_training_summary(train_result.metrics, trainer.evaluate())

    def display_training_summary(self, train_metrics: Dict, eval_metrics: Dict):
        """Display training summary in a nice table"""
        console.print("\n[bold]Training Summary[/bold]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Final Train Loss", f"{train_metrics['train_loss']:.4f}")
        table.add_row("Final Eval Loss", f"{eval_metrics['eval_loss']:.4f}")
        table.add_row("Training Time", f"{train_metrics['train_runtime']:.2f}s")
        table.add_row("Samples/Second", f"{train_metrics['train_samples_per_second']:.2f}")
        table.add_row("Output Directory", self.output_dir)

        console.print(table)


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Enterprise LLM Fine-Tuning Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases instead of MLflow"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Setup only, don't train"
    )

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = SFTTrainer(config_path=args.config, use_wandb=args.wandb)

        # Load data
        trainer.load_data()

        # Setup model
        trainer.setup_model_and_tokenizer()

        # Tokenize
        trainer.tokenize_dataset()

        if not args.dry_run:
            # Train
            trainer.train()

            console.print("\n[bold green]✓ Training completed successfully![/bold green]")
        else:
            console.print("\n[yellow]Dry run completed (training skipped)[/yellow]")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
