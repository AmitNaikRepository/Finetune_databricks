"""
Model Registry - Enterprise ML Model Lifecycle Management

This module demonstrates production model versioning and management patterns.

In production environments, this would integrate with:
- MLflow Model Registry
- Databricks Model Serving
- Model governance and approval workflows
- A/B testing infrastructure

For this portfolio demo, we implement similar functionality with SQLite.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from rich.console import Console
from rich.table import Table

console = Console()


class ModelStatus(str, Enum):
    """Model lifecycle stages"""
    STAGING = "staging"          # Under testing/validation
    PRODUCTION = "production"    # Deployed to production
    ARCHIVED = "archived"        # Deprecated/old version
    FAILED = "failed"           # Training or validation failed


@dataclass
class ModelMetadata:
    """
    Model metadata for tracking

    This captures all important information about a trained model
    """
    version: str
    model_path: str
    model_name: str
    base_model: str
    training_date: str
    status: ModelStatus

    # Performance metrics
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    eval_accuracy: Optional[float] = None

    # Training info
    training_samples: Optional[int] = None
    training_duration: Optional[float] = None
    lora_rank: Optional[int] = None

    # Additional metadata as JSON
    extra_metadata: Optional[Dict] = None

    # Deployment info
    deployment_date: Optional[str] = None
    deployed_by: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['status'] = data['status'].value
        if data['extra_metadata']:
            data['extra_metadata'] = json.dumps(data['extra_metadata'])
        return data


class ModelRegistry:
    """
    Model Registry for tracking and versioning trained models

    Features:
    - Version management (semantic versioning)
    - Status tracking (staging → production → archived)
    - Metrics storage
    - Model comparison
    - Deployment tracking

    In production, this would use:
    - MLflow Model Registry
    - Databricks Model Serving
    - Cloud storage (S3, ADLS, GCS)
    """

    def __init__(self, db_path: str = "monitoring/metrics.db"):
        """
        Initialize model registry

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._initialize_database()

    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """
        Create database tables if they don't exist

        PRODUCTION NOTE:
        In enterprise settings, this would be a proper database (PostgreSQL, etc.)
        with:
        - Proper schema migrations (Alembic)
        - Replication for high availability
        - Backup strategies
        - Access control
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Model registry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    training_date TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,

                    -- Performance metrics
                    train_loss REAL,
                    eval_loss REAL,
                    eval_perplexity REAL,
                    eval_accuracy REAL,

                    -- Training metadata
                    training_samples INTEGER,
                    training_duration REAL,
                    lora_rank INTEGER,

                    -- Extra metadata as JSON
                    extra_metadata TEXT,

                    -- Deployment tracking
                    deployment_date TIMESTAMP,
                    deployed_by TEXT,

                    -- Audit fields
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Deployment history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deployment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    environment TEXT NOT NULL,  -- staging, production, etc.
                    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployed_by TEXT,
                    status TEXT,  -- success, failed, rolled_back
                    notes TEXT,
                    FOREIGN KEY (model_version) REFERENCES model_registry(version)
                )
            """)

            # Model comparison table (for A/B testing)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_a_version TEXT NOT NULL,
                    model_b_version TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    model_a_value REAL,
                    model_b_value REAL,
                    winner TEXT,  -- 'model_a', 'model_b', or 'tie'
                    compared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_a_version) REFERENCES model_registry(version),
                    FOREIGN KEY (model_b_version) REFERENCES model_registry(version)
                )
            """)

            conn.commit()

    def register_model(
        self,
        model_path: str,
        model_name: str,
        metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        status: ModelStatus = ModelStatus.STAGING
    ) -> str:
        """
        Register a new model version

        Args:
            model_path: Path to saved model checkpoint
            model_name: Human-readable model name
            metrics: Dict of evaluation metrics
            metadata: Additional metadata
            status: Initial status (default: staging)

        Returns:
            Model version string

        Example:
            >>> registry = ModelRegistry()
            >>> version = registry.register_model(
            ...     model_path="./models/checkpoints/run_123",
            ...     model_name="customer-support-v1",
            ...     metrics={"eval_loss": 0.45, "eval_accuracy": 0.89},
            ...     metadata={"base_model": "distilgpt2", "lora_rank": 8}
            ... )
            >>> print(f"Registered as {version}")
        """
        # Generate version number
        version = self._generate_version(model_name)

        # Prepare metadata
        metrics = metrics or {}
        metadata = metadata or {}

        model_metadata = ModelMetadata(
            version=version,
            model_path=model_path,
            model_name=model_name,
            base_model=metadata.get("base_model", "unknown"),
            training_date=datetime.now().isoformat(),
            status=status,
            train_loss=metrics.get("train_loss"),
            eval_loss=metrics.get("eval_loss"),
            eval_perplexity=metrics.get("eval_perplexity"),
            eval_accuracy=metrics.get("eval_accuracy"),
            training_samples=metadata.get("training_samples"),
            training_duration=metadata.get("training_duration"),
            lora_rank=metadata.get("lora_config", {}).get("rank"),
            extra_metadata=metadata
        )

        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            data = model_metadata.to_dict()

            cursor.execute("""
                INSERT INTO model_registry (
                    version, model_name, model_path, base_model, training_date, status,
                    train_loss, eval_loss, eval_perplexity, eval_accuracy,
                    training_samples, training_duration, lora_rank, extra_metadata
                )
                VALUES (
                    :version, :model_name, :model_path, :base_model, :training_date, :status,
                    :train_loss, :eval_loss, :eval_perplexity, :eval_accuracy,
                    :training_samples, :training_duration, :lora_rank, :extra_metadata
                )
            """, data)

            conn.commit()

        console.print(f"[green]✓ Model registered: {version} ({status.value})[/green]")
        return version

    def _generate_version(self, model_name: str) -> str:
        """
        Generate next version number

        Version format: {model_name}-v{number}
        Example: customer-support-v1, customer-support-v2, etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version FROM model_registry
                WHERE model_name = ?
                ORDER BY id DESC
                LIMIT 1
            """, (model_name,))

            result = cursor.fetchone()

            if result:
                # Extract version number and increment
                last_version = result[0]
                try:
                    version_num = int(last_version.split('-v')[-1]) + 1
                except (ValueError, IndexError):
                    version_num = 1
            else:
                version_num = 1

        return f"{model_name}-v{version_num}"

    def update_status(
        self,
        version: str,
        new_status: ModelStatus,
        deployed_by: Optional[str] = None
    ):
        """
        Update model status

        Example workflow:
        1. Model trained → STAGING
        2. Validation passed → PRODUCTION
        3. New model deployed → ARCHIVED

        Args:
            version: Model version to update
            new_status: New status
            deployed_by: Who performed the status change
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            update_data = {
                "version": version,
                "status": new_status.value,
                "updated_at": datetime.now().isoformat()
            }

            if new_status == ModelStatus.PRODUCTION:
                update_data["deployment_date"] = datetime.now().isoformat()
                if deployed_by:
                    update_data["deployed_by"] = deployed_by

            cursor.execute("""
                UPDATE model_registry
                SET status = :status,
                    updated_at = :updated_at,
                    deployment_date = COALESCE(:deployment_date, deployment_date),
                    deployed_by = COALESCE(:deployed_by, deployed_by)
                WHERE version = :version
            """, {
                "version": version,
                "status": new_status.value,
                "updated_at": datetime.now().isoformat(),
                "deployment_date": update_data.get("deployment_date"),
                "deployed_by": update_data.get("deployed_by")
            })

            conn.commit()

        console.print(f"[cyan]Model {version} status updated: {new_status.value}[/cyan]")

    def get_model(self, version: str) -> Optional[Dict]:
        """
        Retrieve model metadata by version

        Args:
            version: Model version

        Returns:
            Dict with model metadata or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM model_registry
                WHERE version = ?
            """, (version,))

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_latest_model(
        self,
        model_name: Optional[str] = None,
        status: Optional[ModelStatus] = None
    ) -> Optional[Dict]:
        """
        Get latest model, optionally filtered by name and status

        Args:
            model_name: Filter by model name
            status: Filter by status (e.g., PRODUCTION)

        Returns:
            Dict with model metadata or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM model_registry WHERE 1=1"
            params = []

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY id DESC LIMIT 1"

            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_models(
        self,
        model_name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        List models with optional filters

        Args:
            model_name: Filter by model name
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of model metadata dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM model_registry WHERE 1=1"
            params = []

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += f" ORDER BY id DESC LIMIT {limit}"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def compare_models(self, version_a: str, version_b: str) -> Table:
        """
        Compare two model versions

        Args:
            version_a: First model version
            version_b: Second model version

        Returns:
            Rich Table with comparison
        """
        model_a = self.get_model(version_a)
        model_b = self.get_model(version_b)

        if not model_a or not model_b:
            raise ValueError("One or both models not found")

        table = Table(title=f"Model Comparison: {version_a} vs {version_b}")
        table.add_column("Metric", style="cyan")
        table.add_column(version_a, justify="right", style="green")
        table.add_column(version_b, justify="right", style="yellow")
        table.add_column("Better", justify="center")

        metrics = [
            ("Train Loss", "train_loss", "lower"),
            ("Eval Loss", "eval_loss", "lower"),
            ("Eval Perplexity", "eval_perplexity", "lower"),
            ("Eval Accuracy", "eval_accuracy", "higher"),
            ("Training Samples", "training_samples", None),
            ("LoRA Rank", "lora_rank", None),
        ]

        for metric_name, metric_key, better_direction in metrics:
            val_a = model_a.get(metric_key)
            val_b = model_b.get(metric_key)

            if val_a is None and val_b is None:
                continue

            str_a = f"{val_a:.4f}" if isinstance(val_a, float) else str(val_a) if val_a else "N/A"
            str_b = f"{val_b:.4f}" if isinstance(val_b, float) else str(val_b) if val_b else "N/A"

            # Determine winner
            winner = ""
            if val_a and val_b and better_direction:
                if better_direction == "lower" and val_a < val_b:
                    winner = "←"
                elif better_direction == "lower" and val_b < val_a:
                    winner = "→"
                elif better_direction == "higher" and val_a > val_b:
                    winner = "←"
                elif better_direction == "higher" and val_b > val_a:
                    winner = "→"

            table.add_row(metric_name, str_a, str_b, winner)

        return table

    def display_models(self, models: List[Dict]):
        """Display models in a nice table"""
        if not models:
            console.print("[yellow]No models found[/yellow]")
            return

        table = Table(title="Model Registry")
        table.add_column("Version", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Eval Loss", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Training Date")

        for model in models:
            table.add_row(
                model['version'],
                model['status'],
                f"{model['eval_loss']:.4f}" if model.get('eval_loss') else "N/A",
                f"{model['eval_accuracy']:.2%}" if model.get('eval_accuracy') else "N/A",
                model['training_date'][:10] if model.get('training_date') else "N/A"
            )

        console.print(table)


# Example usage
if __name__ == "__main__":
    """
    Test model registry functionality

    Run: python models/registry.py
    """
    console.print("\n[bold cyan]═══ Model Registry Demo ═══[/bold cyan]\n")

    # Initialize registry
    registry = ModelRegistry()

    # Register some demo models
    console.print("[bold]Registering demo models...[/bold]\n")

    for i in range(3):
        version = registry.register_model(
            model_path=f"./models/checkpoints/demo_run_{i}",
            model_name="customer-support",
            metrics={
                "train_loss": 0.5 - (i * 0.05),
                "eval_loss": 0.6 - (i * 0.05),
                "eval_accuracy": 0.75 + (i * 0.05)
            },
            metadata={
                "base_model": "distilgpt2",
                "training_samples": 1000,
                "lora_config": {"rank": 8}
            }
        )

    # List all models
    console.print("\n[bold]All Registered Models:[/bold]")
    models = registry.list_models(limit=10)
    registry.display_models(models)

    # Promote latest to production
    latest = registry.get_latest_model()
    if latest:
        console.print(f"\n[bold]Promoting {latest['version']} to production...[/bold]")
        registry.update_status(latest['version'], ModelStatus.PRODUCTION, deployed_by="demo_user")

    # Compare models
    if len(models) >= 2:
        console.print("\n[bold]Model Comparison:[/bold]")
        comparison = registry.compare_models(models[0]['version'], models[1]['version'])
        console.print(comparison)

    console.print("\n[green]✓ Model registry demo complete![/green]\n")
