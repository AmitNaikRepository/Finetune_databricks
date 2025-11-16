"""
Experiment Tracking and Monitoring

This module provides experiment tracking capabilities for:
- Training runs
- Model evaluations
- Inference metrics
- Cost tracking

Supports both MLflow and custom SQLite tracking.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class TrainingRun:
    """Training run metadata"""
    run_id: str
    experiment_name: str
    model_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Hyperparameters
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    lora_rank: Optional[int] = None

    # Metrics
    final_train_loss: Optional[float] = None
    final_eval_loss: Optional[float] = None
    best_eval_loss: Optional[float] = None

    # Status
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None

    # Additional metadata
    metadata: Optional[Dict] = None


@dataclass
class InferenceLog:
    """Inference request log"""
    log_id: int
    timestamp: str
    model_version: str
    query: str
    response: str
    latency_ms: float
    tokens_generated: int
    estimated_cost: float


class ExperimentTracker:
    """
    Experiment tracking system

    Tracks:
    - Training runs with hyperparameters and metrics
    - Model evaluations
    - Inference requests
    - Cost metrics

    In production, this would integrate with:
    - MLflow
    - Weights & Biases
    - TensorBoard
    - Custom dashboards (Grafana, etc.)
    """

    def __init__(self, db_path: str = "monitoring/metrics.db"):
        """
        Initialize experiment tracker

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
        """Create database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Training runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration_seconds REAL,

                    -- Hyperparameters
                    learning_rate REAL,
                    batch_size INTEGER,
                    num_epochs INTEGER,
                    lora_rank INTEGER,

                    -- Metrics
                    final_train_loss REAL,
                    final_eval_loss REAL,
                    best_eval_loss REAL,

                    -- Status
                    status TEXT DEFAULT 'running',
                    error_message TEXT,

                    -- Metadata as JSON
                    metadata TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Inference logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tokens_generated INTEGER,
                    estimated_cost REAL
                )
            """)

            # Daily metrics aggregation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    total_inferences INTEGER DEFAULT 0,
                    avg_latency_ms REAL,
                    total_tokens_generated INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    unique_model_versions INTEGER DEFAULT 0
                )
            """)

            # Evaluation results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Metrics
                    perplexity REAL,
                    rouge1_f1 REAL,
                    rouge2_f1 REAL,
                    rougeL_f1 REAL,
                    avg_latency_ms REAL,
                    throughput_tokens_per_sec REAL,

                    -- Dataset info
                    eval_dataset_size INTEGER,
                    eval_dataset_name TEXT,

                    FOREIGN KEY (model_version) REFERENCES model_registry(version)
                )
            """)

            conn.commit()

    def start_training_run(
        self,
        run_id: str,
        experiment_name: str,
        model_name: str,
        hyperparameters: Dict
    ) -> TrainingRun:
        """
        Start tracking a training run

        Args:
            run_id: Unique run identifier
            experiment_name: Experiment name
            model_name: Model being trained
            hyperparameters: Dict of hyperparameters

        Returns:
            TrainingRun object
        """
        run = TrainingRun(
            run_id=run_id,
            experiment_name=experiment_name,
            model_name=model_name,
            start_time=datetime.now().isoformat(),
            learning_rate=hyperparameters.get('learning_rate'),
            batch_size=hyperparameters.get('batch_size'),
            num_epochs=hyperparameters.get('num_epochs'),
            lora_rank=hyperparameters.get('lora_rank'),
            metadata=hyperparameters
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            data = asdict(run)
            data['metadata'] = json.dumps(data['metadata']) if data['metadata'] else None

            cursor.execute("""
                INSERT INTO training_runs (
                    run_id, experiment_name, model_name, start_time,
                    learning_rate, batch_size, num_epochs, lora_rank, metadata
                )
                VALUES (
                    :run_id, :experiment_name, :model_name, :start_time,
                    :learning_rate, :batch_size, :num_epochs, :lora_rank, :metadata
                )
            """, data)

            conn.commit()

        console.print(f"[green]✓ Started tracking run: {run_id}[/green]")
        return run

    def end_training_run(
        self,
        run_id: str,
        final_metrics: Dict,
        status: str = "completed",
        error_message: Optional[str] = None
    ):
        """
        Complete a training run

        Args:
            run_id: Run identifier
            final_metrics: Final metrics from training
            status: "completed" or "failed"
            error_message: Error message if failed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get start time to calculate duration
            cursor.execute("SELECT start_time FROM training_runs WHERE run_id = ?", (run_id,))
            result = cursor.fetchone()

            if result:
                start_time = datetime.fromisoformat(result[0])
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                cursor.execute("""
                    UPDATE training_runs
                    SET end_time = ?,
                        duration_seconds = ?,
                        final_train_loss = ?,
                        final_eval_loss = ?,
                        best_eval_loss = ?,
                        status = ?,
                        error_message = ?
                    WHERE run_id = ?
                """, (
                    end_time.isoformat(),
                    duration,
                    final_metrics.get('train_loss'),
                    final_metrics.get('eval_loss'),
                    final_metrics.get('best_eval_loss'),
                    status,
                    error_message,
                    run_id
                ))

                conn.commit()

                console.print(f"[green]✓ Training run completed: {run_id} ({status})[/green]")

    def log_inference(
        self,
        model_version: str,
        query: str,
        response: str,
        latency_ms: float,
        tokens_generated: int,
        estimated_cost: float
    ):
        """
        Log an inference request

        Args:
            model_version: Model version used
            query: Input query
            response: Generated response
            latency_ms: Response latency
            tokens_generated: Number of tokens generated
            estimated_cost: Estimated cost of inference
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO inference_logs (
                    timestamp, model_version, query, response,
                    latency_ms, tokens_generated, estimated_cost
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model_version,
                query,
                response,
                latency_ms,
                tokens_generated,
                estimated_cost
            ))

            conn.commit()

    def log_evaluation(
        self,
        model_version: str,
        metrics: Dict,
        eval_dataset_name: str = "eval_dataset",
        eval_dataset_size: int = 0
    ):
        """
        Log evaluation results

        Args:
            model_version: Model version evaluated
            metrics: Evaluation metrics
            eval_dataset_name: Name of evaluation dataset
            eval_dataset_size: Size of evaluation dataset
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO evaluation_results (
                    model_version, evaluation_date,
                    perplexity, rouge1_f1, rouge2_f1, rougeL_f1,
                    avg_latency_ms, throughput_tokens_per_sec,
                    eval_dataset_size, eval_dataset_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_version,
                datetime.now().isoformat(),
                metrics.get('perplexity'),
                metrics.get('rouge1_f1'),
                metrics.get('rouge2_f1'),
                metrics.get('rougeL_f1'),
                metrics.get('avg_latency_ms'),
                metrics.get('throughput_tokens_per_sec'),
                eval_dataset_size,
                eval_dataset_name
            ))

            conn.commit()

        console.print(f"[green]✓ Evaluation logged for {model_version}[/green]")

    def get_training_runs(
        self,
        experiment_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get training runs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM training_runs WHERE 1=1"
            params = []

            if experiment_name:
                query += " AND experiment_name = ?"
                params.append(experiment_name)

            query += f" ORDER BY start_time DESC LIMIT {limit}"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_inference_stats(self, days: int = 7) -> Dict:
        """
        Get inference statistics

        Args:
            days: Number of days to look back

        Returns:
            Dict with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    AVG(latency_ms) as avg_latency,
                    SUM(tokens_generated) as total_tokens,
                    SUM(estimated_cost) as total_cost,
                    COUNT(DISTINCT model_version) as unique_models
                FROM inference_logs
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
            """, (days,))

            result = cursor.fetchone()

            return {
                "total_requests": result[0] or 0,
                "avg_latency_ms": round(result[1], 2) if result[1] else 0,
                "total_tokens": result[2] or 0,
                "total_cost": round(result[3], 4) if result[3] else 0,
                "unique_models": result[4] or 0
            }

    def display_training_runs(self, runs: List[Dict]):
        """Display training runs in a table"""
        if not runs:
            console.print("[yellow]No training runs found[/yellow]")
            return

        table = Table(title="Training Runs", show_header=True)

        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("Status", style="magenta")
        table.add_column("Eval Loss", justify="right")
        table.add_column("Duration (s)", justify="right")
        table.add_column("Started")

        for run in runs:
            table.add_row(
                run['run_id'][:12] + "...",
                run['model_name'],
                run['status'],
                f"{run['final_eval_loss']:.4f}" if run.get('final_eval_loss') else "N/A",
                f"{run['duration_seconds']:.0f}" if run.get('duration_seconds') else "Running",
                run['start_time'][:19] if run.get('start_time') else "N/A"
            )

        console.print(table)


# Example usage
if __name__ == "__main__":
    """
    Test experiment tracking

    Run: python monitoring/track_experiments.py
    """
    console.print("\n[bold cyan]═══ Experiment Tracking Demo ═══[/bold cyan]\n")

    tracker = ExperimentTracker()

    # Demo: Start a training run
    run = tracker.start_training_run(
        run_id="demo_run_001",
        experiment_name="customer-support",
        model_name="distilgpt2",
        hyperparameters={
            "learning_rate": 2e-4,
            "batch_size": 4,
            "num_epochs": 3,
            "lora_rank": 8
        }
    )

    # Demo: End the run
    tracker.end_training_run(
        run_id="demo_run_001",
        final_metrics={
            "train_loss": 0.45,
            "eval_loss": 0.52,
            "best_eval_loss": 0.48
        },
        status="completed"
    )

    # Demo: Log some inferences
    for i in range(5):
        tracker.log_inference(
            model_version="customer-support-v1",
            query=f"Test query {i+1}",
            response=f"Test response {i+1}",
            latency_ms=250 + i * 10,
            tokens_generated=50 + i * 5,
            estimated_cost=0.00005
        )

    # Show training runs
    console.print("\n[bold]Training Runs:[/bold]")
    runs = tracker.get_training_runs(limit=5)
    tracker.display_training_runs(runs)

    # Show inference stats
    console.print("\n[bold]Inference Statistics (Last 7 Days):[/bold]")
    stats = tracker.get_inference_stats(days=7)
    for key, value in stats.items():
        console.print(f"  {key}: {value}")

    console.print("\n[green]✓ Experiment tracking demo complete![/green]\n")
