"""
Databricks Data Loader - Enterprise Integration Pattern

This module demonstrates how to integrate with Databricks in production.
For portfolio demo purposes, it simulates the connection and loads local data.

PRODUCTION IMPLEMENTATION WOULD USE:
- databricks-sql-connector for SQL queries
- Delta Lake for data versioning
- Unity Catalog for data governance
- Structured streaming for real-time data
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class DatabricksConnector:
    """
    Enterprise Data Connector with Databricks Integration Pattern

    In a production environment, this would:
    1. Connect to Databricks SQL Warehouse
    2. Query Unity Catalog tables
    3. Handle authentication via OAuth or PAT
    4. Implement connection pooling
    5. Support Delta Lake time travel
    6. Enable incremental data loading

    For this portfolio demo, it simulates these patterns using local data.
    """

    def __init__(
        self,
        workspace_url: Optional[str] = None,
        token: Optional[str] = None,
        catalog: str = "main",
        schema: str = "ml_training",
        use_simulation: bool = True
    ):
        """
        Initialize Databricks connector

        Args:
            workspace_url: Databricks workspace URL (e.g., https://xxx.cloud.databricks.com)
            token: Personal Access Token or OAuth token
            catalog: Unity Catalog name
            schema: Schema/database name
            use_simulation: If True, use local simulation instead of real connection

        Production Example:
            connector = DatabricksConnector(
                workspace_url=os.getenv("DATABRICKS_WORKSPACE_URL"),
                token=os.getenv("DATABRICKS_TOKEN"),
                catalog="production",
                schema="customer_support",
                use_simulation=False
            )
        """
        self.workspace_url = workspace_url or os.getenv("DATABRICKS_WORKSPACE_URL", "SIMULATED")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        self.catalog = catalog
        self.schema = schema
        self.use_simulation = use_simulation or os.getenv("USE_SIMULATED_DATA", "true").lower() == "true"

        # Connection object (would be real in production)
        self.connection = None

        if not self.use_simulation:
            self._connect_to_databricks()
        else:
            console.print("[yellow]Running in SIMULATION mode - using local data[/yellow]")
            console.print(f"[dim]To connect to real Databricks, set USE_SIMULATED_DATA=false[/dim]\n")

    def _connect_to_databricks(self):
        """
        Establish connection to Databricks SQL Warehouse

        PRODUCTION CODE WOULD BE:
        ```python
        from databricks import sql

        self.connection = sql.connect(
            server_hostname=self.workspace_url.replace("https://", ""),
            http_path="/sql/1.0/warehouses/xxxxx",  # SQL Warehouse path
            access_token=self.token,
            catalog=self.catalog,
            schema=self.schema
        )
        console.print(f"[green]✓ Connected to Databricks: {self.workspace_url}[/green]")
        ```
        """
        if not self.workspace_url or self.workspace_url == "SIMULATED":
            raise ValueError("DATABRICKS_WORKSPACE_URL not set. Use simulation mode or provide credentials.")

        # In production, implement actual connection logic here
        console.print("[green]✓ Connected to Databricks (production mode)[/green]")

    def load_training_data(
        self,
        table_name: str = "customer_support_qa",
        limit: Optional[int] = None,
        filters: Optional[Dict[str, any]] = None
    ) -> Dataset:
        """
        Load training data from Databricks table or local simulation

        Args:
            table_name: Name of the table in Unity Catalog
            limit: Maximum number of rows to load
            filters: Dict of column filters (e.g., {"date": "2024-01-01"})

        Returns:
            HuggingFace Dataset object

        PRODUCTION SQL QUERY WOULD BE:
        ```sql
        SELECT
            instruction,
            response,
            context,
            created_at,
            metadata
        FROM {catalog}.{schema}.{table_name}
        WHERE created_at >= '2024-01-01'
        AND quality_score >= 0.8
        LIMIT {limit}
        ```
        """
        if self.use_simulation:
            return self._load_simulated_data(limit=limit)
        else:
            return self._load_from_databricks(table_name, limit, filters)

    def _load_simulated_data(self, limit: Optional[int] = None) -> Dataset:
        """
        Load simulated customer support data for portfolio demo

        This demonstrates the data format expected from Databricks.
        In production, this would come from a real Delta Lake table.
        """
        console.print("[cyan]Loading simulated customer support training data...[/cyan]")

        # Check if we have local sample data
        sample_data_path = Path("data/raw/sample_customer_support.json")

        if sample_data_path.exists():
            console.print(f"[green]✓ Loading from local file: {sample_data_path}[/green]")
            with open(sample_data_path, 'r') as f:
                data = json.load(f)

            if limit:
                data = data[:limit]

            dataset = Dataset.from_list(data)
        else:
            # Generate synthetic data if no local file exists
            console.print("[yellow]No local data found. Generating synthetic dataset...[/yellow]")
            dataset = self._generate_synthetic_data(limit or 1000)

        console.print(f"[green]✓ Loaded {len(dataset)} training examples[/green]\n")
        return dataset

    def _load_from_databricks(
        self,
        table_name: str,
        limit: Optional[int],
        filters: Optional[Dict[str, any]]
    ) -> Dataset:
        """
        Load data from actual Databricks table

        PRODUCTION IMPLEMENTATION:
        ```python
        # Build query
        query = f'''
            SELECT *
            FROM {self.catalog}.{self.schema}.{table_name}
        '''

        # Add filters
        if filters:
            where_clauses = [f"{k} = '{v}'" for k, v in filters.items()]
            query += " WHERE " + " AND ".join(where_clauses)

        # Add limit
        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        # Convert to DataFrame then Dataset
        df = pd.DataFrame(rows, columns=columns)
        dataset = Dataset.from_pandas(df)

        return dataset
        ```
        """
        raise NotImplementedError("Production Databricks loading not implemented in demo")

    def _generate_synthetic_data(self, num_samples: int) -> Dataset:
        """
        Generate synthetic customer support Q&A data

        This simulates the type of data you'd get from a Databricks table
        with customer support conversations.
        """
        templates = [
            {
                "category": "password_reset",
                "instructions": [
                    "How do I reset my password?",
                    "I forgot my password, what should I do?",
                    "Can you help me reset my password?",
                    "My password isn't working, how do I change it?"
                ],
                "response": "To reset your password: 1) Go to the login page, 2) Click 'Forgot Password', 3) Enter your email address, 4) Check your inbox for a reset link, 5) Follow the link and create a new password. If you don't receive the email within 5 minutes, check your spam folder."
            },
            {
                "category": "order_tracking",
                "instructions": [
                    "Where is my order?",
                    "How can I track my package?",
                    "What's the status of order #{}?",
                    "When will my order arrive?"
                ],
                "response": "You can track your order by: 1) Logging into your account, 2) Going to 'Order History', 3) Clicking on the specific order, 4) Viewing real-time tracking information. Orders typically arrive within 3-5 business days. You'll receive tracking updates via email."
            },
            {
                "category": "refund",
                "instructions": [
                    "How do I get a refund?",
                    "I want to return my order",
                    "What's your refund policy?",
                    "Can I get my money back?"
                ],
                "response": "Our refund process: 1) Items can be returned within 30 days of purchase, 2) Go to 'My Orders' and select 'Return Item', 3) Print the prepaid return label, 4) Ship the item back, 5) Refunds are processed within 5-7 business days after we receive the return. Original shipping costs are non-refundable."
            },
            {
                "category": "account_access",
                "instructions": [
                    "I can't log into my account",
                    "My account is locked",
                    "How do I unlock my account?",
                    "Why won't my login work?"
                ],
                "response": "If you're having trouble accessing your account: 1) Ensure you're using the correct email address, 2) Try resetting your password, 3) Clear your browser cache and cookies, 4) If the account is locked due to multiple failed login attempts, wait 30 minutes and try again. For persistent issues, contact our support team."
            },
            {
                "category": "billing",
                "instructions": [
                    "Why was I charged twice?",
                    "I see an unexpected charge on my card",
                    "How can I view my billing history?",
                    "What's this charge for?"
                ],
                "response": "To review your billing: 1) Log into your account, 2) Navigate to 'Billing & Payments', 3) View all transactions and invoices. If you see duplicate or unexpected charges, they may be pending authorizations that will drop off in 3-5 business days. For confirmed duplicate charges, contact our billing department for immediate assistance."
            },
            {
                "category": "shipping",
                "instructions": [
                    "Do you ship internationally?",
                    "How much does shipping cost?",
                    "What shipping options are available?",
                    "Can I get expedited shipping?"
                ],
                "response": "We offer multiple shipping options: 1) Standard shipping (3-5 business days) - Free for orders over $50, 2) Express shipping (2-3 business days) - $9.99, 3) Overnight shipping (1 business day) - $24.99. International shipping is available to select countries with rates calculated at checkout. Tracking information is provided for all shipments."
            },
            {
                "category": "product_info",
                "instructions": [
                    "What are the product specifications?",
                    "Is this item in stock?",
                    "What colors/sizes are available?",
                    "Tell me more about this product"
                ],
                "response": "Product information can be found on the product page including: specifications, available sizes/colors, pricing, and customer reviews. Stock status is shown in real-time. If an item is out of stock, you can sign up for email notifications when it becomes available. For detailed technical specifications, check the 'Product Details' section."
            },
            {
                "category": "technical_support",
                "instructions": [
                    "The app isn't working",
                    "I'm getting an error message",
                    "How do I fix this technical issue?",
                    "The website is not loading"
                ],
                "response": "For technical issues, try these steps: 1) Refresh the page or restart the app, 2) Clear your browser cache and cookies, 3) Ensure you're using the latest version, 4) Try a different browser or device, 5) Check your internet connection. If the problem persists, please provide the error message and device information to our technical support team."
            }
        ]

        import random

        data = []
        for i in range(num_samples):
            template = random.choice(templates)
            instruction = random.choice(template["instructions"])

            # Add some variation to instructions
            if "{}" in instruction:
                instruction = instruction.format(random.randint(10000, 99999))

            data.append({
                "instruction": instruction,
                "response": template["response"],
                "category": template["category"],
                "created_at": datetime.now().isoformat(),
                "quality_score": round(random.uniform(0.8, 1.0), 2)
            })

        return Dataset.from_list(data)

    def save_dataset_locally(self, dataset: Dataset, filename: str = "sample_customer_support.json"):
        """
        Save dataset to local file for future use

        This allows you to cache the simulated data and avoid regeneration
        """
        output_path = Path("data/raw") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataset to list of dicts and save as JSON
        data = [dict(item) for item in dataset]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        console.print(f"[green]✓ Saved {len(data)} examples to {output_path}[/green]")

    def get_table_schema(self, table_name: str) -> Dict:
        """
        Get schema information for a Databricks table

        PRODUCTION CODE:
        ```python
        query = f"DESCRIBE TABLE {self.catalog}.{self.schema}.{table_name}"
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            schema_info = cursor.fetchall()
        return schema_info
        ```
        """
        if self.use_simulation:
            return {
                "columns": [
                    {"name": "instruction", "type": "string", "description": "User question/query"},
                    {"name": "response", "type": "string", "description": "Support agent response"},
                    {"name": "category", "type": "string", "description": "Support category"},
                    {"name": "created_at", "type": "timestamp", "description": "Record creation time"},
                    {"name": "quality_score", "type": "float", "description": "Response quality (0-1)"}
                ],
                "table": f"{self.catalog}.{self.schema}.customer_support_qa",
                "partitioned_by": ["created_at"],
                "format": "delta"
            }
        else:
            raise NotImplementedError("Production schema retrieval not implemented")

    def close(self):
        """Close Databricks connection"""
        if self.connection:
            self.connection.close()
            console.print("[yellow]Databricks connection closed[/yellow]")


class DataPreprocessor:
    """
    Preprocess data for LLM fine-tuning

    Handles:
    - Text cleaning
    - Instruction formatting (Alpaca, ChatML, etc.)
    - Tokenization
    - Train/validation split
    """

    def __init__(self, format: str = "alpaca"):
        """
        Initialize preprocessor

        Args:
            format: Instruction format ('alpaca', 'chatml', 'plain')
        """
        self.format = format

    def format_instruction(self, example: Dict) -> Dict:
        """
        Format examples into instruction-following format

        Alpaca format:
        ```
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        {response}
        ```
        """
        if self.format == "alpaca":
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['response']}"
            )
        elif self.format == "chatml":
            text = (
                f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['response']}<|im_end|>"
            )
        else:  # plain
            text = f"Question: {example['instruction']}\n\nAnswer: {example['response']}"

        return {"text": text, **example}

    def prepare_dataset(self, dataset: Dataset, train_split: float = 0.9) -> Dict[str, Dataset]:
        """
        Prepare dataset for training

        Args:
            dataset: Raw dataset
            train_split: Fraction of data for training

        Returns:
            Dict with 'train' and 'validation' datasets
        """
        console.print(f"[cyan]Formatting {len(dataset)} examples with {self.format} format...[/cyan]")

        # Format all examples
        formatted_dataset = dataset.map(
            self.format_instruction,
            desc="Formatting instructions"
        )

        # Split into train/validation
        split_dataset = formatted_dataset.train_test_split(
            train_size=train_split,
            seed=42
        )

        console.print(f"[green]✓ Train: {len(split_dataset['train'])} examples[/green]")
        console.print(f"[green]✓ Validation: {len(split_dataset['test'])} examples[/green]\n")

        return {
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        }


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Databricks connector

    Run this script to test the data loading pipeline:
    $ python data/data_loader.py
    """
    console.print("\n[bold cyan]═══ Enterprise Data Loading Pipeline ═══[/bold cyan]\n")

    # Initialize connector in simulation mode
    connector = DatabricksConnector(
        use_simulation=True,
        catalog="portfolio_demo",
        schema="ml_training"
    )

    # Show table schema
    console.print("[bold]Table Schema:[/bold]")
    schema = connector.get_table_schema("customer_support_qa")
    for col in schema["columns"]:
        console.print(f"  • {col['name']:20s} ({col['type']:10s}) - {col['description']}")
    console.print()

    # Load training data
    dataset = connector.load_training_data(limit=100)

    # Show sample
    console.print("[bold]Sample Data:[/bold]")
    for i in range(3):
        console.print(f"\n[yellow]Example {i+1}:[/yellow]")
        console.print(f"  Q: {dataset[i]['instruction']}")
        console.print(f"  A: {dataset[i]['response'][:100]}...")

    # Save for future use
    connector.save_dataset_locally(dataset, "sample_customer_support.json")

    # Preprocess data
    preprocessor = DataPreprocessor(format="alpaca")
    prepared_data = preprocessor.prepare_dataset(dataset, train_split=0.9)

    console.print("\n[bold]Formatted Example:[/bold]")
    console.print(prepared_data["train"][0]["text"][:300] + "...")

    console.print("\n[green]✓ Data pipeline test complete![/green]\n")
