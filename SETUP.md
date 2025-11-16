# Setup Guide - Enterprise LLM Fine-Tuning Pipeline

This guide provides detailed setup instructions, troubleshooting tips, and information for running the project locally.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Components](#running-components)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)

---

## System Requirements

### Minimum Requirements

- **OS:** Linux, macOS, or Windows (WSL2 recommended)
- **Python:** 3.10 or higher
- **RAM:** 4 GB minimum (8 GB recommended)
- **Disk Space:** 5 GB free
- **CPU:** Modern multi-core processor

### Recommended for Training

- **RAM:** 16 GB+ for larger datasets
- **GPU:** NVIDIA GPU with 8+ GB VRAM (optional but speeds up training 10-100x)
- **Disk:** SSD for faster data loading

**Note:** This project is designed to run on CPU for portfolio demonstration. With GPU, you can train larger models (GPT-2 Medium/Large, LLaMA, etc.).

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/enterprise-sft-pipeline.git
cd enterprise-sft-pipeline
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Verify activation:**
```bash
which python  # Linux/macOS
where python  # Windows
# Should point to venv/bin/python or venv\Scripts\python.exe
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

**Full installation:**
```bash
pip install -r requirements.txt
```

**Minimal installation (core only):**
```bash
pip install torch transformers datasets peft fastapi uvicorn python-dotenv rich
```

**For GPU support (NVIDIA only):**
```bash
# Linux
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# macOS (Metal/MPS)
pip install torch torchvision torchaudio
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from peft import LoraConfig; print('PEFT: OK')"
```

---

## Configuration

### Environment Variables

1. **Copy the example environment file:**
```bash
cp .env.example .env
```

2. **Edit .env for your setup:**
```bash
# Simulation mode (default for portfolio demo)
USE_SIMULATED_DATA=true
SAMPLE_DATA_SIZE=1000

# Model configuration
BASE_MODEL_NAME=distilgpt2
DEVICE=cpu  # Change to 'cuda' if you have GPU

# Training
NUM_EPOCHS=3
TRAIN_BATCH_SIZE=4
LEARNING_RATE=2e-4

# API
API_PORT=8000

# Experiment tracking
EXPERIMENT_TRACKER=mlflow  # or 'wandb'
```

### Databricks Configuration (For Production)

To connect to a real Databricks workspace:

1. **Uncomment Databricks settings in .env:**
```bash
DATABRICKS_WORKSPACE_URL=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_personal_access_token
DATABRICKS_CATALOG=your_catalog
DATABRICKS_SCHEMA=your_schema
DATABRICKS_TABLE=training_data
USE_SIMULATED_DATA=false
```

2. **Install Databricks connector:**
```bash
pip install databricks-sql-connector delta-spark
```

3. **Update data/data_loader.py** to use real connection (see production notes in code)

### MLflow vs Weights & Biases

**Using MLflow (default):**
- No additional setup needed
- Tracking data stored in `monitoring/mlflow.db`
- View UI: `mlflow ui --backend-store-uri sqlite:///monitoring/mlflow.db`

**Using Weights & Biases:**
1. Install: `pip install wandb`
2. Login: `wandb login`
3. Set in .env: `EXPERIMENT_TRACKER=wandb`
4. Or use flag: `python training/train_sft.py --wandb`

---

## Running Components

### 1. Data Exploration

**Interactive notebook:**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Or run data loader directly:**
```bash
python data/data_loader.py
```

**Expected output:**
- Sample data loaded
- Statistics printed
- Data saved to `data/raw/sample_customer_support.json`

### 2. Model Training

**Basic training:**
```bash
python training/train_sft.py --config training/config.yaml
```

**With options:**
```bash
# Dry run (setup only, no actual training)
python training/train_sft.py --dry-run

# With W&B tracking
python training/train_sft.py --wandb

# Custom config
python training/train_sft.py --config my_custom_config.yaml
```

**Training progress:**
- Progress bars show training/evaluation steps
- Logs written to `training.log`
- Checkpoints saved to `models/checkpoints/`
- Model registered in `monitoring/metrics.db`

**Estimated time (1000 samples, 3 epochs):**
- CPU: ~5-10 minutes
- GPU: ~1-2 minutes

**What to expect:**
```
Training loss should decrease: 3.5 → 2.8 → 1.2 → 0.6
Eval loss should decrease: 3.2 → 2.5 → 1.0 → 0.5
```

### 3. Model Evaluation

**Evaluate latest model:**
```bash
python evaluation/run_evaluation.py --model-version latest --compare-base
```

**Evaluate specific version:**
```bash
python evaluation/run_evaluation.py --model-version customer-support-v1
```

**Evaluate from checkpoint path:**
```bash
python evaluation/run_evaluation.py --model models/checkpoints/my_model_20241116
```

**Outputs:**
- Console: Metrics table with comparison
- File: `evaluation/results/evaluation_YYYYMMDD_HHMMSS.json`
- File: `evaluation/results/evaluation_YYYYMMDD_HHMMSS.html`

### 4. API Server

**Start server:**
```bash
python api/main.py --port 8000
```

**Or with uvicorn (production):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Development mode (auto-reload):**
```bash
uvicorn api.main:app --reload
```

**Access points:**
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/api/v1/health

**Test the API:**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Make prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

### 5. Model Registry

**Interactive Python:**
```python
from models.registry import ModelRegistry, ModelStatus

registry = ModelRegistry()

# List all models
models = registry.list_models(limit=10)
registry.display_models(models)

# Get latest production model
prod = registry.get_latest_model(status=ModelStatus.PRODUCTION)
print(prod)

# Promote model
registry.update_status(
    version="customer-support-v2",
    new_status=ModelStatus.PRODUCTION
)
```

**Or run the demo:**
```bash
python models/registry.py
```

### 6. Experiment Tracking

**View MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///monitoring/mlflow.db --port 5000
```

Then visit: http://localhost:5000

**Query experiments:**
```bash
python monitoring/track_experiments.py
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. CUDA Not Available

**Problem:**
```
Warning: Running on CPU (training will be slow)
```

**Solution:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False but you have NVIDIA GPU, reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Out of Memory During Training

**Problem:**
```
RuntimeError: CUDA out of memory
# or
MemoryError: Unable to allocate array
```

**Solutions:**
```yaml
# In training/config.yaml, reduce:
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size

# Or enable gradient checkpointing:
training:
  gradient_checkpointing: true

# Or use smaller model:
model:
  base_model: "distilgpt2"  # Instead of gpt2-medium
```

#### 4. API Server Won't Start

**Problem:**
```
Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill the process or use different port
python api/main.py --port 8001
```

#### 5. Model Loading Fails

**Problem:**
```
OSError: Can't load tokenizer for 'models/checkpoints/...'
```

**Solution:**
```bash
# Ensure model was saved correctly
ls -la models/checkpoints/your-model/

# Should contain:
# - adapter_config.json
# - adapter_model.bin
# - tokenizer_config.json
# - special_tokens_map.json

# If files missing, retrain:
python training/train_sft.py
```

#### 6. Database Locked Error

**Problem:**
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Close all connections to database
# Stop API server and training scripts

# Or delete and recreate:
rm monitoring/metrics.db
python models/registry.py  # Reinitialize
```

### Performance Issues

#### Training is Slow

**Options:**
1. **Use GPU:** Set `DEVICE=cuda` in .env
2. **Reduce data:** Set `SAMPLE_DATA_SIZE=500` in .env
3. **Use smaller model:** Use `distilgpt2` instead of `gpt2`
4. **Increase batch size:** If you have RAM, increase `TRAIN_BATCH_SIZE`

#### API Latency is High

**Options:**
1. **Pre-load model:** Uncomment model pre-loading in `api/main.py`
2. **Use GPU:** Models run 10-100x faster on GPU
3. **Reduce max_new_tokens:** Default is 128, try 64
4. **Use greedy decoding:** Set `temperature=0` (faster but less creative)

---

## Production Considerations

### 1. Security

**API Security:**
```python
# Add authentication to api/main.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/v1/predict")
async def predict(
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid token")
    ...
```

**Environment secrets:**
```bash
# Never commit .env file
# Use proper secrets management:
# - AWS Secrets Manager
# - Azure Key Vault
# - HashiCorp Vault
```

### 2. Scaling

**Horizontal scaling:**
```bash
# Run multiple API workers
uvicorn api.main:app --workers 4 --port 8000

# Or use Docker + Kubernetes
docker build -t llm-api .
kubectl apply -f k8s/deployment.yaml
```

**Model caching:**
```python
# In api/inference.py
# Models are automatically cached
# Configure cache size based on available RAM
```

### 3. Monitoring

**Add observability:**
```bash
pip install prometheus-client
```

```python
# Add metrics endpoint
from prometheus_client import Counter, Histogram, make_asgi_app

request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 4. Database

**For production, use PostgreSQL:**
```python
# Update MODEL_REGISTRY_DB in .env
MODEL_REGISTRY_DB=postgresql://user:pass@localhost/registry

# Install psycopg2
pip install psycopg2-binary

# Update models/registry.py to use SQLAlchemy properly
```

### 5. Real Databricks Integration

**Update data/data_loader.py:**
```python
from databricks import sql

def _connect_to_databricks(self):
    self.connection = sql.connect(
        server_hostname=self.workspace_url.replace("https://", ""),
        http_path="/sql/1.0/warehouses/your-warehouse-id",
        access_token=self.token
    )
```

---

## Next Steps

After successful setup:

1. ✅ **Explore Data:** Run notebooks to understand the data
2. ✅ **Train Model:** Run a quick training session (100 samples, 1 epoch)
3. ✅ **Evaluate:** Compare base vs fine-tuned
4. ✅ **Test API:** Start server and make predictions
5. ✅ **Customize:** Modify for your specific use case

---

## Getting Help

**Issues with this setup?**

1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Check Python and package versions
4. Search existing GitHub issues
5. Open a new issue with:
   - Error message (full traceback)
   - Python version: `python --version`
   - OS: `uname -a` (Linux/Mac) or `ver` (Windows)
   - Steps to reproduce

**Community Resources:**

- HuggingFace Forums: https://discuss.huggingface.co/
- FastAPI Discord: https://discord.gg/VQjSZaeJmf
- Stack Overflow: Tag questions with `transformers`, `fastapi`, `peft`

---

## Development Tips

### Code Style

```bash
# Format code
pip install black isort
black .
isort .

# Lint
pip install flake8
flake8 --max-line-length=120
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests (when implemented)
pytest tests/ -v --cov=.

# Test specific component
pytest tests/test_data_loader.py -v
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add: description of changes"

# Push to GitHub
git push -u origin feature/my-feature
```

---

<p align="center">
  <strong>You're all set! 🚀</strong>
  <br>
  <i>Ready to fine-tune some models?</i>
</p>
