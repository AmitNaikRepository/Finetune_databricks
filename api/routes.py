"""
API Routes for Model Inference

Defines all API endpoints for the inference service.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status
from rich.console import Console

from api.inference import get_inference_engine
from models.registry import ModelRegistry, ModelStatus

console = Console()

# Create API router
router = APIRouter()


# Request/Response Models (Pydantic schemas)

class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    query: str = Field(..., description="Input query/question", min_length=1, max_length=1000)
    model_version: Optional[str] = Field("latest", description="Model version to use")
    max_new_tokens: int = Field(128, ge=10, le=512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I reset my password?",
                "model_version": "latest",
                "max_new_tokens": 128,
                "temperature": 0.7
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction"""
    queries: List[str] = Field(..., description="List of queries", min_items=1, max_items=100)
    model_version: Optional[str] = Field("latest", description="Model version to use")
    max_new_tokens: int = Field(128, ge=10, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    class Config:
        json_schema_extra = {
            "example": {
                "queries": [
                    "How do I reset my password?",
                    "Where is my order?",
                    "How can I get a refund?"
                ],
                "model_version": "latest"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    query: str
    response: str
    model_version: str
    latency_ms: float
    tokens_generated: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    total_inferences: int
    average_latency_ms: float


class ModelInfo(BaseModel):
    """Model information from registry"""
    version: str
    model_name: str
    status: str
    training_date: str
    eval_loss: Optional[float]
    eval_accuracy: Optional[float]


# API Endpoints

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns service status and statistics
    """
    engine = get_inference_engine()
    stats = engine.get_stats()

    return HealthResponse(
        status="healthy",
        models_loaded=stats["loaded_models"],
        total_inferences=stats["total_inferences"],
        average_latency_ms=stats["average_latency_ms"]
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    """
    Generate response for a single query

    This endpoint:
    1. Loads the specified model version (or latest)
    2. Generates a response using the fine-tuned model
    3. Returns the response with metadata

    Example:
        ```
        POST /predict
        {
            "query": "How do I reset my password?",
            "model_version": "latest",
            "temperature": 0.7
        }
        ```
    """
    try:
        # Get inference engine
        engine = get_inference_engine()

        # Resolve model version
        if request.model_version == "latest":
            registry = ModelRegistry()
            latest_model = registry.get_latest_model(status=ModelStatus.PRODUCTION)

            if not latest_model:
                # Fallback to any latest model
                latest_model = registry.get_latest_model()

            if not latest_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No models found in registry. Please train a model first."
                )

            model_path = latest_model['model_path']
            model_version = latest_model['version']

            # Load model if not already loaded
            if model_version not in engine.loaded_models:
                engine.load_model(model_path, model_version)
        else:
            # Use specified version
            registry = ModelRegistry()
            model = registry.get_model(request.model_version)

            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model version '{request.model_version}' not found"
                )

            model_path = model['model_path']
            model_version = model['version']

            if model_version not in engine.loaded_models:
                engine.load_model(model_path, model_version)

        # Generate prediction
        result = engine.predict(
            query=request.query,
            model_version=model_version,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )

        return PredictionResponse(
            query=request.query,
            response=result["response"],
            model_version=result["model_version"],
            latency_ms=result["latency_ms"],
            tokens_generated=result["tokens_generated"]
        )

    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[red]Prediction error: {e}[/red]")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Generate responses for multiple queries in batch

    Useful for processing multiple requests efficiently.

    Example:
        ```
        POST /predict/batch
        {
            "queries": [
                "How do I reset my password?",
                "Where is my order?",
                "I need a refund"
            ],
            "model_version": "latest"
        }
        ```
    """
    try:
        engine = get_inference_engine()

        # Resolve model version (same logic as single predict)
        if request.model_version == "latest":
            registry = ModelRegistry()
            latest_model = registry.get_latest_model(status=ModelStatus.PRODUCTION)
            if not latest_model:
                latest_model = registry.get_latest_model()
            if not latest_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No models found"
                )
            model_path = latest_model['model_path']
            model_version = latest_model['version']

            if model_version not in engine.loaded_models:
                engine.load_model(model_path, model_version)
        else:
            registry = ModelRegistry()
            model = registry.get_model(request.model_version)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model version '{request.model_version}' not found"
                )
            model_version = model['version']
            if model_version not in engine.loaded_models:
                engine.load_model(model['model_path'], model_version)

        # Batch prediction
        results = engine.predict_batch(
            queries=request.queries,
            model_version=model_version,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )

        # Format responses
        responses = [
            PredictionResponse(
                query=query,
                response=result["response"],
                model_version=result["model_version"],
                latency_ms=result["latency_ms"],
                tokens_generated=result["tokens_generated"]
            )
            for query, result in zip(request.queries, results)
        ]

        return {
            "predictions": responses,
            "total_queries": len(request.queries),
            "total_latency_ms": sum(r.latency_ms for r in responses)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/models", tags=["Models"])
async def list_models(status_filter: Optional[str] = None, limit: int = 10):
    """
    List all registered models

    Query parameters:
    - status_filter: Filter by status (staging, production, archived)
    - limit: Maximum number of models to return

    Example:
        ```
        GET /models?status_filter=production&limit=5
        ```
    """
    try:
        registry = ModelRegistry()

        # Parse status filter
        status_enum = None
        if status_filter:
            try:
                status_enum = ModelStatus(status_filter.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status. Must be one of: {[s.value for s in ModelStatus]}"
                )

        models = registry.list_models(status=status_enum, limit=limit)

        model_infos = [
            ModelInfo(
                version=m['version'],
                model_name=m['model_name'],
                status=m['status'],
                training_date=m['training_date'],
                eval_loss=m.get('eval_loss'),
                eval_accuracy=m.get('eval_accuracy')
            )
            for m in models
        ]

        return {
            "models": model_infos,
            "total": len(model_infos)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/models/{version}", tags=["Models"])
async def get_model(version: str):
    """
    Get details for a specific model version

    Example:
        ```
        GET /models/customer-support-v1
        ```
    """
    try:
        registry = ModelRegistry()
        model = registry.get_model(version)

        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version '{version}' not found"
            )

        return model

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )


@router.get("/metrics/comparison", tags=["Metrics"])
async def get_metrics_comparison():
    """
    Get performance comparison between base and fine-tuned models

    Returns cost savings, quality improvements, and latency metrics
    """
    # This would typically load from evaluation results
    # For demo purposes, return example data

    return {
        "base_model": {
            "name": "distilgpt2",
            "perplexity": 35.2,
            "avg_latency_ms": 245,
            "cost_per_1k_tokens": 0.0001
        },
        "fine_tuned_model": {
            "name": "customer-support-v1",
            "perplexity": 18.7,
            "avg_latency_ms": 250,
            "cost_per_1k_tokens": 0.0001
        },
        "improvements": {
            "perplexity_reduction": "46.9%",
            "cost_savings_vs_gpt4": "99.7%",
            "quality_improvement": "22.7%"
        },
        "estimated_monthly_cost": {
            "fine_tuned_10k_requests": "$1.00",
            "gpt4_10k_requests": "$300.00",
            "monthly_savings": "$299.00"
        }
    }


@router.post("/models/{version}/promote", tags=["Models"])
async def promote_model(version: str, target_status: str):
    """
    Promote a model to a new status

    Example:
        ```
        POST /models/customer-support-v1/promote?target_status=production
        ```
    """
    try:
        # Validate status
        try:
            status_enum = ModelStatus(target_status.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.value for s in ModelStatus]}"
            )

        registry = ModelRegistry()
        registry.update_status(version, status_enum, deployed_by="api_user")

        return {
            "message": f"Model {version} promoted to {target_status}",
            "version": version,
            "new_status": target_status
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote model: {str(e)}"
        )
