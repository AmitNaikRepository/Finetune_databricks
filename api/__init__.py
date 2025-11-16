"""
API module for model inference service
"""

from .inference import ModelInferenceEngine, get_inference_engine

__all__ = ["ModelInferenceEngine", "get_inference_engine"]
