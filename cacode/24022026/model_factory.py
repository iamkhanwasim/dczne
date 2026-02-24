"""
Model Factory for Ollama LLM inference.
Supports pluggable models behind a common interface.
"""

import json
import time
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelResponse:
    """Standard response from any model."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: dict = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1) -> ModelResponse:
        pass

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.ok:
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model_name in m for m in models)
            return False
        except Exception:
            return False


class OllamaLLM(BaseLLM):
    """Ollama-based LLM inference."""

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1) -> ModelResponse:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,
            },
        }

        start = time.time()
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        latency = (time.time() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        return ModelResponse(
            content=data.get("response", ""),
            model=self.model_name,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency,
            raw_response=data,
        )


class MockLLM(BaseLLM):
    """Mock LLM for testing without Ollama running."""

    def __init__(self, model_name: str = "mock", base_url: str = ""):
        super().__init__(model_name, base_url)

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1) -> ModelResponse:
        # Return a realistic mock enrichment response
        mock_output = json.dumps({
            "enriched_entities": [
                {
                    "enriched_concept": "Osteomyelitis of left great toe associated with poorly controlled type 2 diabetes mellitus",
                    "source_entities": ["osteomyelitis", "left great Toe", "poorly controlled diabetes"],
                    "clinical_reasoning": "Osteomyelitis in a diabetic patient with poor glycemic control indicates diabetic osteomyelitis. The left great toe location combined with subsequent amputation suggests diabetic foot complication.",
                    "inference_strength": "strong_suggestion",
                },
                {
                    "enriched_concept": "Amputation of left great toe due to diabetic osteomyelitis",
                    "source_entities": ["amputation of the left great toe", "osteomyelitis", "poorly controlled diabetes"],
                    "clinical_reasoning": "The amputation was performed as treatment for the osteomyelitis, which is a known complication of diabetic foot disease.",
                    "inference_strength": "strong_suggestion",
                },
            ]
        }, indent=2)

        return ModelResponse(
            content=mock_output,
            model="mock",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(mock_output.split()),
            latency_ms=50.0,
        )

    def health_check(self) -> bool:
        return True


# ── Factory ──────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "qwen2.5:7b": OllamaLLM,
    "llama3.1:8b": OllamaLLM,
    "mock": MockLLM,
}


def create_model(model_name: str, base_url: str = "http://localhost:11434") -> BaseLLM:
    """Factory: create an LLM instance by model name."""
    if model_name == "mock":
        return MockLLM()

    model_class = SUPPORTED_MODELS.get(model_name)
    if not model_class:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}"
        )
    return model_class(model_name=model_name, base_url=base_url)


def list_available_models(base_url: str = "http://localhost:11434") -> list[str]:
    """List models available in local Ollama."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.ok:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []
