"""
FastAPI Application for Clinical LLM Enrichment.

This API provides endpoints to enrich Named Entity Recognition (NER) output
from clinical notes using local LLM models via Ollama.

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ner_parser import parse_ner_json, ParsedNER
from src.model_factory import create_model, list_available_models
from src.enrichment_engine import enrich_entities
from src.output_formatter import generate_enriched_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Clinical LLM Enrichment API",
    description="API for enriching clinical NER entities using local LLM models",
    version="1.0.0",
)


# ── Pydantic Models ──────────────────────────────────────────────────────


class CodeMapResponse(BaseModel):
    """CodeMap information for an entity."""
    imo_lexical_title: str = ""
    imo_lexical_code: str = ""
    imo_confidence: str = ""
    icd10_code: str = ""
    icd10_title: str = ""
    snomed_code: str = ""
    snomed_title: str = ""


class NEREntityResponse(BaseModel):
    """NER Entity information."""
    id: str
    text: str
    begin: int
    end: int
    semantic: str
    assertion: str = "present"
    section: str = ""
    origin: str = ""
    sentence_prob: Optional[float] = None
    concept_prob: Optional[float] = None
    linked_entity_ids: List[str] = []
    codemap: Optional[CodeMapResponse] = None


class SourceEntityDetail(BaseModel):
    """Source entity details in enriched output."""
    id: str
    text: str
    semantic: str
    assertion: str
    original_imo: Optional[Dict[str, str]] = None
    original_icd10: Optional[Dict[str, str]] = None


class EnrichedEntityResponse(BaseModel):
    """Enriched clinical concept."""
    enriched_concept: str
    inference_strength: str
    clinical_reasoning: str
    evidence_spans: List[str] = []
    source_entities: List[SourceEntityDetail]
    source_entity_texts: List[str]


class EnrichmentMetadata(BaseModel):
    """Enrichment metadata."""
    model: str
    original_entity_count: int
    enriched_entity_count: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class EnrichmentResponse(BaseModel):
    """Complete enrichment response."""
    metadata: EnrichmentMetadata
    enriched_entities: List[EnrichedEntityResponse]


class EnrichmentRequest(BaseModel):
    """Request to enrich NER entities."""
    ner_data: Dict[str, Any] = Field(..., description="NER JSON data with content, indexes, entities, and relations")
    model_name: str = Field(default="qwen2.5:7b", description="Model name (e.g., 'qwen2.5:7b', 'llama3.1:8b', 'mock')")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")


class BenchmarkRequest(BaseModel):
    """Request to benchmark multiple models."""
    ner_data: Dict[str, Any] = Field(..., description="NER JSON data")
    model_names: List[str] = Field(..., description="List of model names to benchmark")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")


class BenchmarkResponse(BaseModel):
    """Benchmark response with results from multiple models."""
    results: Dict[str, EnrichmentResponse]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_available: bool
    ollama_url: str


class ModelsResponse(BaseModel):
    """Available models response."""
    supported_models: List[str]
    available_ollama_models: List[str]


# ── Helper Functions ─────────────────────────────────────────────────────


def parse_ner_from_dict(ner_data: Dict[str, Any]) -> ParsedNER:
    """Parse NER data from dictionary (instead of file)."""
    import json
    import tempfile

    # Write to temp file and parse
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ner_data, f)
        temp_path = f.name

    try:
        parsed = parse_ner_json(temp_path)
        return parsed
    finally:
        Path(temp_path).unlink(missing_ok=True)


def enrichment_result_to_response(parsed: ParsedNER, result) -> EnrichmentResponse:
    """Convert enrichment result to API response format."""
    enriched_entities = []

    for ee in result.enriched_entities:
        # Gather source entity details
        source_details = []
        for eid in ee.source_entity_ids:
            ent = next((e for e in parsed.entities if e.id == eid), None)
            if ent:
                detail = SourceEntityDetail(
                    id=ent.id,
                    text=ent.text,
                    semantic=ent.semantic,
                    assertion=ent.assertion,
                )
                if ent.codemap:
                    detail.original_imo = {
                        "lexical_title": ent.codemap.imo_lexical_title,
                        "lexical_code": ent.codemap.imo_lexical_code,
                    }
                    if ent.codemap.icd10_code:
                        detail.original_icd10 = {
                            "code": ent.codemap.icd10_code,
                            "title": ent.codemap.icd10_title,
                        }
                source_details.append(detail)

        enriched_entities.append(EnrichedEntityResponse(
            enriched_concept=ee.enriched_concept,
            inference_strength=ee.inference_strength,
            clinical_reasoning=ee.clinical_reasoning,
            evidence_spans=ee.evidence_spans,
            source_entities=source_details,
            source_entity_texts=ee.source_entity_texts,
        ))

    return EnrichmentResponse(
        metadata=EnrichmentMetadata(
            model=result.model_name,
            original_entity_count=result.original_entity_count,
            enriched_entity_count=result.enriched_entity_count,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=round(result.latency_ms, 1),
        ),
        enriched_entities=enriched_entities,
    )


# ── API Endpoints ────────────────────────────────────────────────────────


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Clinical LLM Enrichment API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "enrich": "/enrich",
            "benchmark": "/enrich/benchmark",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(ollama_url: str = "http://localhost:11434"):
    """
    Check API and Ollama service health.

    Args:
        ollama_url: Ollama base URL

    Returns:
        Health status including Ollama availability
    """
    import requests

    ollama_available = False
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        ollama_available = resp.ok
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")

    return HealthResponse(
        status="healthy",
        ollama_available=ollama_available,
        ollama_url=ollama_url,
    )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def get_models(ollama_url: str = "http://localhost:11434"):
    """
    List supported and available models.

    Args:
        ollama_url: Ollama base URL

    Returns:
        Supported model names and available Ollama models
    """
    from src.model_factory import SUPPORTED_MODELS

    supported = list(SUPPORTED_MODELS.keys())
    available = list_available_models(ollama_url)

    return ModelsResponse(
        supported_models=supported,
        available_ollama_models=available,
    )


@app.post("/enrich", response_model=EnrichmentResponse, tags=["Enrichment"])
async def enrich(request: EnrichmentRequest):
    """
    Enrich NER entities using a single LLM model.

    Args:
        request: Enrichment request with NER data and model configuration

    Returns:
        Enriched entities with clinical reasoning and metadata

    Raises:
        HTTPException: If model is unavailable or enrichment fails
    """
    try:
        # Parse NER data
        logger.info(f"Parsing NER data for enrichment with model: {request.model_name}")
        parsed = parse_ner_from_dict(request.ner_data)
        logger.info(
            f"Parsed {len(parsed.entities)} entities, {len(parsed.relations)} relations "
            f"from note ({len(parsed.clinical_note)} chars)"
        )

        # Create model
        llm = create_model(request.model_name, base_url=request.ollama_url)

        # Health check
        if not llm.health_check():
            if request.model_name != "mock":
                available = list_available_models(request.ollama_url)
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": f"Model {request.model_name} not available",
                        "available_models": available,
                        "suggestion": "Use model_name='mock' for testing without Ollama"
                    }
                )

        # Run enrichment
        result = enrich_entities(parsed, llm, temperature=request.temperature)
        logger.info(
            f"Enrichment complete: {result.enriched_entity_count} enriched concepts "
            f"from {result.original_entity_count} original entities"
        )

        # Convert to response
        return enrichment_result_to_response(parsed, result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrichment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {str(e)}")


@app.post("/enrich/benchmark", response_model=BenchmarkResponse, tags=["Enrichment"])
async def benchmark(request: BenchmarkRequest):
    """
    Benchmark enrichment across multiple models.

    Args:
        request: Benchmark request with NER data and multiple model names

    Returns:
        Results from all models with comparison summary

    Raises:
        HTTPException: If parsing or enrichment fails
    """
    try:
        # Parse NER data once
        logger.info(f"Running benchmark with {len(request.model_names)} models")
        parsed = parse_ner_from_dict(request.ner_data)

        results = {}
        summary_data = {}

        for model_name in request.model_names:
            try:
                logger.info(f"Benchmarking with model: {model_name}")

                # Create model
                llm = create_model(model_name, base_url=request.ollama_url)

                # Health check
                if not llm.health_check():
                    if model_name != "mock":
                        logger.warning(f"Model {model_name} not available, skipping")
                        continue

                # Run enrichment
                result = enrich_entities(parsed, llm, temperature=request.temperature)

                # Convert to response
                response = enrichment_result_to_response(parsed, result)
                results[model_name] = response

                # Add to summary
                summary_data[model_name] = {
                    "enriched_entity_count": result.enriched_entity_count,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_ms": round(result.latency_ms, 1),
                }

            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                summary_data[model_name] = {"error": str(e)}

        if not results:
            raise HTTPException(
                status_code=503,
                detail="No models were available for benchmarking"
            )

        return BenchmarkResponse(
            results=results,
            summary=summary_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


# ── Startup/Shutdown Events ──────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("Clinical LLM Enrichment API started")
    logger.info("=" * 60)
    logger.info("Docs available at: http://localhost:8000/docs")
    logger.info("Health check: http://localhost:8000/health")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("Clinical LLM Enrichment API shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
