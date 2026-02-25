"""
Example API Client for Clinical LLM Enrichment.

This script demonstrates how to interact with the FastAPI enrichment service.

Usage:
    python example_client.py
"""

import json
import requests
from pathlib import Path


# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_health():
    """Check if the API is healthy and Ollama is available."""
    print("Checking API health...")
    response = requests.get(f"{API_BASE_URL}/health")
    health = response.json()

    print(f"API Status: {health['status']}")
    print(f"Ollama Available: {health['ollama_available']}")
    print(f"Ollama URL: {health['ollama_url']}")
    return health['ollama_available']


def list_models():
    """List available models."""
    print("\nListing available models...")
    response = requests.get(f"{API_BASE_URL}/models")
    models = response.json()

    print("Supported models:", ", ".join(models['supported_models']))
    print("Available Ollama models:", ", ".join(models['available_ollama_models']))
    return models


def enrich_single_model(ner_data_path: str, model_name: str = "qwen2.5:7b"):
    """Enrich NER entities using a single model."""
    print(f"\nEnriching with model: {model_name}")

    # Load NER data
    with open(ner_data_path) as f:
        ner_data = json.load(f)

    # Make API request
    response = requests.post(
        f"{API_BASE_URL}/enrich",
        json={
            "ner_data": ner_data,
            "model_name": model_name,
            "temperature": 0.1
        }
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

    result = response.json()

    # Display results
    print(f"\nEnrichment Results:")
    print(f"  Model: {result['metadata']['model']}")
    print(f"  Original entities: {result['metadata']['original_entity_count']}")
    print(f"  Enriched concepts: {result['metadata']['enriched_entity_count']}")
    print(f"  Latency: {result['metadata']['latency_ms']:.1f}ms")
    print(f"  Tokens: {result['metadata']['prompt_tokens']} prompt + {result['metadata']['completion_tokens']} completion")

    print(f"\nEnriched Concepts:")
    for i, entity in enumerate(result['enriched_entities'], 1):
        print(f"\n  {i}. {entity['enriched_concept']}")
        print(f"     Inference: {entity['inference_strength']}")
        print(f"     Reasoning: {entity['clinical_reasoning'][:100]}...")
        print(f"     Source entities: {', '.join(entity['source_entity_texts'])}")

    return result


def benchmark_models(ner_data_path: str, model_names: list[str]):
    """Benchmark multiple models."""
    print(f"\nBenchmarking {len(model_names)} models: {', '.join(model_names)}")

    # Load NER data
    with open(ner_data_path) as f:
        ner_data = json.load(f)

    # Make API request
    response = requests.post(
        f"{API_BASE_URL}/enrich/benchmark",
        json={
            "ner_data": ner_data,
            "model_names": model_names,
            "temperature": 0.1
        }
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

    result = response.json()

    # Display benchmark summary
    print("\nBenchmark Summary:")
    print("=" * 80)
    print(f"{'Model':<20} {'Enriched':<12} {'Latency (ms)':<15} {'Tokens':<20}")
    print("=" * 80)

    for model_name, summary in result['summary'].items():
        if 'error' in summary:
            print(f"{model_name:<20} ERROR: {summary['error']}")
        else:
            tokens = f"{summary['prompt_tokens']}+{summary['completion_tokens']}"
            print(
                f"{model_name:<20} "
                f"{summary['enriched_entity_count']:<12} "
                f"{summary['latency_ms']:<15.1f} "
                f"{tokens:<20}"
            )

    print("=" * 80)

    return result


def main():
    """Main example workflow."""
    print("=" * 60)
    print("Clinical LLM Enrichment API - Example Client")
    print("=" * 60)

    # 1. Health check
    ollama_available = check_health()

    # 2. List models
    models_info = list_models()

    # 3. Check if we have NER data
    ner_data_path = "data/ner_ext.json"
    if not Path(ner_data_path).exists():
        print(f"\nError: NER data file not found at {ner_data_path}")
        print("Please provide a valid NER JSON file to continue.")
        return

    # 4. Single model enrichment
    if ollama_available and models_info['available_ollama_models']:
        # Use the first available model
        model = models_info['available_ollama_models'][0]
        enrich_single_model(ner_data_path, model)
    else:
        # Use mock model
        print("\nOllama not available, using mock model...")
        enrich_single_model(ner_data_path, "mock")

    # 5. Optional: Benchmark (uncomment to run)
    # if ollama_available and len(models_info['available_ollama_models']) > 1:
    #     models_to_benchmark = models_info['available_ollama_models'][:2]
    #     benchmark_models(ner_data_path, models_to_benchmark)


if __name__ == "__main__":
    main()
