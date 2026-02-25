"""
CLI Runner for Clinical LLM Enrichment POC.

Usage:
  # With Ollama running locally:
  python run.py --model qwen2.5:7b --input data/ner_ext.json

  # Benchmark multiple models:
  python run.py --model qwen2.5:7b --model llama3.1:8b --input data/ner_ext.json

  # Mock mode (no Ollama needed):
  python run.py --model mock --input data/ner_ext.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ner_parser import parse_ner_json
from src.model_factory import create_model, list_available_models
from src.enrichment_engine import enrich_entities
from src.output_formatter import generate_enriched_json, generate_comparison_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_enrichment(input_path: str, model_name: str, output_dir: str, ollama_url: str) -> dict:
    """Run enrichment for a single model and return results."""
    logger.info(f"{'='*60}")
    logger.info(f"Running enrichment with model: {model_name}")
    logger.info(f"{'='*60}")

    # 1. Parse NER output
    parsed = parse_ner_json(input_path)
    logger.info(
        f"Parsed {len(parsed.entities)} entities, {len(parsed.relations)} relations "
        f"from note ({len(parsed.clinical_note)} chars)"
    )

    problems = parsed.get_problems()
    logger.info(f"Problem entities (assertion=present): {len(problems)}")
    for p in problems:
        imo = p.codemap.imo_lexical_title if p.codemap else "no codemap"
        logger.info(f"  - \"{p.text}\" -> {imo}")

    # 2. Create model
    llm = create_model(model_name, base_url=ollama_url)

    if not llm.health_check():
        logger.warning(f"Model {model_name} not available. Use --model mock for testing.")
        if model_name != "mock":
            available = list_available_models(ollama_url)
            if available:
                logger.info(f"Available Ollama models: {available}")
            else:
                logger.info("No Ollama models found. Is Ollama running?")
            return {}

    # 3. Run enrichment
    result = enrich_entities(parsed, llm)
    logger.info(
        f"Enrichment complete: {result.enriched_entity_count} enriched concepts "
        f"from {result.original_entity_count} original entities"
    )

    # 4. Generate outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace(":", "_").replace("/", "_")

    # Enriched JSON
    json_path = out / f"enriched_{safe_model}.json"
    enriched_data = generate_enriched_json(parsed, result, json_path)
    logger.info(f"Enriched JSON -> {json_path}")

    # Comparison table
    table_path = out / f"comparison_{safe_model}.md"
    generate_comparison_table(parsed, result, table_path)
    logger.info(f"Comparison table -> {table_path}")

    # Raw LLM response (for debugging)
    raw_path = out / f"raw_response_{safe_model}.txt"
    with open(raw_path, "w") as f:
        f.write(result.raw_llm_response)

    return enriched_data


def run_benchmark(input_path: str, models: list[str], output_dir: str, ollama_url: str):
    """Run enrichment across multiple models and produce a benchmark summary."""
    all_results = {}

    for model_name in models:
        result = run_enrichment(input_path, model_name, output_dir, ollama_url)
        if result:
            all_results[model_name] = result

    if len(all_results) > 1:
        # Generate benchmark comparison
        summary_path = Path(output_dir) / "benchmark_summary.md"
        lines = [
            "# Model Benchmark Summary",
            "",
            "| Model | Enriched Concepts | Prompt Tokens | Completion Tokens | Latency (ms) |",
            "|-------|------------------|---------------|-------------------|-------------|",
        ]

        for model_name, data in all_results.items():
            meta = data.get("metadata", {})
            lines.append(
                f"| {model_name} | {meta.get('enriched_entity_count', 0)} | "
                f"{meta.get('prompt_tokens', 0)} | {meta.get('completion_tokens', 0)} | "
                f"{meta.get('latency_ms', 0)} |"
            )

        lines.extend(["", "---", ""])

        # Compare enriched concepts across models
        lines.append("## Enriched Concepts by Model")
        lines.append("")
        for model_name, data in all_results.items():
            lines.append(f"### {model_name}")
            for ee in data.get("enriched_entities", []):
                lines.append(f"- [{ee['inference_strength']}] {ee['enriched_concept']}")
            lines.append("")

        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Benchmark summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Clinical LLM Enrichment POC")
    parser.add_argument("--input", "-i", required=True, help="Path to NER JSON file")
    parser.add_argument("--model", "-m", action="append", required=True,
                        help="Model name(s). Use multiple --model flags for benchmark.")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")

    args = parser.parse_args()

    if len(args.model) == 1:
        run_enrichment(args.input, args.model[0], args.output, args.ollama_url)
    else:
        run_benchmark(args.input, args.model, args.output, args.ollama_url)


if __name__ == "__main__":
    main()
