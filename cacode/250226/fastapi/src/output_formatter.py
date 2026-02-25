"""
Output Formatter.
Produces two outputs:
  1. Enriched JSON — the enrichment results in structured format
  2. Comparison table — before (NER) vs after (LLM enriched) side by side
"""

import json
from dataclasses import asdict
from pathlib import Path

from .ner_parser import ParsedNER
from .enrichment_engine import EnrichmentResult


def generate_enriched_json(parsed: ParsedNER, result: EnrichmentResult, output_path: str | Path) -> dict:
    """Generate the enriched entities JSON output."""
    output = {
        "metadata": {
            "model": result.model_name,
            "original_entity_count": result.original_entity_count,
            "enriched_entity_count": result.enriched_entity_count,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": round(result.latency_ms, 1),
        },
        "enriched_entities": [],
    }

    for ee in result.enriched_entities:
        # Gather source entity details
        source_details = []
        for eid in ee.source_entity_ids:
            ent = next((e for e in parsed.entities if e.id == eid), None)
            if ent:
                detail = {
                    "id": ent.id,
                    "text": ent.text,
                    "semantic": ent.semantic,
                    "assertion": ent.assertion,
                }
                if ent.codemap:
                    detail["original_imo"] = {
                        "lexical_title": ent.codemap.imo_lexical_title,
                        "lexical_code": ent.codemap.imo_lexical_code,
                    }
                    if ent.codemap.icd10_code:
                        detail["original_icd10"] = {
                            "code": ent.codemap.icd10_code,
                            "title": ent.codemap.icd10_title,
                        }
                source_details.append(detail)

        output["enriched_entities"].append({
            "enriched_concept": ee.enriched_concept,
            "inference_strength": ee.inference_strength,
            "clinical_reasoning": ee.clinical_reasoning,
            "evidence_spans": ee.evidence_spans,
            "source_entities": source_details,
            "source_entity_texts": ee.source_entity_texts,
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def generate_comparison_table(parsed: ParsedNER, result: EnrichmentResult, output_path: str | Path) -> str:
    """Generate a markdown comparison table: NER entities vs LLM enriched."""
    lines = [
        "# NER vs LLM Enrichment Comparison",
        "",
        f"**Model**: {result.model_name}",
        f"**Original entities (with codemaps)**: {result.original_entity_count}",
        f"**Enriched concepts**: {result.enriched_entity_count}",
        f"**Latency**: {result.latency_ms:.0f}ms",
        f"**Tokens**: {result.prompt_tokens} prompt + {result.completion_tokens} completion",
        "",
        "---",
        "",
        "## Original NER Entities (problems & procedures with codemaps)",
        "",
        "| # | Entity Text | Type | Assertion | IMO Lexical | ICD-10 |",
        "|---|------------|------|-----------|-------------|--------|",
    ]

    idx = 1
    for ent in parsed.entities:
        if ent.codemap and ent.semantic in ("problem", "procedure", "treatment"):
            icd = f"{ent.codemap.icd10_code} ({ent.codemap.icd10_title})" if ent.codemap.icd10_code else "—"
            lines.append(
                f"| {idx} | {ent.text} | {ent.semantic} | {ent.assertion} | "
                f"{ent.codemap.imo_lexical_title} | {icd} |"
            )
            idx += 1

    lines.extend([
        "",
        "---",
        "",
        "## LLM Enriched Concepts",
        "",
        "| # | Enriched Concept | Source Entities | Inference | Clinical Reasoning |",
        "|---|-----------------|----------------|-----------|-------------------|",
    ])

    for i, ee in enumerate(result.enriched_entities, 1):
        sources = ", ".join(ee.source_entity_texts)
        reasoning = ee.clinical_reasoning[:100] + "..." if len(ee.clinical_reasoning) > 100 else ee.clinical_reasoning
        lines.append(
            f"| {i} | {ee.enriched_concept} | {sources} | "
            f"{ee.inference_strength} | {reasoning} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Entity Linking Map",
        "",
        "Shows which NER entities were combined into each enriched concept:",
        "",
    ])

    for i, ee in enumerate(result.enriched_entities, 1):
        lines.append(f"### Enriched #{i}: {ee.enriched_concept}")
        lines.append("")
        for src_text in ee.source_entity_texts:
            # Find matching NER entity
            matched = next(
                (e for e in parsed.entities if e.text.lower().strip() == src_text.lower().strip()
                 or src_text.lower() in e.text.lower()),
                None
            )
            if matched and matched.codemap:
                lines.append(
                    f"- **\"{matched.text}\"** ({matched.semantic}) -> "
                    f"IMO: {matched.codemap.imo_lexical_title} | "
                    f"ICD-10: {matched.codemap.icd10_code or '—'}"
                )
            else:
                lines.append(f"- **\"{src_text}\"** (no codemap)")

        lines.append(f"- **Reasoning**: {ee.clinical_reasoning}")
        if ee.evidence_spans:
            lines.append(f"- **Evidence**: {'; '.join(ee.evidence_spans)}")
        lines.append("")

    md = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(md)

    return md
