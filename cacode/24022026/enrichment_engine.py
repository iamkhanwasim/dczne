"""
LLM Enrichment Engine.
Takes parsed NER entities + clinical note → LLM → enriched clinical concepts.
Focus: cross-sentence context linking to produce richer, more specific entities.
"""

import json
import logging
from dataclasses import dataclass, field

from .ner_parser import ParsedNER, NEREntity
from .model_factory import BaseLLM, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class EnrichedEntity:
    """An enriched clinical concept produced by LLM cross-sentence linking."""
    enriched_concept: str
    source_entity_ids: list[str]
    source_entity_texts: list[str]
    clinical_reasoning: str
    inference_strength: str  # explicit | strong_suggestion | weak_suggestion
    evidence_spans: list[str] = field(default_factory=list)


@dataclass
class EnrichmentResult:
    """Full result of LLM enrichment for one clinical note."""
    model_name: str
    enriched_entities: list[EnrichedEntity]
    original_entity_count: int
    enriched_entity_count: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw_llm_response: str = ""


# ── Prompt Construction ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical NLP expert specializing in medical coding. Your task is to analyze extracted NER entities from a clinical note and produce enriched clinical concepts by linking related entities across sentences.

RULES:
1. Only combine entities that have a genuine clinical relationship supported by the note text.
2. Produce enriched concepts that are MORE SPECIFIC than the individual entities.
3. Include laterality, severity, causation, and temporal context when present in the note.
4. Each enriched concept should cite which source entities it combines.
5. Rate inference strength:
   - "explicit": directly stated in the note (e.g., "diabetic osteomyelitis")
   - "strong_suggestion": strongly implied by clinical context (e.g., osteomyelitis + diabetes + amputation → diabetic osteomyelitis)
   - "weak_suggestion": possible but requires clinical judgment
6. Do NOT invent information not present in the note.
7. Do NOT fix NER errors or suggest new codes — only produce enriched concept text.

Respond ONLY with valid JSON, no markdown backticks, no preamble."""


def _build_entity_summary(parsed: ParsedNER) -> str:
    """Build a concise entity summary for the LLM prompt."""
    lines = []
    for ent in parsed.entities:
        if ent.codemap or ent.semantic in ("problem", "procedure", "treatment", "drug"):
            imo = ""
            icd = ""
            if ent.codemap:
                imo = f" | IMO: {ent.codemap.imo_lexical_title}"
                if ent.codemap.icd10_code:
                    icd = f" | ICD10: {ent.codemap.icd10_code} ({ent.codemap.icd10_title})"

            lines.append(
                f"  - [{ent.semantic}] \"{ent.text}\" "
                f"(pos {ent.begin}-{ent.end}, assertion={ent.assertion})"
                f"{imo}{icd}"
            )

    return "\n".join(lines)


def _build_relation_summary(parsed: ParsedNER) -> str:
    """Build a concise relation summary for the LLM prompt."""
    lines = []
    note = parsed.clinical_note
    for rel in parsed.relations:
        # Extract text spans from entity IDs (format: begin_end_Entity_semantic)
        parts = rel.from_entity_id.split("_")
        from_text = note[int(parts[0]):int(parts[1])] if len(parts) >= 2 else ""
        parts = rel.to_entity_id.split("_")
        to_text = note[int(parts[0]):int(parts[1])] if len(parts) >= 2 else ""
        lines.append(f"  - {rel.semantic}: \"{from_text}\" → \"{to_text}\"")
    return "\n".join(lines)


def build_enrichment_prompt(parsed: ParsedNER) -> str:
    """Build the full enrichment prompt from parsed NER output."""
    entity_summary = _build_entity_summary(parsed)
    relation_summary = _build_relation_summary(parsed)

    prompt = f"""CLINICAL NOTE:
\"\"\"
{parsed.clinical_note}
\"\"\"

EXTRACTED NER ENTITIES:
{entity_summary}

EXTRACTED RELATIONS:
{relation_summary}

TASK: Analyze the clinical note and the extracted entities above. Identify entities that should be linked across sentences to form richer, more specific clinical concepts. For each enriched concept, explain the clinical reasoning.

Respond with this exact JSON structure:
{{
  "enriched_entities": [
    {{
      "enriched_concept": "The enriched clinical concept text (e.g., 'Osteomyelitis of left great toe due to poorly controlled type 2 diabetes mellitus')",
      "source_entities": ["entity text 1", "entity text 2"],
      "clinical_reasoning": "Why these entities are clinically related",
      "inference_strength": "explicit | strong_suggestion | weak_suggestion",
      "evidence_spans": ["exact text from the note supporting this link"]
    }}
  ]
}}"""

    return prompt


# ── Enrichment Execution ─────────────────────────────────────────────

def _parse_llm_response(response_text: str) -> list[dict]:
    """Parse LLM JSON response, handling common issues."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.startswith("json"):
        text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
        return data.get("enriched_entities", [])
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Raw response: {text[:500]}")
        return []


def _match_source_entities(source_texts: list[str], parsed: ParsedNER) -> list[str]:
    """Match LLM-reported source entity texts back to NER entity IDs."""
    matched_ids = []
    for src in source_texts:
        src_lower = src.lower().strip()
        for ent in parsed.entities:
            if ent.text.lower().strip() == src_lower or src_lower in ent.text.lower():
                matched_ids.append(ent.id)
                break
    return matched_ids


def enrich_entities(parsed: ParsedNER, llm: BaseLLM, temperature: float = 0.1) -> EnrichmentResult:
    """
    Main enrichment function.
    Takes parsed NER output + LLM → produces enriched clinical concepts.
    """
    prompt = build_enrichment_prompt(parsed)
    logger.info(f"Sending enrichment prompt to {llm.model_name} ({len(prompt)} chars)")

    response: ModelResponse = llm.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
    )

    logger.info(f"LLM response: {response.completion_tokens} tokens, {response.latency_ms:.0f}ms")

    raw_entities = _parse_llm_response(response.content)

    enriched = []
    for raw in raw_entities:
        source_texts = raw.get("source_entities", [])
        source_ids = _match_source_entities(source_texts, parsed)

        enriched.append(EnrichedEntity(
            enriched_concept=raw.get("enriched_concept", ""),
            source_entity_ids=source_ids,
            source_entity_texts=source_texts,
            clinical_reasoning=raw.get("clinical_reasoning", ""),
            inference_strength=raw.get("inference_strength", "weak_suggestion"),
            evidence_spans=raw.get("evidence_spans", []),
        ))

    codemap_entities = parsed.get_entities_with_codemaps()

    return EnrichmentResult(
        model_name=llm.model_name,
        enriched_entities=enriched,
        original_entity_count=len(codemap_entities),
        enriched_entity_count=len(enriched),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        raw_llm_response=response.content,
    )
