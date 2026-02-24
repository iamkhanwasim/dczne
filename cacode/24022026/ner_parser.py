"""
NER Output Parser.
Reads the in-house NER JSON format and produces clean structured entities.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeMap:
    imo_lexical_title: str = ""
    imo_lexical_code: str = ""
    imo_confidence: str = ""
    icd10_code: str = ""
    icd10_title: str = ""
    snomed_code: str = ""
    snomed_title: str = ""


@dataclass
class NEREntity:
    id: str
    text: str
    begin: int
    end: int
    semantic: str  # problem, procedure, drug, test, treatment, bodyloc, temporal, etc.
    assertion: str = "present"
    section: str = ""
    origin: str = ""  # dictionary | model
    sentence_prob: float | None = None
    concept_prob: float | None = None
    linked_entity_ids: list[str] = field(default_factory=list)
    codemap: CodeMap | None = None


@dataclass
class NERRelation:
    id: str
    semantic: str  # problem-bodyloc, problem-temporal, drug-route, etc.
    from_entity_id: str
    to_entity_id: str
    from_semantic: str
    to_semantic: str


@dataclass
class ParsedNER:
    clinical_note: str
    entities: list[NEREntity]
    relations: list[NERRelation]

    def get_entity_by_span(self, begin: int, end: int) -> NEREntity | None:
        for e in self.entities:
            if e.begin == begin and e.end == end:
                return e
        return None

    def get_problems(self) -> list[NEREntity]:
        return [e for e in self.entities if e.semantic == "problem" and e.assertion == "present"]

    def get_procedures(self) -> list[NEREntity]:
        return [e for e in self.entities if e.semantic in ("procedure", "treatment") and e.assertion == "present"]

    def get_entities_with_codemaps(self) -> list[NEREntity]:
        return [e for e in self.entities if e.codemap is not None]

    def get_related_entities(self, entity: NEREntity) -> list[tuple[str, NEREntity]]:
        """Get all entities related to the given entity, with relation type."""
        related = []
        for r in self.relations:
            if r.from_entity_id == entity.id:
                target = next((e for e in self.entities if e.id == self._span_to_id(r.to_entity_id)), None)
                if target:
                    related.append((r.semantic, target))
            elif r.to_entity_id == entity.id:
                source = next((e for e in self.entities if e.id == self._span_to_id(r.from_entity_id)), None)
                if source:
                    related.append((r.semantic, source))
        return related

    @staticmethod
    def _span_to_id(span_ref: str) -> str:
        return span_ref


def _parse_codemap(codemaps_raw: dict | None) -> CodeMap | None:
    if not codemaps_raw:
        return None

    imo = codemaps_raw.get("imo", {})
    icd10_codes = codemaps_raw.get("icd10cm", {}).get("codes", [])
    snomed_codes = codemaps_raw.get("snomedInternational", {}).get("codes", [])

    return CodeMap(
        imo_lexical_title=imo.get("lexical_title", ""),
        imo_lexical_code=imo.get("lexical_code", ""),
        imo_confidence=imo.get("confidence", ""),
        icd10_code=icd10_codes[0]["code"] if icd10_codes else "",
        icd10_title=icd10_codes[0]["title"] if icd10_codes else "",
        snomed_code=snomed_codes[0]["code"] if snomed_codes else "",
        snomed_title=snomed_codes[0]["title"] if snomed_codes else "",
    )


def parse_ner_json(filepath: str | Path) -> ParsedNER:
    """Parse raw in-house NER JSON into structured format."""
    with open(filepath) as f:
        data = json.load(f)

    content = data["content"]
    entities = []
    relations = []

    for _pos, types in data["indexes"].items():
        if "Entity" in types:
            for ent_key, ent_val in types["Entity"].items():
                text = content[ent_val["begin"]:ent_val["end"]]
                attrs = ent_val.get("attrs", {})
                codemaps_raw = attrs.get("codemaps")
                codemaps = json.loads(codemaps_raw) if codemaps_raw else None

                linked = attrs.get("linked_entities", "")
                linked_ids = [lid.strip("[]") for lid in linked.split(",")] if linked else []

                entities.append(NEREntity(
                    id=ent_key,
                    text=text,
                    begin=ent_val["begin"],
                    end=ent_val["end"],
                    semantic=ent_val.get("semantic", ""),
                    assertion=attrs.get("assertion", "present"),
                    section=attrs.get("section", ""),
                    origin=attrs.get("origin", ""),
                    sentence_prob=float(attrs["sentence_prob"]) if "sentence_prob" in attrs else None,
                    concept_prob=float(attrs["concept_prob"]) if "concept_prob" in attrs else None,
                    linked_entity_ids=linked_ids,
                    codemap=_parse_codemap(codemaps),
                ))

        if "Relation" in types:
            for rel_key, rel_val in types["Relation"].items():
                from_ent = rel_val["fromEnt"]
                to_ent = rel_val["toEnt"]

                # Build entity IDs from span + semantic
                from_id = f"{from_ent['begin']}_{from_ent['end']}_Entity_{from_ent['semantic']}"
                to_id = f"{to_ent['begin']}_{to_ent['end']}_Entity_{to_ent['semantic']}"

                relations.append(NERRelation(
                    id=rel_key,
                    semantic=rel_val.get("semantic", ""),
                    from_entity_id=from_id,
                    to_entity_id=to_id,
                    from_semantic=from_ent.get("semantic", ""),
                    to_semantic=to_ent.get("semantic", ""),
                ))

    return ParsedNER(clinical_note=content, entities=entities, relations=relations)
