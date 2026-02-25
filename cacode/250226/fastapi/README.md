# CA LLM Enrichment

A clinical NLP pipeline that uses a local LLM (via Ollama) to enrich Named Entity Recognition (NER) output from clinical notes — linking related entities across sentences into more specific, coded clinical concepts.

**Now available as both a REST API (FastAPI) and CLI application!**

## What it does

1. **Parses** structured NER JSON output (entities, relations, codemaps) from a clinical NLP system
2. **Builds** a prompt summarising entities and their relationships
3. **Sends** the prompt to a local Ollama LLM
4. **Parses** the LLM response into enriched clinical concepts
5. **Outputs** a structured JSON file with enriched entities and clinical reasoning

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally at `http://localhost:11434`
- At least one supported model pulled in Ollama

### Installation

```bash
pip install -r requirements.txt
```

## Supported Models

| Model | Ollama tag |
|-------|-----------|
| Qwen 2.5 7B | `qwen2.5:7b` |
| Llama 3.1 8B | `llama3.1:8b` |

Pull a model with:
```bash
ollama pull qwen2.5:7b
```

## Input

Place your NER JSON file at `data/ner_ext.json`. The expected format is the CCP NER output with `content`, `indexes`, entities (with `begin`, `end`, `semantic`, `attrs`), and relations.

## Usage

### Option 1: FastAPI REST API (Recommended)

Start the API server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive documentation is at `http://localhost:8000/docs`.

#### API Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**List Available Models**
```bash
curl http://localhost:8000/models
```

**Enrich NER Entities (Single Model)**
```bash
curl -X POST http://localhost:8000/enrich \
  -H "Content-Type: application/json" \
  -d '{
    "ner_data": { ... },  # Your NER JSON data
    "model_name": "qwen2.5:7b",
    "temperature": 0.1
  }'
```

**Benchmark Multiple Models**
```bash
curl -X POST http://localhost:8000/enrich/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "ner_data": { ... },
    "model_names": ["qwen2.5:7b", "llama3.1:8b"],
    "temperature": 0.1
  }'
```

**Python Client Example**
```python
import requests
import json

# Load your NER data
with open("data/ner_ext.json") as f:
    ner_data = json.load(f)

# Call the API
response = requests.post(
    "http://localhost:8000/enrich",
    json={
        "ner_data": ner_data,
        "model_name": "qwen2.5:7b",
        "temperature": 0.1
    }
)

result = response.json()
print(f"Enriched {result['metadata']['enriched_entity_count']} concepts")
```

### Option 2: CLI Application

Run enrichment with a single model:
```bash
python run.py --input data/ner_ext.json --model qwen2.5:7b
```

Benchmark multiple models:
```bash
python run.py --input data/ner_ext.json --model qwen2.5:7b --model llama3.1:8b
```

Mock mode (no Ollama needed):
```bash
python run.py --input data/ner_ext.json --model mock
```

### Option 3: Jupyter Notebook

Open and run `test.ipynb` top to bottom. The key steps are:

```python
# 1. Parse NER output
parsed = parse_ner_json("data/ner_ext.json")

# 2. Run enrichment
llm = create_model("qwen2.5:7b")
result = enrich_entities(parsed, llm)

# 3. Save output
generate_enriched_json(parsed, result, "enriched_qwen2.5_7b.json")
```

## Output

### API Response Format

The API returns JSON with the following structure:

```json
{
  "metadata": {
    "model": "qwen2.5:7b",
    "original_entity_count": 5,
    "enriched_entity_count": 2,
    "prompt_tokens": 450,
    "completion_tokens": 280,
    "latency_ms": 1234.5
  },
  "enriched_entities": [
    {
      "enriched_concept": "Osteomyelitis of left great toe due to poorly controlled type 2 diabetes mellitus",
      "inference_strength": "strong_suggestion",
      "clinical_reasoning": "Osteomyelitis in a diabetic patient with poor glycemic control indicates diabetic osteomyelitis...",
      "evidence_spans": ["poorly controlled diabetes", "osteomyelitis", "left great toe"],
      "source_entities": [
        {
          "id": "123_145_Entity_problem",
          "text": "osteomyelitis",
          "semantic": "problem",
          "assertion": "present",
          "original_imo": {
            "lexical_title": "Osteomyelitis",
            "lexical_code": "IMO123"
          },
          "original_icd10": {
            "code": "M86.9",
            "title": "Osteomyelitis, unspecified"
          }
        }
      ],
      "source_entity_texts": ["osteomyelitis", "poorly controlled diabetes", "left great toe"]
    }
  ]
}
```

### CLI Output Files

When using the CLI (`run.py`), files are saved to the `output/` directory:
- `enriched_<model_name>.json` - Structured enrichment results
- `comparison_<model_name>.md` - Side-by-side comparison table
- `raw_response_<model_name>.txt` - Raw LLM response
- `benchmark_summary.md` - Multi-model comparison (when using multiple models)

## Key Classes

| Class | Purpose |
|-------|---------|
| `ParsedNER` | Container for parsed NER output (entities + relations) |
| `NEREntity` | Single NER entity with text, span, semantic type, and codemap |
| `CodeMap` | IMO, ICD-10, and SNOMED codes for an entity |
| `OllamaLLM` | Client for local Ollama model inference |
| `EnrichmentResult` | Full enrichment output including LLM metadata |
| `EnrichedEntity` | A single enriched clinical concept produced by the LLM |

## Project Structure

```
ca-llm-enrichement/
├── main.py                       # FastAPI application
├── run.py                        # CLI application
├── requirements.txt              # Python dependencies
├── test.ipynb                    # Jupyter notebook
├── data/
│   └── ner_ext.json             # Input NER JSON (example)
├── output/                       # CLI output directory
│   ├── enriched_*.json          # Enrichment results
│   ├── comparison_*.md          # Comparison tables
│   └── benchmark_summary.md     # Multi-model comparison
└── src/                          # Core modules
    ├── ner_parser.py            # NER JSON parser
    ├── model_factory.py         # LLM model factory (Ollama/Mock)
    ├── enrichment_engine.py     # Enrichment logic
    └── output_formatter.py      # Output generation
```

## API Features

- **RESTful API**: Clean, well-documented REST endpoints
- **Interactive Docs**: Auto-generated Swagger UI at `/docs`
- **Model Flexibility**: Support for multiple Ollama models and mock mode
- **Benchmarking**: Compare multiple models in a single request
- **Health Checks**: Monitor API and Ollama service status
- **Validation**: Pydantic models for request/response validation
- **Error Handling**: Detailed error messages and status codes

## Development

**Start the API in development mode:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Run tests (if available):**
```bash
pytest
```

**Check API health:**
```bash
curl http://localhost:8000/health
```

## Deployment

The FastAPI application can be deployed using:
- **Docker**: Containerize with `uvicorn` as the entry point
- **Cloud platforms**: Deploy to AWS, GCP, Azure, Heroku, etc.
- **Reverse proxy**: Use nginx or Apache as a reverse proxy

Example Dockerfile:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
