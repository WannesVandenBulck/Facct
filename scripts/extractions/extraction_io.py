"""
Extractor settings and file handling used by run_extraction_batches.py (Grok batch API): the extractor
model, system prompt, file locations and JSON parsing.
"""

import json
import os

EXTRACTOR_PROVIDER = "grok"
EXTRACTOR_MODEL = "grok-4.20-0309-non-reasoning"
EXTRACTOR_TEMPERATURE = 0
EXTRACTOR_MAX_TOKENS = 10000

SYSTEM_PROMPT = ("You are an expert data analyst. Your task is to extract information from narratives and fill in a JSON "
                 "template. Return ONLY the completed JSON in a code block - no other text.")


def extraction_path(dataset, condition, narrative_provider, extractor, instance_idx, base="results/extractions"):
    return f"{base}/{dataset}/{condition}/{narrative_provider}/{extractor}/instance_{instance_idx}.json"


def raw_path(dataset, condition, narrative_provider, extractor, instance_idx, base="results/extractions"):
    return f"{base}/{dataset}/raw/{condition}/{narrative_provider}/{extractor}/instance_{instance_idx}.txt"


def parse_and_save(response, dataset, condition, narrative_provider, extractor, instance_idx, base="results/extractions"):
    """Save the raw response, parse the ```json block and save it. Returns (extraction or None, error or None)."""
    raw = raw_path(dataset, condition, narrative_provider, extractor, instance_idx, base)
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    with open(raw, "w", encoding="utf-8") as f:
        f.write(response or "")

    if not response or "```json" not in response:
        return None, "No JSON block found in response"
    start = response.find("```json") + 7
    end = response.find("```", start)
    try:
        extraction = json.loads(response[start:end if end != -1 else len(response)].strip())
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    out = extraction_path(dataset, condition, narrative_provider, extractor, instance_idx, base)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(extraction, f, indent=2)
    return extraction, None
