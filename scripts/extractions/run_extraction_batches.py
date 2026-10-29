#!/usr/bin/env python
"""
Extract all narratives with Grok through the xAI batch API: one batch per (dataset, narrative provider, condition).

Workflow (run from the repo root):

    python scripts/extractions/run_extraction_batches.py submit --datasets credit
    python scripts/extractions/run_extraction_batches.py status
    python scripts/extractions/run_extraction_batches.py collect
    python scripts/extractions/run_extraction_batches.py retry      # resubmit failed / unparseable extractions only

Options:
    --datasets credit law        restrict to some datasets (default: all)
    --providers openai claude    narrative providers to extract (default: all four; missing ones are skipped)
    --conditions include_pa      include_pa / exclude_pa / counterfactual_pa (default: all three)
    --test N                     test run: first N narratives per condition / label, kept apart under
                                 results/extraction_batches_test/ and results/extractions_test/

submit skips narratives that already have an extraction, so it can be rerun safely. The extractor prompts that are
sent are frozen in results/extraction_batches/requests/; results are written to:
    results/extractions/{dataset}/{condition}/{narrative_provider}/grok/instance_{i}.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from llm_tools.batch.providers import XAIBatch
from scripts.extractions.extraction_io import (EXTRACTOR_PROVIDER, EXTRACTOR_MODEL, EXTRACTOR_TEMPERATURE,
                                               EXTRACTOR_MAX_TOKENS, SYSTEM_PROMPT, extraction_path, parse_and_save)
from scripts.extractions.generate_extractor_prompt import generate_extractor_prompt

DATASETS = ["credit", "law", "saudi", "student"]
NARRATIVE_PROVIDERS = ["openai", "mistral", "claude", "deepseek"]
CONDITIONS = ["include_pa", "exclude_pa", "counterfactual_pa"]
CFG = dict(provider=EXTRACTOR_PROVIDER, model=EXTRACTOR_MODEL, temperature=EXTRACTOR_TEMPERATURE, max_tokens=EXTRACTOR_MAX_TOKENS)


def paths(test):
    batches = Path("results/extraction_batches_test" if test else "results/extraction_batches")
    extractions = "results/extractions_test" if test else "results/extractions"
    return batches, extractions


def load_registry(base):
    f = base / "registry.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_registry(base, registry):
    base.mkdir(parents=True, exist_ok=True)
    (base / "registry.json").write_text(json.dumps(registry, indent=1), encoding="utf-8")


def narrative_instances(dataset, condition, provider):
    """Instance ids that have a successful narrative for this condition folder and provider."""
    ids = []
    for f in Path("results/narratives", dataset, condition, provider).glob("*/instance_*.json"):
        if json.loads(f.read_text(encoding="utf-8")).get("status") == "success":
            ids.append(int(f.stem.split("_")[1]))
    return sorted(ids)


def condition_folders(dataset, condition):
    """include_pa / exclude_pa as is; counterfactual_pa expands to every label folder on disk."""
    if condition != "counterfactual_pa":
        return [condition]
    base = Path("results/narratives", dataset, "counterfactual_pa")
    return [f"counterfactual_pa/{d.name}" for d in sorted(base.iterdir()) if d.is_dir()] if base.exists() else []


def build_requests(dataset, provider, condition, limit, out_base):
    reqs = []
    for folder in condition_folders(dataset, condition):
        for i in narrative_instances(dataset, folder, provider)[:limit]:
            if os.path.exists(extraction_path(dataset, folder, provider, EXTRACTOR_PROVIDER, i, out_base)):
                continue  # already extracted
            prompt = generate_extractor_prompt(dataset, i, provider, "shap", condition=folder)
            if prompt.startswith("Error:"):
                sys.exit(prompt)
            reqs.append({"custom_id": f"{dataset}__{folder.replace('/', '__')}__{provider}__{i}", "dataset": dataset,
                         "condition": folder, "narrative_provider": provider, "instance_idx": i,
                         "system": SYSTEM_PROMPT, "prompt": prompt})
    return reqs


def submit_group(registry, base, key, reqs, attempt=0):
    req_file = base / "requests" / f"{key.replace('/', '__')}.jsonl"
    req_file.parent.mkdir(parents=True, exist_ok=True)
    req_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
    batch_id = XAIBatch().submit(reqs, CFG, key.replace("/", "__"))
    registry[key] = {"batch_id": batch_id, "requests_file": str(req_file), "custom_ids": [r["custom_id"] for r in reqs],
                     "attempt": attempt, "submitted_at": datetime.now().isoformat(timespec="seconds"),
                     "state": "submitted", "collected": False}
    print(f"  submitted {key}: {len(reqs)} extractions (batch {batch_id})")


def cmd_submit(args):
    base, out_base = paths(args.test)
    registry = load_registry(base)
    for dataset in args.datasets:
        for provider in args.providers:
            if not Path("results/narratives", dataset, "include_pa", provider).exists():
                print(f"  skip {dataset}/{provider}: no narratives")
                continue
            for condition in args.conditions:
                key = f"{dataset}/{provider}/{condition}"
                if key in registry and not registry[key]["collected"]:
                    print(f"  skip {key}: batch still open ({registry[key]['state']})")
                    continue
                reqs = build_requests(dataset, provider, condition, args.test, out_base)
                if not reqs:
                    print(f"  skip {key}: everything already extracted")
                    continue
                attempt = registry[key]["attempt"] + 1 if key in registry else 0
                try:
                    submit_group(registry, base, key, reqs, attempt)
                except Exception as e:
                    body = getattr(getattr(e, "response", None), "text", "")
                    print(f"  FAILED {key}: {e} {body[:300]}")
                finally:
                    save_registry(base, registry)


def cmd_status(args):
    base, _ = paths(args.test)
    registry = load_registry(base)
    for key, entry in registry.items():
        if entry["state"] != "done":
            try:
                s = XAIBatch().status(entry["batch_id"], CFG)
                entry["state"], entry["detail"] = s["state"], s["detail"]
            except Exception as e:  # e.g. a temporary network error: keep the last known state
                entry["detail"] = f"status check failed ({type(e).__name__}), will retry"
        print(f"  {key:55s} {'collected' if entry['collected'] else entry['state']:10s} {entry.get('detail', '')}")
    save_registry(base, registry)


def cmd_collect(args):
    base, out_base = paths(args.test)
    registry = load_registry(base)
    for key, entry in registry.items():
        if entry["collected"]:
            continue
        if entry["state"] != "done":
            s = XAIBatch().status(entry["batch_id"], CFG)
            entry["state"], entry["detail"] = s["state"], s["detail"]
            if s["state"] != "done":
                print(f"  {key}: still running ({s['detail']})")
                continue
        results = XAIBatch().fetch(entry["batch_id"], CFG)
        reqs = {r["custom_id"]: r for r in map(json.loads, Path(entry["requests_file"]).read_text(encoding="utf-8").splitlines())}
        ok, failed = 0, []
        for cid in entry["custom_ids"]:
            r, res = reqs[cid], results.get(cid)
            if not res or res["status"] != "success":
                failed.append(cid)
                continue
            extraction, error = parse_and_save(res["narrative"], r["dataset"], r["condition"], r["narrative_provider"],
                                               EXTRACTOR_PROVIDER, r["instance_idx"], out_base)
            if extraction is None:
                failed.append(cid)
            else:
                ok += 1
        entry["collected"], entry["failed_ids"] = True, failed
        print(f"  {key}: {ok}/{len(entry['custom_ids'])} extractions written"
              + (f", {len(failed)} failed or unparseable (run 'retry')" if failed else ""))
        save_registry(base, registry)


def cmd_retry(args):
    """Failed or unparseable extractions have no output file, so a new submit picks up exactly those."""
    base, _ = paths(args.test)
    registry = load_registry(base)
    pending = [k for k, e in registry.items() if e["collected"] and e.get("failed_ids")]
    if not pending:
        print("  nothing to retry")
        return
    for key in pending:
        dataset, provider, condition = key.split("/")
        a = argparse.Namespace(**{**vars(args), "datasets": [dataset], "providers": [provider], "conditions": [condition]})
        cmd_submit(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["submit", "status", "collect", "retry"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--providers", nargs="+", choices=NARRATIVE_PROVIDERS, default=NARRATIVE_PROVIDERS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--test", type=int, metavar="N", help="test run with the first N narratives per condition / label")
    args = parser.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect, "retry": cmd_retry}[args.command](args)


if __name__ == "__main__":
    main()
