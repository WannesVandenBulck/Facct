#!/usr/bin/env python
"""
Generate narratives with provider batch APIs: one batch per (model, dataset, condition).

Workflow (run from the repo root; batches can take up to 24h, so nothing waits for them):

    python scripts/narrative_generation/run_batches.py build     # freeze all prompts to results/batches/requests/
    python scripts/narrative_generation/run_batches.py submit    # send every group not yet submitted
    python scripts/narrative_generation/run_batches.py status    # check progress (run again later)
    python scripts/narrative_generation/run_batches.py collect   # write finished results to results/narratives/
    python scripts/narrative_generation/run_batches.py retry     # resubmit failed / missing requests only

Options for submit / status / collect / retry:
    --datasets credit law       restrict to some datasets
    --conditions include_pa     restrict to some conditions
    --models gpt-4o-mini        restrict to some models (by model name)
    --test N                    test run: first N requests per group, kept apart under results/batches_test/
                                and results/narratives_test/ so they never mix with the real results

Every batch is recorded in results/batches/registry.json (batch id, request ids, state), so you can stop
and come back later. Narratives are written in the same format and folder structure as before:
    results/narratives/{dataset}/{condition}/{provider}/{model}/instance_{i}.json
    results/narratives/{dataset}/counterfactual_pa/{label}/{provider}/{model}/instance_{i}.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from llm_tools.batch.requests_builder import CONDITIONS, DATASETS, build_requests
from llm_tools.batch.providers import get_adapter

# ===== CONFIGURATION =====

# One entry per model. provider: openai | claude | gemini | mistral | grok | deepseek
# Use pinned model versions (dated snapshots) where the provider offers them.
# mode: "batch" (default) or "sync" (parallel normal calls; deepseek has no batch API).
# api_version (claude only): Claude API version 2026-10-21, the SDK default, rejects temperature,
# so the Claude batches are pinned to 2023-06-01 to keep temperature=0 like the other providers.
MODELS = [
    dict(provider="openai", model="gpt-4.1-2025-04-14", temperature=0, max_tokens=4096),
    dict(provider="mistral", model="mistral-large-2512", temperature=0, max_tokens=4096),
    dict(provider="deepseek", model="deepseek-v4-pro", temperature=0, max_tokens=4096, mode="sync"),
    dict(provider="claude", model="claude-sonnet-4-6", temperature=0, max_tokens=4096, api_version="2023-06-01"),
]

# ===== END CONFIGURATION =====


def paths(test):
    base = ROOT / ("results/batches_test" if test else "results/batches")
    narratives = ROOT / ("results/narratives_test" if test else "results/narratives")
    return base, narratives


def load_registry(base):
    f = base / "registry.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_registry(base, registry):
    base.mkdir(parents=True, exist_ok=True)
    (base / "registry.json").write_text(json.dumps(registry, indent=1), encoding="utf-8")


def requests_file(dataset, condition):
    return ROOT / "results" / "batches" / "requests" / f"{dataset}__{condition}.jsonl"


def load_requests(dataset, condition):
    f = requests_file(dataset, condition)
    if not f.exists():
        sys.exit(f"Missing {f.relative_to(ROOT)}: run 'build' first")
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def group_key(cfg, dataset, condition, attempt=0):
    return f"{cfg['provider']}/{cfg['model']}/{dataset}/{condition}" + (f"/retry{attempt}" if attempt else "")


def model_dir_name(model):
    return model.replace("/", "_").replace(":", "_")


def selected(args):
    models = [m for m in MODELS if not args.models or m["model"] in args.models]
    if not models:
        sys.exit("No models selected: fill in MODELS at the top of this script")
    for m in models:
        m.setdefault("temperature", 0)
        m.setdefault("max_tokens", 4096)
    datasets = args.datasets or DATASETS
    conditions = args.conditions or CONDITIONS
    return models, datasets, conditions


def find_cfg(provider, model):
    for m in MODELS:
        if m["provider"] == provider and m["model"] == model:
            m.setdefault("temperature", 0)
            m.setdefault("max_tokens", 4096)
            return m
    sys.exit(f"{provider}/{model} is in the registry but no longer in MODELS")


# ----------------------------------------------------------------------------- commands

def cmd_build(args):
    """Freeze every prompt to JSONL so all providers receive byte-identical requests."""
    total = 0
    for dataset in args.datasets or DATASETS:
        for condition in args.conditions or CONDITIONS:
            reqs = build_requests(dataset, condition)
            f = requests_file(dataset, condition)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
            total += len(reqs)
            print(f"  {dataset:8s} {condition:18s} {len(reqs):4d} requests -> {f.relative_to(ROOT)}")
    print(f"Built {total} requests per model.")


def submit_group(registry, cfg, dataset, condition, reqs, attempt=0):
    key = group_key(cfg, dataset, condition, attempt)
    name = key.replace("/", "__").replace(".", "-")
    batch_id = get_adapter(cfg).submit(reqs, cfg, name)
    registry[key] = {"provider": cfg["provider"], "model": cfg["model"], "dataset": dataset, "condition": condition,
                     "attempt": attempt, "batch_id": batch_id, "custom_ids": [r["custom_id"] for r in reqs],
                     "submitted_at": datetime.now().isoformat(timespec="seconds"), "state": "submitted",
                     "collected": False}
    print(f"  submitted {key}: {len(reqs)} requests (batch {batch_id})")


def cmd_submit(args):
    base, _ = paths(args.test)
    registry = load_registry(base)
    models, datasets, conditions = selected(args)
    for cfg in models:
        for dataset in datasets:
            for condition in conditions:
                key = group_key(cfg, dataset, condition)
                if key in registry:
                    print(f"  skip {key}: already submitted ({registry[key]['state']})")
                    continue
                reqs = load_requests(dataset, condition)
                if args.test:
                    reqs = reqs[:args.test]
                try:
                    submit_group(registry, cfg, dataset, condition, reqs)
                except Exception as e:  # report and carry on with the other groups; nothing is recorded for this one
                    body = getattr(getattr(e, "response", None), "text", "")
                    print(f"  FAILED {key}: {e} {body[:300]}")
                finally:
                    save_registry(base, registry)  # keep every successful submission even if a later one fails


def matches(entry, args):
    return ((not args.datasets or entry["dataset"] in args.datasets)
            and (not args.conditions or entry["condition"] in args.conditions)
            and (not args.models or entry["model"] in args.models))


def cmd_status(args):
    base, _ = paths(args.test)
    registry = load_registry(base)
    for key, entry in registry.items():
        if not matches(entry, args):
            continue
        if entry["state"] != "done":
            cfg = find_cfg(entry["provider"], entry["model"])
            try:
                s = get_adapter(cfg).status(entry["batch_id"], cfg)
                entry["state"], entry["detail"] = s["state"], s["detail"]
            except Exception as e:  # e.g. a temporary network error: keep the last known state
                entry["detail"] = f"status check failed ({type(e).__name__}), will retry"
        flag = "collected" if entry["collected"] else entry["state"]
        print(f"  {key:70s} {flag:10s} {entry.get('detail', '')}")
    save_registry(base, registry)


def write_narrative(narratives_dir, req, cfg, entry, res):
    cond_dir = f"counterfactual_pa/{req['label']}" if req["condition"] == "counterfactual_pa" else req["condition"]
    out_dir = narratives_dir / req["dataset"] / cond_dir / cfg["provider"] / model_dir_name(cfg["model"])
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": req["dataset"],
        "instance_idx": req["instance_idx"],
        "provider": cfg["provider"],
        "model": cfg["model"],
        "model_version": res.get("model_version"),
        "timestamp": datetime.now().isoformat(),
        "status": res["status"],
        "narrative": res["narrative"],
        "error": res["error"],
        "system_prompt": req["system"],
        "prompt": req["prompt"],
        "condition": req["condition"],
        "exclude_protected_attributes": req["condition"] == "exclude_pa",
        "label": req["label"],
        "attr_column": req["attr_column"],
        "override_value": req["override_value"],
        "target_value_label": req["target_value_label"],
        "temperature": cfg.get("temperature"),
        "max_tokens": cfg["max_tokens"],
        "api_version": cfg.get("api_version"),
        "usage": res.get("usage"),
        "batch_id": entry["batch_id"],
        "custom_id": req["custom_id"],
    }
    (out_dir / f"instance_{req['instance_idx']}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_collect(args):
    base, narratives_dir = paths(args.test)
    registry = load_registry(base)
    for key, entry in registry.items():
        if entry["collected"] or not matches(entry, args):
            continue
        cfg = find_cfg(entry["provider"], entry["model"])
        adapter = get_adapter(cfg)
        if entry["state"] != "done":
            s = adapter.status(entry["batch_id"], cfg)
            entry["state"], entry["detail"] = s["state"], s["detail"]
            if s["state"] != "done":
                print(f"  {key}: still running ({s['detail']})")
                continue
        if cfg["provider"] == "gemini" and cfg.get("mode") != "sync":
            results = adapter.fetch(entry["batch_id"], cfg, custom_ids=entry["custom_ids"])
        else:
            results = adapter.fetch(entry["batch_id"], cfg)

        reqs = {r["custom_id"]: r for r in load_requests(entry["dataset"], entry["condition"])}
        ok, failed = 0, []
        for cid in entry["custom_ids"]:
            res = results.get(cid)
            if res and res["status"] == "success":
                write_narrative(narratives_dir, reqs[cid], cfg, entry, res)
                ok += 1
            else:
                failed.append(cid)
        entry["collected"], entry["failed_ids"] = True, failed
        print(f"  {key}: {ok}/{len(entry['custom_ids'])} narratives written"
              + (f", {len(failed)} failed or missing (run 'retry')" if failed else ""))
        save_registry(base, registry)
    save_registry(base, registry)


def cmd_retry(args):
    base, _ = paths(args.test)
    registry = load_registry(base)
    for key, entry in list(registry.items()):
        if not entry["collected"] or not entry.get("failed_ids") or entry.get("retried") or not matches(entry, args):
            continue
        cfg = find_cfg(entry["provider"], entry["model"])
        reqs = {r["custom_id"]: r for r in load_requests(entry["dataset"], entry["condition"])}
        try:
            submit_group(registry, cfg, entry["dataset"], entry["condition"],
                         [reqs[c] for c in entry["failed_ids"]], attempt=entry["attempt"] + 1)
            entry["retried"] = True
        finally:
            save_registry(base, registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "submit", "status", "collect", "retry"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--test", type=int, metavar="N", help="test run with the first N requests per group")
    args = parser.parse_args()
    {"build": cmd_build, "submit": cmd_submit, "status": cmd_status,
     "collect": cmd_collect, "retry": cmd_retry}[args.command](args)


if __name__ == "__main__":
    main()
