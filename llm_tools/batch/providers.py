"""
Provider adapters for batch generation. Every adapter offers the same three calls:

    submit(requests, cfg, name)  -> batch_id
    status(batch_id, cfg)        -> {"state": "running" | "done", "detail": str}
    fetch(batch_id, cfg)         -> {custom_id: {"status", "narrative", "error", "model_version", "usage"}}

`requests` are the provider-independent dicts from requests_builder.make_request; `cfg` is one entry of
MODELS in scripts/narrative_generation/run_batches.py (provider, model, temperature, max_tokens).
"state" is "done" once the provider has finished (also when it ended early, e.g. expired or cancelled):
whatever results exist are then fetched and missing/failed requests can be retried.
"""

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests as http
import yaml

ROOT = Path(__file__).parent.parent.parent
KEYS = yaml.safe_load(open(ROOT / "config" / "keys.yaml", encoding="utf-8"))


def _result(narrative=None, error=None, model_version=None, usage=None):
    return {"status": "success" if narrative and not error else "error", "narrative": narrative,
            "error": error, "model_version": model_version, "usage": usage}


def _chat_body(req, cfg, token_param="max_completion_tokens"):
    body = {"model": cfg["model"],
            "messages": [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["prompt"]}],
            token_param: cfg["max_tokens"]}
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    return body


def _parse_chat_completion(body):
    """OpenAI-style chat completion body -> result."""
    choice = body["choices"][0]
    return _result(choice["message"].get("content"), None, body.get("model"), body.get("usage"))


# ----------------------------------------------------------------------------- OpenAI

class OpenAIBatch:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=KEYS["openai_key"])

    def submit(self, requests, cfg, name):
        lines = [json.dumps({"custom_id": r["custom_id"], "method": "POST", "url": "/v1/chat/completions",
                             "body": _chat_body(r, cfg)}, ensure_ascii=False) for r in requests]
        f = self.client.files.create(file=(f"{name}.jsonl", "\n".join(lines).encode("utf-8")), purpose="batch")
        batch = self.client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                           completion_window="24h", metadata={"name": name[:512]})
        return batch.id

    def status(self, batch_id, cfg):
        b = self.client.batches.retrieve(batch_id)
        c = b.request_counts
        detail = f"{b.status} ({c.completed}/{c.total} done, {c.failed} failed)" if c else b.status
        return {"state": "done" if b.status in ("completed", "failed", "expired", "cancelled") else "running", "detail": detail}

    def fetch(self, batch_id, cfg):
        b = self.client.batches.retrieve(batch_id)
        out = {}
        for file_id in (b.output_file_id, b.error_file_id):
            if not file_id:
                continue
            for line in self.client.files.content(file_id).text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                resp, err = row.get("response") or {}, row.get("error")
                if resp.get("status_code") == 200 and not err:
                    out[row["custom_id"]] = _parse_chat_completion(resp["body"])
                else:
                    out[row["custom_id"]] = _result(error=json.dumps(err or resp.get("body")))
        return out


# ----------------------------------------------------------------------------- Anthropic

class AnthropicBatch:
    """api_version: e.g. "2023-06-01" to keep temperature, which API version 2026-10-21 (the SDK 2.x default)
    rejects. Batch results always follow the version the batch was created with."""

    def __init__(self, api_version=None):
        import anthropic
        headers = {"anthropic-version": api_version} if api_version else None
        self.client = anthropic.Anthropic(api_key=KEYS["claude_key"], default_headers=headers)

    def submit(self, requests, cfg, name):
        items = []
        for r in requests:
            params = {"model": cfg["model"], "max_tokens": cfg["max_tokens"], "system": r["system"],
                      "messages": [{"role": "user", "content": r["prompt"]}]}
            if cfg.get("temperature") is not None:
                params["temperature"] = cfg["temperature"]
            items.append({"custom_id": r["custom_id"], "params": params})
        return self.client.messages.batches.create(requests=items).id

    def status(self, batch_id, cfg):
        b = self.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        detail = f"{b.processing_status} ({c.succeeded} succeeded, {c.errored} errored, {c.processing} processing)"
        return {"state": "done" if b.processing_status == "ended" else "running", "detail": detail}

    def fetch(self, batch_id, cfg):
        out = {}
        for entry in self.client.messages.batches.results(batch_id):
            res = entry.result
            if res.type == "succeeded":
                msg = res.message
                text = "".join(block.text for block in msg.content if block.type == "text")
                out[entry.custom_id] = _result(text, None, msg.model, msg.usage.model_dump())
            else:
                detail = getattr(res, "error", None)
                out[entry.custom_id] = _result(error=f"{res.type}: {detail.model_dump_json() if detail else ''}")
        return out


# ----------------------------------------------------------------------------- Gemini

class GeminiBatch:
    """Inline batch; responses come back in request order, so the custom_ids are kept in the registry."""

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=KEYS["gemini_key"])

    def submit(self, requests, cfg, name):
        src = []
        for r in requests:
            config = {"system_instruction": r["system"], "max_output_tokens": cfg["max_tokens"]}
            if cfg.get("temperature") is not None:
                config["temperature"] = cfg["temperature"]
            src.append({"contents": [{"role": "user", "parts": [{"text": r["prompt"]}]}], "config": config})
        job = self.client.batches.create(model=cfg["model"], src=src, config={"display_name": name[:128]})
        return job.name

    def status(self, batch_id, cfg):
        state = self.client.batches.get(name=batch_id).state.name
        terminal = ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")
        return {"state": "done" if state in terminal else "running", "detail": state}

    def fetch(self, batch_id, cfg, custom_ids=None):
        job = self.client.batches.get(name=batch_id)
        responses = (job.dest.inlined_responses or []) if job.dest else []
        if custom_ids is None or len(responses) != len(custom_ids):
            raise RuntimeError(f"Gemini returned {len(responses)} responses for {len(custom_ids or [])} requests; cannot match by order")
        out = {}
        for cid, item in zip(custom_ids, responses):
            if item.error or not item.response:
                out[cid] = _result(error=str(item.error))
            else:
                usage = item.response.usage_metadata.model_dump() if item.response.usage_metadata else None
                out[cid] = _result(item.response.text, None, item.response.model_version, usage)
        return out


# ----------------------------------------------------------------------------- Mistral

class MistralBatch:
    BASE = "https://api.mistral.ai/v1"

    def __init__(self):
        self.headers = {"Authorization": f"Bearer {KEYS['mistral_key']}"}

    def submit(self, requests, cfg, name):
        lines = [json.dumps({"custom_id": r["custom_id"], "body": _chat_body(r, cfg, token_param="max_tokens")},
                            ensure_ascii=False) for r in requests]
        up = http.post(f"{self.BASE}/files", headers=self.headers, data={"purpose": "batch"},
                       files={"file": (f"{name}.jsonl", io.BytesIO("\n".join(lines).encode("utf-8")))})
        up.raise_for_status()
        job = http.post(f"{self.BASE}/batch/jobs", headers=self.headers,
                        json={"input_files": [up.json()["id"]], "model": cfg["model"],
                              "endpoint": "/v1/chat/completions", "metadata": {"name": name}})
        job.raise_for_status()
        return job.json()["id"]

    def _job(self, batch_id):
        r = http.get(f"{self.BASE}/batch/jobs/{batch_id}", headers=self.headers)
        r.raise_for_status()
        return r.json()

    def status(self, batch_id, cfg):
        j = self._job(batch_id)
        detail = f"{j['status']} ({j.get('succeeded_requests')}/{j.get('total_requests')} succeeded, {j.get('failed_requests')} failed)"
        return {"state": "running" if j["status"] in ("QUEUED", "RUNNING", "CANCELLATION_REQUESTED") else "done", "detail": detail}

    def _download(self, file_id):
        r = http.get(f"{self.BASE}/files/{file_id}/content", headers=self.headers)
        r.raise_for_status()
        text = r.content.decode("utf-8", errors="replace")
        if not text.lstrip().startswith("{"):  # docs show a base64 example; accept both
            text = base64.b64decode(text).decode("utf-8")
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    def fetch(self, batch_id, cfg):
        j = self._job(batch_id)
        out = {}
        for file_id in (j.get("output_file"), j.get("error_file")):
            if not file_id:
                continue
            for row in self._download(file_id):
                resp = row.get("response") or {}
                body = resp.get("body", resp)
                if row.get("error") or "choices" not in body:
                    out[row["custom_id"]] = _result(error=json.dumps(row.get("error") or resp))
                else:
                    out[row["custom_id"]] = _parse_chat_completion(body)
        return out


# ----------------------------------------------------------------------------- xAI (Grok)

class XAIBatch:
    BASE = "https://api.x.ai/v1"
    CHUNK = 100

    def __init__(self):
        self.headers = {"Authorization": f"Bearer {KEYS['grok_key']}"}

    def submit(self, requests, cfg, name):
        r = http.post(f"{self.BASE}/batches", headers=self.headers, json={"name": name})
        r.raise_for_status()
        batch_id = r.json()["batch_id"]
        for start in range(0, len(requests), self.CHUNK):
            chunk = [{"batch_request_id": q["custom_id"],
                      "batch_request": {"chat_get_completion": _chat_body(q, cfg)}}
                     for q in requests[start:start + self.CHUNK]]
            r = http.post(f"{self.BASE}/batches/{batch_id}/requests", headers=self.headers, json={"batch_requests": chunk})
            r.raise_for_status()
        return batch_id

    def status(self, batch_id, cfg):
        r = http.get(f"{self.BASE}/batches/{batch_id}", headers=self.headers)
        r.raise_for_status()
        s = r.json()["state"]
        detail = f"{s['num_success']}/{s['num_requests']} succeeded, {s['num_error']} errors, {s['num_pending']} pending"
        return {"state": "done" if s["num_pending"] == 0 else "running", "detail": detail}

    def fetch(self, batch_id, cfg):
        out, token = {}, None
        while True:
            params = {"limit": 1000, **({"pagination_token": token} if token else {})}
            r = http.get(f"{self.BASE}/batches/{batch_id}/results", headers=self.headers, params=params)
            r.raise_for_status()
            page = r.json()
            for item in page.get("results", []):
                res = item.get("batch_result") or {}
                completion = (res.get("response") or {}).get("chat_get_completion")
                err = res.get("error") or res.get("error_message") or item.get("error_message")
                out[item["batch_request_id"]] = _parse_chat_completion(completion) if completion and not err else _result(error=str(err))
            token = page.get("pagination_token")
            if not token:
                return out


# ----------------------------------------------------------------------------- synchronous fallback

class SyncFallback:
    """For providers without a batch API (e.g. DeepSeek): normal parallel calls, same result format.
    Runs everything inside submit() and stores the results locally; batch_id is that file's path."""

    def __init__(self, workers=8):
        from llm_tools.other.llm_client import generate_text
        self.generate_text = generate_text
        self.workers = workers

    def _one(self, req, cfg):
        messages = [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["prompt"]}]
        try:
            text = self.generate_text(messages, provider=cfg["provider"], model=cfg["model"],
                                      temperature=cfg.get("temperature") or 0, max_tokens=cfg["max_tokens"])
            return req["custom_id"], _result(text, None, cfg["model"])
        except Exception as e:  # keep going; failed requests are retried later
            return req["custom_id"], _result(error=str(e))

    def submit(self, requests, cfg, name):
        with ThreadPoolExecutor(self.workers) as pool:
            out = dict(pool.map(lambda r: self._one(r, cfg), requests))
        path = ROOT / "results" / "batches" / "sync" / f"{name}_{int(time.time())}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        return str(path.relative_to(ROOT))

    def status(self, batch_id, cfg):
        return {"state": "done", "detail": "completed synchronously"}

    def fetch(self, batch_id, cfg):
        return json.loads((ROOT / batch_id).read_text(encoding="utf-8"))


ADAPTERS = {
    "openai": OpenAIBatch,
    "claude": AnthropicBatch,
    "gemini": GeminiBatch,
    "mistral": MistralBatch,
    "grok": XAIBatch,
    "deepseek": SyncFallback,
}


def get_adapter(cfg):
    if cfg.get("mode") == "sync":
        return SyncFallback()
    if cfg["provider"] == "claude":
        return AnthropicBatch(cfg.get("api_version"))
    return ADAPTERS[cfg["provider"]]()
