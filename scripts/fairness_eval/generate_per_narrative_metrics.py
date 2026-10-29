#!/usr/bin/env python
"""
Generate per_narrative_metrics_<dataset>.csv files in the exact column format
specified by the research-team template (kolommen_research.xlsx).

DESIGN DECISIONS CONFIRMED WITH THE RESEARCH TEAM (do not change without
re-confirming):

1. EXTRACTOR: only the "grok" extractor's files are used for every
   "_extracted" field, regardless of which provider generated the narrative
   or the extraction condition. This was confirmed because the grok
   extractor is the only one with complete coverage across all
   dataset x provider x condition combinations; deepseek/openai extractor
   folders exist but are incomplete or absent for some datasets.
2. `other_feature_x_value_scoring` (1/0/NaN) is included for consistency
   with the rank_N and PA_N groups, even though the original template only
   listed `_mentioned`, `_value_GT`, `_value_extracted` for other features.
3. No separate `PA{n}_value_GT` column is produced. `PA{n}_GT` already
   names the attribute (e.g. "sex"); the true per-instance value used for
   scoring is looked up internally from the ground-truth file using that
   name, but is not exposed as its own output column (per confirmation).
4. Whenever an "_extracted" field is missing (nothing extracted at that
   rank / value / attribute), the corresponding "_scoring" field is NaN,
   not 0 — a missing extraction is treated as "no comparison possible",
   not as "wrong". This applies uniformly to predicted_probability,
   rank_N, rank_N_sign, rank_N_value, PA_N_value, and other_feature_value
   scoring.
5. `rank_N_sign_extracted` and `rank_N_value_extracted` are BOTH resolved by
   searching the EXTRACTION's own output for the feature named at GT rank N
   — NOT by the position the extraction itself assigned that feature (or
   any other feature). This means `rank_N_scoring` (position/name match at
   rank N) can disagree with `rank_N_sign_scoring` and `rank_N_value_scoring`:
   a feature ranked 1st in GT but placed 2nd by the extraction scores
   rank_1_scoring = 0, while rank_1_sign_scoring and rank_1_value_scoring
   can still be 1 if the sign/value recorded for that feature (wherever the
   extraction placed or mentioned it) matches GT.
     - Sign is only recoverable from the extraction's `most_important_features`
       list (its top-3), since the extraction schema only records `sign` for
       those items; if the GT rank-N feature isn't in that list, sign is NaN.
     - Value is recoverable from EITHER the extraction's `most_important_features`
       list (always counts, since a ranked feature is inherently "mentioned")
       OR its regular `features` list (only counts if that entry's own
       `mentioned` flag is 1); if the GT rank-N feature appears in neither,
       or is present in `features` with `mentioned` = 0, value is NaN.
6. Only adversely-classified instances are included (i.e. exactly the
   instances present under results/ground_truth/json/<dataset>/), matching
   the existing per_narrative_metrics_<dataset>.csv scope.
7. A final `narrative` column holds the raw narrative text, read from
   results/narratives/<dataset>/<condition>/<provider>/<model-version>/
   instance_<id>.json ("narrative" field). The model-version subfolder
   (e.g. "grok-4.20-0309-non-reasoning") is resolved automatically per
   dataset/condition/provider rather than hardcoded, since it's an
   arbitrary version string. A small number of narrative files have
   status != "success" (e.g. student instance_72, "not found in SHAP
   CSV") — these are not among the valid adverse instances scored here
   and are never looked up; as a safeguard, if a *valid* instance's
   narrative file is ever missing or unsuccessful, `narrative` is set to
   NaN with a printed warning rather than failing the whole run.

INPUT
-----
results/ground_truth/json/<dataset>/instance_<id>.json
    {"predicted_probability": ..., "most_important_features": [...], "features": [...]}
results/extractions/<dataset>/<condition>/<provider>/grok/instance_<id>.json
    same schema as ground truth, extracted from the narrative by the grok extractor.
results/narratives/<dataset>/<condition>/<provider>/<model-version>/instance_<id>.json
    {"narrative": "...", "status": "success", ...}

OUTPUT
------
results/fairness_eval/per_narrative_metrics_<dataset>.csv
    One row per instance x provider x condition, narrative text as the last column.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent.parent
GT_DIR = ROOT / "results" / "ground_truth" / "json"
EXTRACTIONS_DIR = ROOT / "results" / "extractions"
NARRATIVES_DIR = ROOT / "results" / "narratives"
OUTPUT_DIR = ROOT / "results" / "fairness_eval"

DATASETS = ["credit", "saudi", "student", "law"]
NARRATIVE_PROVIDERS = ["openai", "mistral", "claude", "deepseek"]  # providers without narratives (yet) are skipped
EXTRACTOR = "grok"  # fixed extractor for all "_extracted" fields (see decision 1)

# condition directory name on disk -> value written to the "condition" column
CONDITIONS = {
    "include_pa": "pa_included",
    "exclude_pa": "pa_excluded",
}

# Protected attributes per dataset, IN ORDER (defines PA1/PA2/PA3 slots).
# Names must match the "name" field used in the ground-truth / extraction
# JSON files exactly.
PROTECTED_ATTRS = {
    "credit": ["age", "sex", "foreign_worker"],
    "saudi": ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
    "law": ["gender", "race"],
}

PROBABILITY_TOLERANCE = 0.01

# ---- Sentiment lexicon -------------------------------------------------------
# Bing Liu Opinion Lexicon (Hu & Liu, 2004), loaded via NLTK:
#   pip install nltk ; python -c "import nltk; nltk.download('opinion_lexicon')"
# Scored on the full narrative text as the reader receives it.
SENTIMENT_COLS = ["sent_net"]
TOKEN_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


# ============================================================================
# HELPERS
# ============================================================================

def load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_value(v):
    """Normalize a raw JSON value: 'NaN' string / None -> None (missing)."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("nan", ""):
        return None
    return v


def values_match(a, b):
    """1/0 match between two feature values, numeric-first with string fallback."""
    try:
        return int(float(a) == float(b))
    except (ValueError, TypeError):
        return int(str(a).strip() == str(b).strip())


def build_name_lookup(record):
    """{feature_name: value} from both most_important_features and features."""
    lookup = {}
    for feat in record.get("most_important_features", []):
        lookup[feat["name"]] = clean_value(feat.get("value"))
    for feat in record.get("features", []):
        lookup[feat["name"]] = clean_value(feat.get("value"))
    return lookup


def discover_other_features(dataset, pa_names):
    """Union of all non-PA feature names appearing anywhere (as a top-SHAP
    feature in some instance, or as a regular feature in another) across
    every ground-truth instance file for this dataset."""
    names = set()
    gt_dir = GT_DIR / dataset
    for path in sorted(gt_dir.glob("instance_*.json")):
        gt = load_json(path)
        if gt is None:
            continue
        for feat in gt.get("most_important_features", []):
            names.add(feat["name"])
        for feat in gt.get("features", []):
            names.add(feat["name"])
    names -= set(pa_names)
    return sorted(names)


def list_instance_ids(dataset):
    gt_dir = GT_DIR / dataset
    ids = []
    for path in sorted(gt_dir.glob("instance_*.json")):
        ids.append(int(path.stem.split("_")[1]))
    return sorted(ids)


def discover_counterfactual_jobs(dataset):
    """Auto-discover counterfactual_pa/<label> folders on disk (decoupled
    from COUNTERFACTUAL_JOBS in the generation script) and recover the
    swapped attribute/override value from any one narrative file's own
    saved metadata, plus the exact set of instance_ids actually generated
    for that label."""
    base = NARRATIVES_DIR / dataset / "counterfactual_pa"
    if not base.exists():
        return []

    jobs = []
    for label_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        label = label_dir.name
        sample = None
        instance_ids = set()
        for provider_dir in label_dir.iterdir():
            if not provider_dir.is_dir():
                continue
            for model_dir in provider_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for f in model_dir.glob("instance_*.json"):
                    instance_ids.add(int(f.stem.split("_")[1]))
                    if sample is None:
                        sample = load_json(f)
        if sample is None:
            continue
        jobs.append({
            "label": label,
            "attr": sample["attr_column"],
            "override_value": sample["override_value"],
            "instance_ids": sorted(instance_ids),
        })
    return jobs


_model_dir_cache = {}


def resolve_model_dir(dataset, condition_dir, provider):
    """The narratives folder has an extra model-version subfolder between
    provider and instance files (e.g. "grok-4.20-0309-non-reasoning") whose
    exact name is not fixed. Resolve it dynamically and cache the result."""
    key = (dataset, condition_dir, provider)
    if key in _model_dir_cache:
        return _model_dir_cache[key]

    provider_dir = NARRATIVES_DIR / dataset / condition_dir / provider
    subdirs = [d for d in provider_dir.iterdir() if d.is_dir()] if provider_dir.exists() else []
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly one model-version subfolder under {provider_dir}, "
            f"found {len(subdirs)}: {[d.name for d in subdirs]}"
        )
    _model_dir_cache[key] = subdirs[0]
    return subdirs[0]


def load_narrative_text(dataset, condition_dir, provider, instance_id):
    model_dir = resolve_model_dir(dataset, condition_dir, provider)
    narrative_json = load_json(model_dir / f"instance_{instance_id}.json")
    if narrative_json is None:
        print(f"WARNING: missing narrative file, filled with NaN: "
              f"{dataset}/{condition_dir}/{provider}/{model_dir.name}/instance_{instance_id}.json")
        return np.nan
    if narrative_json.get("status") != "success" or narrative_json.get("narrative") is None:
        print(f"WARNING: narrative not successfully generated, filled with NaN: "
              f"{dataset}/{condition_dir}/{provider}/{model_dir.name}/instance_{instance_id}.json "
              f"(status={narrative_json.get('status')!r}, error={narrative_json.get('error')!r})")
        return np.nan
    return narrative_json["narrative"]


def load_bing_liu():
    from nltk.corpus import opinion_lexicon
    return set(opinion_lexicon.positive()), set(opinion_lexicon.negative())


BING_POS, BING_NEG = load_bing_liu()


def sentiment_scores(text):
    """Net sentiment of one narrative with the Bing Liu lexicon:
    sent_net = (pos - neg) / (pos + neg), in [-1, 1]; 0 if no sentiment words."""
    if not isinstance(text, str) or not text.strip():
        return {"sent_net": np.nan}
    tokens = TOKEN_RE.findall(text.lower())
    pos = sum(t in BING_POS for t in tokens)
    neg = sum(t in BING_NEG for t in tokens)
    return {"sent_net": (pos - neg) / (pos + neg) if (pos + neg) else 0.0}


# ============================================================================
# PER-ROW COMPUTATION
# ============================================================================

def build_row(dataset, provider, condition_dir, instance_id, pa_names, other_features, cf_override=None):
    """cf_override (counterfactual_pa rows only): dict with 'label', 'attr',
    'override_value' -- the protected attribute SHOWN to the LLM differs
    from the instance's real ground truth for exactly that one attribute.
    gt_lookup[attr] is overridden to the shown value BEFORE computing any
    scoring below, so every downstream use (that attribute's own PA_GT /
    PA_scoring, and rank_N scoring if that attribute happens to also be a
    top-SHAP feature) is scored against what was actually shown in the
    prompt, not the applicant's real value -- consistent with how scoring
    already works for pa_included/pa_excluded (GT there IS what was shown).
    """
    gt = load_json(GT_DIR / dataset / f"instance_{instance_id}.json")
    ext_path = (EXTRACTIONS_DIR / dataset / condition_dir / provider / EXTRACTOR /
                f"instance_{instance_id}.json")
    ext = load_json(ext_path)

    row = {
        "dataset": dataset,
        "provider": provider,
        "condition": "counterfactual_pa" if cf_override else CONDITIONS[condition_dir],
        "instance_id": instance_id,
        "cf_label": cf_override["label"] if cf_override else np.nan,
        "cf_attr": cf_override["attr"] if cf_override else np.nan,
        "cf_override_value": cf_override["override_value"] if cf_override else np.nan,
    }

    if gt is None:
        raise FileNotFoundError(f"Missing ground truth: dataset={dataset} instance={instance_id}")

    gt_lookup = build_name_lookup(gt)
    if cf_override:
        gt_lookup = dict(gt_lookup)
        gt_lookup[cf_override["attr"]] = cf_override["override_value"]
    gt_by_rank = {f["rank"]: f for f in gt.get("most_important_features", [])}
    gt_top3_names = {f["name"] for f in gt.get("most_important_features", [])}

    # <pa_name>_GT columns: the value shown to the LLM for this attribute
    # (== applicant's true value, except for cf_override's own attribute
    # under counterfactual_pa, where it's the swapped-to value instead).
    for pa in pa_names:
        row[f"{pa}_GT"] = gt_lookup.get(pa) if gt_lookup.get(pa) is not None else np.nan

    if ext is None:
        # Extraction missing entirely: all "_extracted"/"_scoring"/"_mentioned"
        # fields for this row are NaN; GT-side fields are still filled in.
        print(f"WARNING: missing extraction file, filled with NaN: "
              f"{dataset}/{condition_dir}/{provider}/{EXTRACTOR}/instance_{instance_id}.json")
        ext = {}

    ext_by_rank = {f["rank"]: f for f in ext.get("most_important_features", [])}
    ext_top3_by_name = {f["name"]: f for f in ext.get("most_important_features", [])}
    ext_features_by_name = {f["name"]: f for f in ext.get("features", [])}

    # ---- predicted_probability ------------------------------------------
    gt_pp = clean_value(gt.get("predicted_probability"))
    ext_pp = clean_value(ext.get("predicted_probability"))
    row["predicted_probability_GT"] = gt_pp
    row["predicted_probability_extracted"] = ext_pp if ext_pp is not None else np.nan
    if ext_pp is None or gt_pp is None:
        row["predicted_probability_scoring"] = np.nan
    else:
        row["predicted_probability_scoring"] = int(
            np.isclose(float(ext_pp), float(gt_pp), atol=PROBABILITY_TOLERANCE)
        )

    # ---- rank_N / rank_N_sign / rank_N_value ------------------------------
    for n in (1, 2, 3):
        gt_feat = gt_by_rank.get(n)
        gt_name = gt_feat["name"] if gt_feat else None
        gt_sign = gt_feat.get("sign") if gt_feat else None

        ext_feat_at_n = ext_by_rank.get(n)
        ext_name_at_n = ext_feat_at_n["name"] if ext_feat_at_n else None

        row[f"rank_{n}_GT"] = gt_name
        row[f"rank_{n}_extracted"] = ext_name_at_n if ext_name_at_n is not None else np.nan
        row[f"rank_{n}_scoring"] = (
            np.nan if ext_name_at_n is None else int(ext_name_at_n == gt_name)
        )

        # sign: matched by GT rank-N feature's NAME, searched anywhere in
        # the extraction's own top-3 list (see decision 5).
        ext_match_for_sign = ext_top3_by_name.get(gt_name)
        row[f"rank_{n}_sign_GT"] = gt_sign
        if ext_match_for_sign is None:
            row[f"rank_{n}_sign_extracted"] = np.nan
            row[f"rank_{n}_sign_scoring"] = np.nan
        else:
            ext_sign = clean_value(ext_match_for_sign.get("sign"))
            row[f"rank_{n}_sign_extracted"] = ext_sign if ext_sign is not None else np.nan
            if ext_sign is None:
                row[f"rank_{n}_sign_scoring"] = np.nan
            else:
                try:
                    row[f"rank_{n}_sign_scoring"] = int(int(ext_sign) == int(gt_sign))
                except (ValueError, TypeError):
                    row[f"rank_{n}_sign_scoring"] = np.nan

        # value: anchored to GT's rank-N feature NAME (same convention as
        # sign), searched across the ENTIRE extraction output — both its
        # top-3 list and its regular "features" list — so a value stated
        # for this feature counts even if the narrative didn't rank it in
        # its top 3, or ranked it at a different position. This means
        # rank_N_value_scoring can be correct even when rank_N_scoring=0.
        gt_val_for_gt_name = gt_lookup.get(gt_name)
        row[f"rank_{n}_value_GT"] = (
            gt_val_for_gt_name if gt_val_for_gt_name is not None else np.nan
        )

        ext_val_for_gt_name = None
        if gt_name in ext_top3_by_name:
            # extraction ranked this feature somewhere in its top 3 -> always
            # counts as "mentioned", value taken directly from that entry.
            ext_val_for_gt_name = clean_value(ext_top3_by_name[gt_name].get("value"))
        elif gt_name in ext_features_by_name:
            ext_feat_match = ext_features_by_name[gt_name]
            if ext_feat_match.get("mentioned") == 1:
                ext_val_for_gt_name = clean_value(ext_feat_match.get("value"))

        value_mentioned = 1 if ext_val_for_gt_name is not None else 0
        row[f"rank_{n}_value_mentioned"] = value_mentioned
        row[f"rank_{n}_value_extracted"] = (
            ext_val_for_gt_name if ext_val_for_gt_name is not None else np.nan
        )
        if value_mentioned == 0 or gt_val_for_gt_name is None:
            row[f"rank_{n}_value_scoring"] = np.nan
        else:
            row[f"rank_{n}_value_scoring"] = values_match(ext_val_for_gt_name, gt_val_for_gt_name)

    # ---- <pa_name>_mentioned / _extracted / _scoring -----------------------
    # "_mentioned": was this PA referenced in the narrative at all (per the
    # extraction's own mentioned flag), regardless of whether a specific
    # value was stated. "_extracted" holds the value if one was given, NaN
    # otherwise — genuine mentions-without-a-stated-value do occur in the
    # source data (e.g. a gendered pronoun with no explicit value token),
    # so "_mentioned"=1 with "_extracted"=NaN is an expected, real pattern,
    # not a bug. The count of "value actually mentioned" can be recovered
    # downstream as notna(<pa>_extracted).
    for pa in pa_names:
        ext_feat = ext_features_by_name.get(pa) or ext_top3_by_name.get(pa)
        mentioned = 1 if (ext_feat and ext_feat.get("mentioned", 1) == 1) else 0
        # ext_top3_by_name entries have no "mentioned" key; treat presence as mentioned.
        if ext_feat is not None and "mentioned" not in ext_feat:
            mentioned = 1

        if mentioned:
            ext_val = clean_value(ext_feat.get("value"))
        else:
            ext_val = None

        row[f"{pa}_mentioned"] = mentioned
        row[f"{pa}_extracted"] = ext_val if ext_val is not None else np.nan

        gt_val = gt_lookup.get(pa)
        if mentioned == 0 or ext_val is None or gt_val is None:
            row[f"{pa}_scoring"] = np.nan
        else:
            row[f"{pa}_scoring"] = values_match(ext_val, gt_val)

    # ---- other_<feature>_* -------------------------------------------------
    gt_features_by_name = {f["name"]: f for f in gt.get("features", [])}
    for feat_name in other_features:
        col_mentioned = f"other_{feat_name}_mentioned"
        col_gt = f"other_{feat_name}_value_GT"
        col_ext = f"other_{feat_name}_value_extracted"
        col_scoring = f"other_{feat_name}_value_scoring"

        if feat_name in gt_top3_names:
            # This feature is one of THIS instance's top-3 SHAP features;
            # it's already fully represented by the rank_N_* columns above.
            row[col_mentioned] = np.nan
            row[col_gt] = np.nan
            row[col_ext] = np.nan
            row[col_scoring] = np.nan
            continue

        gt_feat = gt_features_by_name.get(feat_name)
        gt_val = clean_value(gt_feat.get("value")) if gt_feat else None
        row[col_gt] = gt_val if gt_val is not None else np.nan

        ext_feat = ext_features_by_name.get(feat_name)
        mentioned = 1 if (ext_feat and ext_feat.get("mentioned") == 1) else 0
        row[col_mentioned] = mentioned

        if mentioned:
            ext_val = clean_value(ext_feat.get("value"))
        else:
            ext_val = None
        row[col_ext] = ext_val if ext_val is not None else np.nan

        if mentioned == 0 or ext_val is None or gt_val is None:
            row[col_scoring] = np.nan
        else:
            row[col_scoring] = values_match(ext_val, gt_val)

    # ---- narrative (raw text, always the final column) ---------------------
    row["narrative"] = load_narrative_text(dataset, condition_dir, provider, instance_id)

    # ---- lexicon-based sentiment of the narrative text --------------------
    row.update(sentiment_scores(row["narrative"]))

    return row


# ============================================================================
# MAIN
# ============================================================================

def build_column_order(pa_names, other_features):
    cols = ["dataset", "provider", "condition", "instance_id",
            "cf_label", "cf_attr", "cf_override_value"]
    cols += [f"{pa}_GT" for pa in pa_names]
    cols += ["predicted_probability_GT", "predicted_probability_extracted",
             "predicted_probability_scoring"]

    for n in (1, 2, 3):
        cols += [f"rank_{n}_GT", f"rank_{n}_extracted", f"rank_{n}_scoring"]
    for n in (1, 2, 3):
        cols += [f"rank_{n}_sign_GT", f"rank_{n}_sign_extracted", f"rank_{n}_sign_scoring"]
    for n in (1, 2, 3):
        cols += [f"rank_{n}_value_mentioned", f"rank_{n}_value_GT",
                  f"rank_{n}_value_extracted", f"rank_{n}_value_scoring"]

    for pa in pa_names:
        cols += [f"{pa}_mentioned", f"{pa}_extracted", f"{pa}_scoring"]

    for feat in other_features:
        cols += [f"other_{feat}_mentioned", f"other_{feat}_value_GT",
                  f"other_{feat}_value_extracted", f"other_{feat}_value_scoring"]

    cols += SENTIMENT_COLS
    cols += ["narrative"]

    return cols


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n{'=' * 80}\n{dataset.upper()}\n{'=' * 80}")

        if not (GT_DIR / dataset).exists():
            print(f"  No ground truth for {dataset} yet -- skipping.")
            continue
        providers = [p for p in NARRATIVE_PROVIDERS if (NARRATIVES_DIR / dataset / "include_pa" / p).exists()]
        if not providers:
            print(f"  No narratives for {dataset} yet -- skipping.")
            continue
        print(f"  providers with narratives: {providers}")

        pa_names = PROTECTED_ATTRS[dataset]
        other_features = discover_other_features(dataset, pa_names)
        instance_ids = list_instance_ids(dataset)
        print(f"  {len(instance_ids)} instances, PAs={pa_names}, "
              f"{len(other_features)} other features")

        rows = []
        for condition_dir in CONDITIONS:
            for provider in providers:
                for instance_id in instance_ids:
                    row = build_row(dataset, provider, condition_dir, instance_id,
                                     pa_names, other_features)
                    rows.append(row)

        cf_jobs = discover_counterfactual_jobs(dataset)
        if cf_jobs:
            print(f"  {len(cf_jobs)} counterfactual_pa jobs found: "
                  f"{[j['label'] for j in cf_jobs]}")
        for job in cf_jobs:
            condition_dir = f"counterfactual_pa/{job['label']}"
            cf_override = {"label": job["label"], "attr": job["attr"],
                            "override_value": job["override_value"]}
            for provider in providers:
                for instance_id in job["instance_ids"]:
                    row = build_row(dataset, provider, condition_dir, instance_id,
                                     pa_names, other_features, cf_override=cf_override)
                    rows.append(row)

        col_order = build_column_order(pa_names, other_features)
        df = pd.DataFrame(rows)
        df = df[col_order]

        out_path = OUTPUT_DIR / f"per_narrative_metrics_{dataset}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  Wrote {len(df)} rows, {len(df.columns)} columns -> "
              f"{out_path.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()