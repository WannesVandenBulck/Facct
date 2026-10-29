"""
Build the provider-independent narrative requests for the three conditions:

- include_pa:        the original prompt, protected attributes shown
- exclude_pa:        the original prompt with the protected-attribute lines removed
- counterfactual_pa: the original prompt with ONE protected attribute swapped to the other group

Every request is a plain dict (see make_request) identified by a unique custom_id, so the
same requests can be sent to any provider and the results matched back afterwards.
"""

import re
import statistics
from pathlib import Path

import pandas as pd

from llm_tools.prompts import prompt_credit, prompt_law, prompt_saudi, prompt_student

ROOT = Path(__file__).parent.parent.parent

PROMPT_MODULES = {
    "credit": prompt_credit,
    "law": prompt_law,
    "saudi": prompt_saudi,
    "student": prompt_student,
}

DATASETS = list(PROMPT_MODULES)
CONDITIONS = ["include_pa", "exclude_pa", "counterfactual_pa"]

SYSTEM_PROMPT = ("You are an expert at explaining machine learning predictions to non-technical users. "
                 "Write clear narratives that help people understand model decisions.")


def median_split_low(s):
    return s < s.median()


def median_split_high(s):
    return s >= s.median()


def median_of(group):
    """Swap target = median of the OTHER group (median_low, so it is always a value that occurs)."""
    def target(s):
        return statistics.median_low(sorted(s[group(s)].tolist()))
    return target


# Counterfactual swaps: each instance falls in exactly one "from" group per attribute and is swapped
# to the other group. Groups mirror COMPARISON_GROUPS in datasets_prep/scripts/predictions.py and
# assign_groups() in scripts/fairness_eval/RQ1_total_disparity.py.
# "to" is a fixed raw value, or a function of the attribute column (median of the other group).
COUNTERFACTUAL_SWAPS = {
    "credit": [
        dict(label="age_below_to_old", attr="age", frm=median_split_low, to=median_of(median_split_high)),
        dict(label="age_atabove_to_young", attr="age", frm=median_split_high, to=median_of(median_split_low)),
        dict(label="sex_male_to_female", attr="sex", frm=lambda s: s == 0, to=1),
        dict(label="sex_female_to_male", attr="sex", frm=lambda s: s == 1, to=0),
        dict(label="foreignworker_yes_to_no", attr="foreign_worker", frm=lambda s: s == 1, to=2),
        dict(label="foreignworker_no_to_yes", attr="foreign_worker", frm=lambda s: s == 2, to=1),
    ],
    "law": [
        dict(label="gender_female_to_male", attr="gender", frm=lambda s: s == 0, to=1),
        dict(label="gender_male_to_female", attr="gender", frm=lambda s: s == 1, to=0),
        # black (3) is the largest non-white group
        dict(label="race_white_to_black", attr="race", frm=lambda s: s == 0, to=3),
        dict(label="race_nonwhite_to_white", attr="race", frm=lambda s: s != 0, to=0),
    ],
    "saudi": [
        dict(label="gender_female_to_male", attr="Gender", frm=lambda s: s == 0, to=1),
        dict(label="gender_male_to_female", attr="Gender", frm=lambda s: s == 1, to=0),
        dict(label="age_young_to_old", attr="Age", frm=lambda s: s == 0, to=median_of(lambda s: s > 0)),
        dict(label="age_old_to_young", attr="Age", frm=lambda s: s > 0, to=0),
        dict(label="health_no_to_yes", attr="Health_Issues", frm=lambda s: s == 0, to=1),
        dict(label="health_yes_to_no", attr="Health_Issues", frm=lambda s: s == 1, to=0),
    ],
    "student": [
        dict(label="sex_female_to_male", attr="sex", frm=lambda s: s == 0, to=1),
        dict(label="sex_male_to_female", attr="sex", frm=lambda s: s == 1, to=0),
        dict(label="age_below_to_old", attr="age", frm=median_split_low, to=median_of(median_split_high)),
        dict(label="age_atabove_to_young", attr="age", frm=median_split_high, to=median_of(median_split_low)),
        dict(label="health_poor_to_verygood", attr="health", frm=lambda s: s <= 3, to=median_of(lambda s: s >= 4)),
        dict(label="health_good_to_bad", attr="health", frm=lambda s: s >= 4, to=median_of(lambda s: s <= 3)),
    ],
}


def load_adverse(dataset):
    return pd.read_csv(ROOT / "datasets_prep" / "data" / f"{dataset}_dataset" / f"{dataset}_adverse.csv")


def resolve_counterfactual_jobs(dataset):
    """Turn COUNTERFACTUAL_SWAPS into concrete jobs: instance ids per job and a single raw swap value."""
    adverse = load_adverse(dataset)
    jobs = []
    for swap in COUNTERFACTUAL_SWAPS[dataset]:
        col = adverse[swap["attr"]]
        mask = swap["frm"](col)
        to = swap["to"](col) if callable(swap["to"]) else swap["to"]
        to = int(to) if float(to).is_integer() else float(to)
        jobs.append(dict(label=swap["label"], attr=swap["attr"], override_value=to,
                         instances=adverse.loc[mask, "instance_index"].astype(int).tolist()))

    # every instance must be swapped exactly once per attribute
    for attr in {j["attr"] for j in jobs}:
        ids = [i for j in jobs if j["attr"] == attr for i in j["instances"]]
        assert sorted(ids) == sorted(adverse["instance_index"]), f"{dataset}/{attr}: groups do not partition the sample"
    return jobs


def apply_swap(prompt, module, attr, raw_value):
    """Replace only the value on the '- {attr} = ... (...)' line, formatted exactly like the original
    values (label mapping, lowercase, number formatting). Fails loudly unless exactly one line matches."""
    mapped = module.map_attribute_value(attr, raw_value)
    shown = module.format_number(attr, mapped) if isinstance(mapped, (int, float)) else mapped
    new_prompt, n = re.subn(rf"(?m)^(- {re.escape(attr)} = )(.+?)( \(.+?\))",
                            lambda m: f"{m.group(1)}{shown}{m.group(3)}", prompt, count=1)
    if n != 1:
        raise ValueError(f"apply_swap: expected exactly 1 line for '{attr}', found {n}")
    return new_prompt, shown


def make_request(dataset, condition, instance_idx, prompt, label=None, attr=None, override_value=None, target_value_label=None):
    parts = [dataset, condition] + ([label] if label else []) + [str(instance_idx)]
    custom_id = "__".join(parts)
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", custom_id), f"custom_id not valid for all providers: {custom_id}"
    return {
        "custom_id": custom_id,
        "dataset": dataset,
        "condition": condition,
        "label": label,
        "instance_idx": int(instance_idx),
        "attr_column": attr,
        "override_value": override_value,
        "target_value_label": target_value_label,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
    }


def check_threshold(dataset):
    """The threshold stated in the prompt must be the one predictions.py used: every adverse score at or
    above it, every favourable (positive) score below it."""
    t = PROMPT_MODULES[dataset].DECISION_THRESHOLD
    adverse = load_adverse(dataset)["prediction_score"]
    positive = pd.read_csv(ROOT / "datasets_prep" / "data" / f"{dataset}_dataset" / f"{dataset}_positive.csv")["prediction_score"]
    assert adverse.min() >= t and positive.max() < t, \
        f"{dataset}: DECISION_THRESHOLD {t} does not match the predictions (adverse min {adverse.min():.3f}, positive max {positive.max():.3f})"


def build_requests(dataset, condition):
    """All requests for one (dataset, condition) group."""
    module = PROMPT_MODULES[dataset]
    check_threshold(dataset)
    ids = load_adverse(dataset)["instance_index"].astype(int).tolist()

    if condition == "include_pa":
        return [make_request(dataset, condition, i, module.build_shap_prompt(i)) for i in ids]

    if condition == "exclude_pa":
        return [make_request(dataset, condition, i, module.build_shap_prompt(i, exclude_protected_attributes=True)) for i in ids]

    if condition == "counterfactual_pa":
        requests = []
        for job in resolve_counterfactual_jobs(dataset):
            for i in job["instances"]:
                prompt, shown = apply_swap(module.build_shap_prompt(i), module, job["attr"], job["override_value"])
                requests.append(make_request(dataset, condition, i, prompt, label=job["label"], attr=job["attr"],
                                             override_value=job["override_value"], target_value_label=shown))
        return requests

    raise ValueError(f"Unknown condition: {condition}")
