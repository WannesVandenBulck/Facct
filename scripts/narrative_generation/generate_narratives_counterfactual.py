#!/usr/bin/env python
"""
Generate LLM narratives for counterfactual (PA-swapped) SHAP explanations.

This is a SEPARATE script from generate_narratives.py -- it does not modify
or overwrite it, and it does not modify any of the llm_tools/prompts/*.py
source files either.

WHY THIS SCRIPT DOES ITS OWN PROMPT SUBSTITUTION (IMPORTANT)
-----------------------------------------------------------------
build_shap_prompt()'s built-in *_override parameters have a bug: when an
override is applied, the substituted value is inserted as a raw string
(str(override_value)) instead of being passed through the same
map_attribute_value() lookup used for every other value in the prompt.
For any attribute with a categorical label mapping (sex, foreign_worker,
Gender, Age, Health_Issues, gender, race, health -- i.e. every protected
attribute except the two raw-numeric ages, credit/age and student/age,
which have no mapping to begin with) this means the LLM would be shown a
bare, unlabeled number (e.g. "sex = 1") instead of the correct label
(e.g. "sex = female"), while every other line in the prompt is unaffected.
Verified directly against the actual prompt-building code: this bug would
have affected 9 of 11 protected attributes (18 of 22 jobs, ~88% of all
planned counterfactual narratives).

Rather than edit the shared llm_tools/prompts/*.py files (which are also
used by generate_narratives.py, and by anything else that calls
build_shap_prompt), this script builds the ORIGINAL prompt with NO
override applied at all, then performs its own correct substitution using
each module's own (correctly implemented) map_attribute_value() function,
via a regex that only ever touches the single "- {attr} = ... (...)"
line for the swapped attribute -- verified, for every dataset and
attribute type, to change exactly that one line and nothing else.

Edit PROVIDERS_TO_RUN / COUNTERFACTUAL_JOBS below and run:
    python scripts/generate_narratives_counterfactual.py
"""

import os
import sys
import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_tools.prompts import prompt_credit, prompt_law, prompt_saudi, prompt_student
from llm_tools.other.llm_client import generate_text

DATASETS = {
    "credit": prompt_credit,
    "law": prompt_law,
    "saudi": prompt_saudi,
    "student": prompt_student,
}


# ===== BATCH CONFIGURATION (edit these values and run) =====

PROVIDERS_TO_RUN = [
    ("openai", "gpt-4o-mini"),
    ("grok", "grok-4.20-0309-non-reasoning"),
    ("deepseek", "deepseek-v4-flash"),
]

# Each job = one (dataset, attribute, swap-direction) counterfactual.
# "instances" = exact instance_ids belonging to the "from" group for this
# swap (pulled directly from true group membership -- independently
# verified against raw ground_truth/json files, see conversation history).
# "attr_column" = the exact attribute name as it appears in the prompt's
# "- {attr_column} = value (description)" line (note: casing differs from
# the override kwarg for some datasets, e.g. saudi uses "Gender"/"Age"/
# "Health_Issues", not "gender"/"age"/"health").
# "override_value" = the raw (unmapped) value to swap to; this script maps
# it to the correct label itself via apply_correct_swap() below.
COUNTERFACTUAL_JOBS = [
    {
        "dataset": "credit",
        "label": "age_below_to_old",
        "instances": [0, 1, 3, 5, 7, 10, 12, 16, 18, 20, 22, 24, 25, 29, 30, 33, 36, 37, 38, 40, 44, 45, 49, 51, 52, 54, 55, 61, 62, 64, 66, 67, 68, 69, 71, 72, 73, 78, 79, 84, 85, 86, 92, 93, 95],
        "attr_column": "age",
        "override_kwarg": "age_override",
        "override_value": 40,
    },
    {
        "dataset": "credit",
        "label": "age_atabove_to_young",
        "instances": [2, 4, 6, 8, 9, 11, 13, 14, 15, 17, 19, 21, 23, 26, 27, 28, 31, 32, 34, 35, 39, 41, 42, 43, 46, 47, 48, 50, 53, 56, 57, 58, 59, 60, 63, 65, 70, 74, 75, 76, 77, 80, 81, 82, 83, 87, 88, 89, 90, 91, 94, 96],
        "attr_column": "age",
        "override_kwarg": "age_override",
        "override_value": 26,
    },
    {
        "dataset": "credit",
        "label": "sex_male_to_female",
        "instances": [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 32, 33, 36, 39, 41, 42, 43, 46, 48, 50, 51, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 75, 76, 77, 79, 80, 81, 82, 86, 87, 88, 90, 93, 94, 95, 96],
        "attr_column": "sex",
        "override_kwarg": "sex_override",
        "override_value": 1,
    },
    {
        "dataset": "credit",
        "label": "sex_female_to_male",
        "instances": [0, 7, 12, 26, 27, 28, 30, 31, 34, 35, 37, 38, 40, 44, 45, 47, 49, 52, 54, 55, 70, 74, 78, 83, 84, 85, 89, 91, 92],
        "attr_column": "sex",
        "override_kwarg": "sex_override",
        "override_value": 0,
    },
    {
        "dataset": "credit",
        "label": "foreignworker_yes_to_no",
        "instances": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
        "attr_column": "foreign_worker",
        "override_kwarg": "foreign_worker_override",
        "override_value": 2,
    },
    {
        "dataset": "credit",
        "label": "foreignworker_no_to_yes",
        "instances": [30, 60],
        "attr_column": "foreign_worker",
        "override_kwarg": "foreign_worker_override",
        "override_value": 1,
    },
    {
        "dataset": "saudi",
        "label": "gender_female_to_male",
        "instances": [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 15, 16, 17, 19, 21, 26, 27, 31, 32, 33, 35, 37, 38, 39, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 59, 60, 61, 63, 65, 66, 67, 72, 74, 77, 82, 84, 85, 86, 88, 89, 92, 93, 94, 95, 96, 102, 104, 105],
        "attr_column": "Gender",
        "override_kwarg": "gender_override",
        "override_value": 1,
    },
    {
        "dataset": "saudi",
        "label": "gender_male_to_female",
        "instances": [5, 7, 9, 11, 13, 18, 20, 22, 23, 24, 25, 28, 29, 30, 34, 36, 40, 42, 43, 44, 56, 58, 62, 64, 68, 69, 70, 71, 73, 75, 76, 78, 79, 80, 81, 83, 87, 90, 91, 97, 98, 99, 100, 101, 103],
        "attr_column": "Gender",
        "override_kwarg": "gender_override",
        "override_value": 0,
    },
    {
        "dataset": "saudi",
        "label": "age_young_to_old",
        "instances": [0, 1, 4, 6, 8, 9, 10, 11, 12, 20, 21, 24, 26, 27, 31, 32, 33, 35, 37, 41, 45, 49, 50, 52, 53, 54, 56, 57, 59, 61, 62, 63, 64, 65, 66, 69, 74, 76, 82, 84, 85, 87, 88, 89, 94, 95, 102, 104],
        "attr_column": "Age",
        "override_kwarg": "age_override",
        "override_value": 1,
    },
    {
        "dataset": "saudi",
        "label": "age_old_to_young",
        "instances": [2, 3, 5, 7, 13, 14, 15, 16, 17, 18, 19, 22, 23, 25, 28, 29, 30, 34, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48, 51, 55, 58, 60, 67, 68, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 83, 86, 90, 91, 92, 93, 96, 97, 98, 99, 100, 101, 103, 105],
        "attr_column": "Age",
        "override_kwarg": "age_override",
        "override_value": 0,
    },
    {
        "dataset": "saudi",
        "label": "health_no_to_yes",
        "instances": [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 96, 98, 99, 100, 101, 102, 103, 104, 105],
        "attr_column": "Health_Issues",
        "override_kwarg": "health_override",
        "override_value": 1,
    },
    {
        "dataset": "saudi",
        "label": "health_yes_to_no",
        "instances": [1, 2, 5, 34, 38, 53, 61, 68, 73, 78, 88, 95, 97],
        "attr_column": "Health_Issues",
        "override_kwarg": "health_override",
        "override_value": 0,
    },
    {
        "dataset": "student",
        "label": "sex_female_to_male",
        "instances": [0, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 23, 25, 26, 27, 28, 29, 32, 33, 34, 36, 39, 41, 43, 45, 46, 49, 50, 51, 53, 55, 57, 59, 60, 61, 62, 64, 66, 67, 68],
        "attr_column": "sex",
        "override_kwarg": "sex_override",
        "override_value": 1,
    },
    {
        "dataset": "student",
        "label": "sex_male_to_female",
        "instances": [1, 2, 3, 8, 13, 19, 20, 21, 22, 24, 30, 31, 35, 37, 38, 40, 42, 44, 47, 48, 52, 54, 56, 58, 63, 65, 69, 70, 71],
        "attr_column": "sex",
        "override_kwarg": "sex_override",
        "override_value": 0,
    },
    {
        "dataset": "student",
        "label": "age_below_to_old",
        "instances": [2, 4, 6, 9, 14, 15, 16, 19, 20, 22, 25, 26, 29, 32, 38, 41, 42, 43, 45, 48, 51, 52, 54, 55, 56, 58, 62, 65, 68],
        "attr_column": "age",
        "override_kwarg": "age_override",
        "override_value": 18,
    },
    {
        "dataset": "student",
        "label": "age_atabove_to_young",
        "instances": [0, 1, 3, 5, 7, 8, 10, 11, 12, 13, 17, 18, 21, 23, 24, 27, 28, 30, 31, 33, 34, 35, 36, 37, 39, 40, 44, 46, 47, 49, 50, 53, 57, 59, 60, 61, 63, 64, 66, 67, 69, 70, 71],
        "attr_column": "age",
        "override_kwarg": "age_override",
        "override_value": 16,
    },
    {
        "dataset": "student",
        "label": "health_poor_to_verygood",
        "instances": [0, 4, 10, 12, 15, 16, 20, 24, 25, 26, 30, 32, 37, 38, 40, 42, 43, 44, 45, 49, 50, 51, 57, 58, 59, 60, 61, 64, 65, 67, 68, 69],
        "attr_column": "health",
        "override_kwarg": "health_override",
        "override_value": 5,
    },
    {
        "dataset": "student",
        "label": "health_good_to_verybad",
        "instances": [1, 2, 3, 5, 6, 7, 8, 9, 11, 13, 14, 17, 18, 19, 21, 22, 23, 27, 28, 29, 31, 33, 34, 35, 36, 39, 41, 46, 47, 48, 52, 53, 54, 55, 56, 62, 63, 66, 70, 71],
        "attr_column": "health",
        "override_kwarg": "health_override",
        "override_value": 1,
    },
    {
        "dataset": "law",
        "label": "gender_female_to_male",
        "instances": [0, 1, 3, 4, 6, 7, 11, 16, 18, 19, 25, 27, 28, 29, 32, 33, 34, 35, 36, 37, 42, 43, 47, 51, 52, 54, 56, 58, 60, 63, 64, 65, 73, 74, 79, 81, 83, 84, 85, 87, 92, 95, 97, 98, 99, 100, 101, 105, 106, 107, 109, 112, 115, 120, 121, 122, 123, 125, 126, 128, 130, 133, 135, 140, 141, 142, 143, 144, 145, 148, 149, 151, 153, 154, 156, 158, 159, 160, 162, 163, 164, 165, 166, 167, 172, 174, 175, 178, 183, 185, 186, 187, 191, 193, 195, 196, 199, 200, 203, 204, 205, 206, 207, 212, 213, 216, 218, 222, 224, 225, 227, 229, 234, 235, 238, 239, 240, 245, 253, 257, 259, 260, 263, 265, 266, 270, 275, 276, 279, 280, 281, 282, 284, 286, 287, 288, 289, 290, 291, 292, 294, 297, 299, 305],
        "attr_column": "gender",
        "override_kwarg": "gender_override",
        "override_value": 1,
    },
    {
        "dataset": "law",
        "label": "gender_male_to_female",
        "instances": [2, 5, 8, 9, 10, 12, 13, 14, 15, 17, 20, 21, 22, 23, 24, 26, 30, 31, 38, 39, 40, 41, 44, 45, 46, 48, 49, 50, 53, 55, 57, 59, 61, 62, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 78, 80, 82, 86, 88, 89, 90, 91, 93, 94, 96, 102, 103, 104, 108, 110, 111, 113, 114, 116, 117, 118, 119, 124, 127, 129, 131, 132, 134, 136, 137, 138, 139, 146, 147, 150, 152, 155, 157, 161, 168, 169, 170, 171, 173, 176, 177, 179, 180, 181, 182, 184, 188, 189, 190, 192, 194, 197, 198, 201, 202, 208, 209, 210, 211, 214, 215, 217, 219, 220, 221, 223, 226, 228, 230, 231, 232, 233, 236, 237, 241, 242, 243, 244, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 258, 261, 262, 264, 267, 268, 269, 271, 272, 273, 274, 277, 278, 283, 285, 293, 295, 296, 298, 300, 301, 302, 303, 304, 306, 307],
        "attr_column": "gender",
        "override_kwarg": "gender_override",
        "override_value": 0,
    },
    {
        "dataset": "law",
        "label": "race_white_to_asian",
        "instances": [0, 1, 3, 5, 7, 8, 9, 10, 14, 15, 18, 20, 21, 22, 24, 25, 26, 31, 36, 39, 40, 43, 47, 48, 49, 50, 52, 53, 54, 55, 56, 59, 60, 61, 62, 63, 66, 69, 73, 75, 77, 78, 80, 88, 90, 91, 93, 94, 97, 102, 103, 104, 106, 107, 108, 109, 110, 111, 114, 115, 120, 121, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 145, 146, 147, 150, 151, 153, 154, 157, 159, 162, 164, 167, 168, 170, 173, 174, 175, 176, 178, 179, 180, 182, 184, 187, 188, 192, 194, 197, 198, 200, 202, 203, 206, 207, 208, 210, 212, 214, 215, 216, 218, 220, 223, 227, 230, 231, 233, 235, 236, 237, 239, 240, 243, 244, 247, 249, 250, 251, 253, 255, 259, 261, 266, 267, 269, 270, 271, 272, 273, 277, 282, 283, 284, 285, 288, 289, 290, 293, 295, 296, 298, 300, 303, 304, 307],
        "attr_column": "race",
        "override_kwarg": "race_override",
        "override_value": 2,
    },
    {
        "dataset": "law",
        "label": "race_nonwhite_to_white",
        "instances": [2, 4, 6, 11, 12, 13, 16, 17, 19, 23, 27, 28, 29, 30, 32, 33, 34, 35, 37, 38, 41, 42, 44, 45, 46, 51, 57, 58, 64, 65, 67, 68, 70, 71, 72, 74, 76, 79, 81, 82, 83, 84, 85, 86, 87, 89, 92, 95, 96, 98, 99, 100, 101, 105, 112, 113, 116, 117, 118, 119, 122, 123, 124, 125, 129, 138, 141, 142, 143, 144, 148, 149, 152, 155, 156, 158, 160, 161, 163, 165, 166, 169, 171, 172, 177, 181, 183, 185, 186, 189, 190, 191, 193, 195, 196, 199, 201, 204, 205, 209, 211, 213, 217, 219, 221, 222, 224, 225, 226, 228, 229, 232, 234, 238, 241, 242, 245, 246, 248, 252, 254, 256, 257, 258, 260, 262, 263, 264, 265, 268, 274, 275, 276, 278, 279, 280, 281, 286, 287, 291, 292, 294, 297, 299, 301, 302, 305, 306],
        "attr_column": "race",
        "override_kwarg": "race_override",
        "override_value": 0,
    },
]
# ====== END BATCH CONFIGURATION ======


def apply_correct_swap(original_prompt, attr_column, new_raw_value, prompt_module):
    """
    Replace only the VALUE on the "- {attr_column} = ... (...)" line,
    using the prompt module's own map_attribute_value() (which IS correct
    -- only build_shap_prompt()'s internal override substitution is buggy).
    The description parenthetical, and every other line in the prompt, is
    left completely untouched. Raises if it doesn't find exactly one match,
    so a typo'd attr_column fails loudly instead of silently doing nothing.
    """
    new_label = prompt_module.map_attribute_value(attr_column, new_raw_value)
    pattern = rf"(- {re.escape(attr_column)} = )(.+?)( \(.+?\))"
    corrected, n_subs = re.subn(
        pattern, lambda m: f"{m.group(1)}{new_label}{m.group(3)}",
        original_prompt, count=1
    )
    if n_subs != 1:
        raise ValueError(
            f"apply_correct_swap: expected exactly 1 substitution for "
            f"attr_column='{attr_column}', got {n_subs}. Check attr_column "
            f"matches the exact line format in this dataset's prompt."
        )
    return corrected


def generate_counterfactual_narrative(dataset_name, instance_idx, attr_column,
                                       override_value, label, provider="openai", model=None):
    """
    Generate a counterfactual SHAP narrative: same as generate_narrative()
    in generate_narratives.py, but builds the prompt with NO override
    applied via build_shap_prompt(), then performs the PA substitution
    itself via apply_correct_swap() to avoid the override-mapping bug.

    Exactly ONE protected attribute (attr_column) is swapped per call; every
    other attribute in the prompt keeps its original value. `label` is the
    COUNTERFACTUAL_JOBS job label (e.g. "age_below_to_old") and is carried
    through to the saved result so the swap direction is unambiguous later.
    """
    prompt_module = DATASETS[dataset_name]
    result = {
        "dataset": dataset_name,
        "instance_idx": instance_idx,
        "provider": provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "narrative": None,
        "error": None,
        "prompt": None,
        "label": label,
        "attr_column": attr_column,
        "override_value": override_value,
        "target_value_label": prompt_module.map_attribute_value(attr_column, override_value),
        "exclude_protected_attributes": False,
    }

    try:
        # Build the ORIGINAL prompt with NO override -- this uses the
        # existing, unmodified, already-correct code path.
        original_prompt = prompt_module.build_shap_prompt(
            instance_idx, exclude_protected_attributes=False
        )

        # Apply our own correct substitution for the swapped attribute.
        full_prompt = apply_correct_swap(original_prompt, attr_column, override_value, prompt_module)

        # Persisted verbatim below so the exact text sent to the LLM is
        # auditable later without having to re-derive it.
        result["prompt"] = full_prompt

        messages = [
            {
                "role": "system",
                "content": "You are an expert at explaining machine learning predictions to non-technical users. Write clear narratives that help people understand model decisions."
            },
            {"role": "user", "content": full_prompt}
        ]

        narrative = generate_text(
            messages,
            provider=provider,
            model=model,
            temperature=0,
            max_tokens=4096
        )

        result["narrative"] = narrative
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def save_result(result, output_dir):
    """Save result to JSON file, using a condition folder named after the
    job label (e.g. "counterfactual_pa/age_below_to_old") so the swapped
    attribute AND direction are both unambiguous from the path alone, not
    just from the raw target value."""
    dataset = result["dataset"]
    instance = result["instance_idx"]
    provider = result["provider"]
    model = result["model"]
    label = result["label"]

    condition = f"counterfactual_pa/{label}"

    result_dir = Path(output_dir) / dataset / condition / provider / model
    result_dir.mkdir(parents=True, exist_ok=True)

    filepath = result_dir / f"instance_{instance}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return filepath


def run_batch_generation(output_dir="results/narratives", datasets_filter=None):
    """Run all (or a subset of) COUNTERFACTUAL_JOBS x PROVIDERS_TO_RUN.

    datasets_filter: optional iterable of dataset names (e.g. ["credit"]) to
    restrict the run to -- lets you generate everything for one dataset in
    a single call. None (default) runs every dataset's jobs at once.
    """
    start_time = datetime.now()
    jobs = COUNTERFACTUAL_JOBS
    if datasets_filter:
        jobs = [j for j in COUNTERFACTUAL_JOBS if j["dataset"] in datasets_filter]
        if not jobs:
            print(f"No counterfactual jobs found for dataset(s): {sorted(datasets_filter)}")
            return 1

    job_provider_pairs = [
        (job, provider, model)
        for job in jobs
        for (provider, model) in PROVIDERS_TO_RUN
    ]
    total_narratives = sum(len(job["instances"]) for job, _, _ in job_provider_pairs)

    print("=" * 80)
    print(f"GENERATING COUNTERFACTUAL NARRATIVES - BATCH MODE")
    print(f"Datasets: {sorted(set(j['dataset'] for j in jobs))}")
    print(f"Total: {total_narratives} narratives ({len(jobs)} jobs x {len(PROVIDERS_TO_RUN)} providers)")
    print("=" * 80)
    print()

    completed = 0
    failed = 0
    all_results = []

    for config_idx, (job, provider, model) in enumerate(job_provider_pairs, 1):
        dataset = job["dataset"]
        instances = job["instances"]
        attr_column = job["attr_column"]
        override_value = job["override_value"]
        label = job.get("label", "")

        print(f"[{config_idx}/{len(job_provider_pairs)}] Dataset: {dataset.upper()}, Job: {label}, "
              f"Provider: {provider.upper()}, Model: {model}")
        print(f"    Swap: {attr_column} -> {override_value}")
        print(f"    Instances: {len(instances)} total")
        print(f"    Time: {datetime.now().strftime('%H:%M:%S')}")

        for instance_idx in instances:
            print(f"  Processing instance {instance_idx}...", end=" ")
            try:
                result = generate_counterfactual_narrative(
                    dataset, instance_idx, attr_column, override_value, label,
                    provider=provider, model=model
                )
                all_results.append(result)
                filepath = save_result(result, output_dir)

                if result["status"] == "success":
                    print("OK")
                    completed += 1
                else:
                    print(f"FAIL Error: {result['error']}")
                    failed += 1
            except Exception as e:
                print(f"FAIL Exception: {str(e)}")
                failed += 1
                all_results.append({
                    "dataset": dataset,
                    "instance_idx": instance_idx,
                    "provider": provider,
                    "model": model,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        print()

    print("=" * 80)
    print("GENERATION COMPLETE")
    print(f"Completed: {completed}/{total_narratives} narratives")
    print(f"Failed: {failed}/{total_narratives} narratives")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Total time: {int(duration.total_seconds() / 60)} minutes {int(duration.total_seconds() % 60)} seconds")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Optional CLI args restrict the run to specific datasets, e.g.:
    #   python scripts/narrative_generation/generate_narratives_counterfactual.py credit
    #   python scripts/narrative_generation/generate_narratives_counterfactual.py credit saudi
    # No args runs every dataset's jobs in one go.
    cli_datasets = set(sys.argv[1:]) or None
    sys.exit(run_batch_generation(datasets_filter=cli_datasets))
