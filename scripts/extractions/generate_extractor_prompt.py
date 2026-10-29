import pandas as pd
import json
import os
import glob
from pathlib import Path
import sys
import pickle

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_tools.prompts.prompt_credit import ATTRIBUTE_VALUE_MAPPINGS as CREDIT_MAPPINGS, MAX_SHAP_FEATURES as CREDIT_MAX_SHAP, DECISION_THRESHOLD as CREDIT_THRESHOLD
from llm_tools.prompts.prompt_law import ATTRIBUTE_VALUE_MAPPINGS as LAW_MAPPINGS, MAX_SHAP_FEATURES as LAW_MAX_SHAP, DECISION_THRESHOLD as LAW_THRESHOLD
from llm_tools.prompts.prompt_saudi import ATTRIBUTE_VALUE_MAPPINGS as SAUDI_MAPPINGS, MAX_SHAP_FEATURES as SAUDI_MAX_SHAP, DECISION_THRESHOLD as SAUDI_THRESHOLD
from llm_tools.prompts.prompt_student import ATTRIBUTE_VALUE_MAPPINGS as STUDENT_MAPPINGS, MAX_SHAP_FEATURES as STUDENT_MAX_SHAP, DECISION_THRESHOLD as STUDENT_THRESHOLD

DATASETS = {
    "credit": {
        "path": r"datasets_prep/data/credit_dataset",
        "shap_file": "credit_shap.csv",
        "adverse_file": "credit_adverse.csv",
        "target_col": "target_credit",
        "protected_attrs": ["age", "sex", "foreign_worker"],
        "num_top_features": CREDIT_MAX_SHAP,
        "mappings": CREDIT_MAPPINGS,
        "dataset_info_file": "datasets_prep/data/credit_dataset/dataset_info",
        # same wording as the narrative prompt (llm_tools/prompts/prompt_credit.py)
        "person": "applicant",
        "pos": "bad credit risk",
        "neg": "good credit risk",
        "favourable": "approved applicants",
        "threshold": CREDIT_THRESHOLD,
    },
    "law": {
        "path": r"datasets_prep/data/law_dataset",
        "shap_file": "law_shap.csv",
        "adverse_file": "law_adverse.csv",
        "target_col": "target_law",
        "protected_attrs": ["gender", "race"],
        "num_top_features": LAW_MAX_SHAP,
        "mappings": LAW_MAPPINGS,
        "dataset_info_file": "datasets_prep/data/law_dataset/dataset_info",
        # same wording as the narrative prompt (llm_tools/prompts/prompt_law.py)
        "person": "student",
        "pos": "bar exam failure",
        "neg": "bar exam success",
        "favourable": "admitted students",
        "threshold": LAW_THRESHOLD,
    },
    "saudi": {
        "path": r"datasets_prep/data/saudi_dataset",
        "shap_file": "saudi_shap.csv",
        "adverse_file": "saudi_adverse.csv",
        "target_col": "target_saudi",
        "protected_attrs": ["Gender", "Age", "Health_Issues"],
        "num_top_features": SAUDI_MAX_SHAP,
        "mappings": SAUDI_MAPPINGS,
        "dataset_info_file": "datasets_prep/data/saudi_dataset/dataset_info",
        # same wording as the narrative prompt (llm_tools/prompts/prompt_saudi.py)
        "person": "employee",
        "pos": "leaving the job",
        "neg": "staying in the job",
        "favourable": "employees who were granted a promotion",
        "threshold": SAUDI_THRESHOLD,
    },
    "student": {
        "path": r"datasets_prep/data/student_dataset",
        "shap_file": "student_shap.csv",
        "adverse_file": "student_adverse.csv",
        "target_col": "target_student",
        "protected_attrs": ["sex", "age", "health"],
        "num_top_features": STUDENT_MAX_SHAP,
        "mappings": STUDENT_MAPPINGS,
        "dataset_info_file": "datasets_prep/data/student_dataset/dataset_info",
        # same wording as the narrative prompt (llm_tools/prompts/prompt_student.py)
        "person": "student",
        "pos": "failing the final year",
        "neg": "passing the final year",
        "favourable": "students who are not obliged to take remedial courses",
        "threshold": STUDENT_THRESHOLD,
    }
}


def load_dataset_info(dataset_name, config):
    """Load dataset information."""
    info_file = config["dataset_info_file"]
    with open(info_file, "rb") as f:
        dataset_info = pickle.load(f)
    return dataset_info


def load_narrative(dataset_name, instance_idx, provider="openai", prompt_type="shap", condition=None):
    """Load the narrative JSON file."""
    if not condition:
        raise ValueError("condition is required (include_pa, exclude_pa or counterfactual_pa/<label>)")
    narrative_pattern = f"results/narratives/{dataset_name}/{condition}/{provider}/**/instance_{instance_idx}.json"
    files = glob.glob(narrative_pattern, recursive=True)
    if files:
        with open(files[0], "r", encoding="utf-8") as f:
            narrative_data = json.load(f)
        return narrative_data.get("narrative", "")
    return None


def load_ground_truth_row(dataset_name, instance_idx):
    """Load ground truth row for comparison reference."""
    gt_path = f"results/ground_truth/csv/{dataset_name}/ground_truth_{dataset_name}.csv"
    gt_df = pd.read_csv(gt_path)
    row = gt_df[gt_df["instance_index"] == instance_idx]
    if len(row) > 0:
        return row.iloc[0]
    return None


def number_to_word(num):
    """Convert number to English word (1-10)."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(num, str(num))


def number_to_ordinal(num):
    """Convert number to ordinal word (1st, 2nd, 3rd, etc)."""
    ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
                6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
    return ordinals.get(num, f"{num}th")


def format_attribute_mappings(dataset_name, config):
    """Format attribute value mappings for the prompt."""
    mappings_text = ""
    mappings = config["mappings"]

    for feature_name, value_dict in mappings.items():
        if not value_dict:  # numeric features (e.g. age) have no mapping
            continue
        mappings_text += f"**{feature_name}:**\n"
        for code, readable in value_dict.items():
            code = int(code) if float(code).is_integer() else code
            mappings_text += f"  - {code}: {readable}\n"
        mappings_text += "\n"
    
    return mappings_text


def load_template_json(dataset_name):
    """Load pre-generated JSON template from file."""
    template_path = f"results/ground_truth/json/{dataset_name}/template_{dataset_name}.json"
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template_json = json.load(f)
        return json.dumps(template_json, indent=2)
    except FileNotFoundError:
        return f"Error: Template not found at {template_path}"
    except json.JSONDecodeError:
        return f"Error: Invalid JSON in template file {template_path}"


def generate_extractor_prompt(dataset_name, instance_idx, provider="openai", prompt_type="shap", condition=None):
    """
    Generate the complete extractor prompt (consolidated and token-optimized).
    """
    config = DATASETS[dataset_name]

    # Load all necessary data
    dataset_info = load_dataset_info(dataset_name, config)
    narrative = load_narrative(dataset_name, instance_idx, provider, prompt_type, condition=condition)
    
    if not narrative:
        return f"Error: Narrative not found for {dataset_name} instance {instance_idx} with provider {provider}"
    
    # Extract dataset info values
    dataset_description = dataset_info["dataset_description"]
    target_description = dataset_info["target_description"]
    person, pos, neg, favourable = config["person"], config["pos"], config["neg"], config["favourable"]
    threshold = f"{config['threshold']:.0%}"
    feature_df_str = dataset_info["feature_description"][["feature_name", "feature_desc"]].to_string(index=False)
    
    # Get dynamic number of top features
    num_top_features = config["num_top_features"]
    num_word = number_to_word(num_top_features)
    
    # Build the prompt
    prompt = f"""EXTRACTION TASK: Analyze the following narrative to extract prediction explanation data.

CONTEXT: An LLM generated this narrative to explain why a model predicted a negative outcome for this instance. You must extract information from the narrative to validate its accuracy.

1. DATASET & TARGET INFORMATION

Dataset: {dataset_description}

Target: {target_description}

Features:
{feature_df_str}

Categorical Feature Mappings (translate mentions to codes):

{format_attribute_mappings(dataset_name, config)}

2. WHAT TO EXTRACT

You must fill a JSON object with three sections:

SECTION 1: predicted_probability
The model's predicted probability of {pos} for this {person}, if explicitly mentioned in the narrative (as decimal 0.0-1.0). Otherwise fill with "NaN".
This is NOT the decision threshold ({threshold}), which the narrative may also mention.
Do not guess - only fill if clearly stated.

SECTION 2: most_important_features
These are the TOP {num_top_features} most important features the narrative emphasizes.

FOR EACH of the {num_top_features} ranks (1 to {num_top_features}):
- name: The official feature name from the feature list above (exact match, case-sensitive). 
  Fill with: The actual feature name as stated in the feature list above
  
- sign: 1 if the narrative says this feature pushes toward {pos} (the adverse outcome), -1 if it pushes toward {neg}.
  Fill with: 1 or -1 only
  
- value: 
    The numeric value of a numeric feature for this instance as mentioned in narrative.
    For categorical features, fill with the numeric code corresponding to the mentioned category (using mappings above).
    If the narrative does not mention a value for this feature, fill with "NaN".
  Fill with: A number, decimal, or "NaN" if not mentioned
  
- rank: Already filled (1, 2, 3, etc.) - DO NOT CHANGE

SECTION 3: features
ALL features from the narrative are listed here with their names ALREADY FILLED IN.
You ONLY need to fill:

- mentioned: 1 if this feature is discussed in narrative, 0 if not mentioned
  Fill with: 0 or 1 only
  
- value: The value of a feature as mentioned in the narrative
  Fill with: A number, decimal, code, or "NaN" if not mentioned
  
- name: ALREADY PRE-FILLED - DO NOT CHANGE

3. FORMAT RULES 
- Numbers: No decimals for integers (24 not 24.0), decimals include point (0.87, 0.42), return numeric values without quotes (24, not "24")
- NaN: Use exact string "NaN" (not nan, null, or empty)
- Binary (0/1): Only 0 or 1, never true/false/yes/no
- Codes: Use numeric codes from mappings (not readable text)
- Feature names: MUST match exactly from feature list (case-sensitive)

4. TEMPLATE TO FILL (JSON FORMAT)

Fill all fields marked with * based on the narrative. Feature names in "features" are pre-filled.

```json
{load_template_json(dataset_name)}
```

5. NARRATIVE TO EXTRACT FROM

{narrative}

6. FINAL INSTRUCTIONS
- The narrative might not state features explicitely, but include descriptions that imply them. 
- It is your task to extract the features the narrative includes by leveraging the feature descriptions and categorical feature mappings. 
- Based on the feature names, categorical mappings, JSON format and these instructions, you must extract the relevant information from the narrative and fill in the JSON template accurately.
- Values are always this {person}'s own values. The narrative may cite averages or distributions among {favourable} for comparison: never extract those as the {person}'s value.
- If the narrative does not mention a specific piece of information, fill with "NaN" (for values) or 0 (for mentioned) as appropriate. Do not guess or infer beyond what is explicitly stated.
- Return ONLY the completed JSON in a code block, no other text or explanation, and do NOT change the template format
"""
    return prompt


def generate_and_display_prompt(dataset_name, instance_idx, provider="openai", prompt_type="shap"):
    """Generate prompt and display it."""
    prompt = generate_extractor_prompt(dataset_name, instance_idx, provider, prompt_type)
    print(prompt)
    
    output_file = f"results/extractor_prompt_{dataset_name}_instance_{instance_idx}_{provider}.txt"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n? Prompt saved to: {output_file}")
    
    return prompt


if __name__ == "__main__":
    dataset_name = "credit"
    instance_idx = 0
    provider = "openai"
    prompt_type = "shap"
    generate_and_display_prompt(dataset_name, instance_idx, provider, prompt_type)
