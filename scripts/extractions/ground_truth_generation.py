"""
Ground Truth Generation for Faithfulness Evaluation (SHAP Only)

This script generates files for faithfulness evaluation, organized by type and format:

1. GROUND TRUTH DATA:
   - CSV: Contains all known information from SHAP values and feature data
     - SHAP features (1-3): rank (1-3), sign (+1/-1), value (1-indexed naming: SHAP_feature_1/2/3)
     - Other features (includes protected attributes): value only
     - Path: results/ground_truth/csv/{dataset}/ground_truth_{dataset}.csv
   - JSON: Per-instance conversion with matching structure
     - Path: results/ground_truth/json/{dataset}/instance_{index}.json

2. LLM EXTRACTION TEMPLATE:
   - CSV: Features with names pre-filled, fields marked for LLM extraction (EMPTY)
     - SHAP features: name (filled), rank (EMPTY), sign (EMPTY), value (EMPTY)
     - Other features: name (filled), value (EMPTY), mentioned (EMPTY) [includes protected attributes]
     - Path: results/ground_truth/csv/{dataset}/template_{dataset}.csv
   - JSON: Same template structure with asterisks (*) for fields to fill
     - Path: results/ground_truth/json/{dataset}/template_{dataset}.json
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import sys

# Add parent path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_tools.prompts.prompt_credit import ATTRIBUTE_VALUE_MAPPINGS as CREDIT_MAPPINGS, MAX_SHAP_FEATURES as CREDIT_MAX_SHAP
from llm_tools.prompts.prompt_law import ATTRIBUTE_VALUE_MAPPINGS as LAW_MAPPINGS, MAX_SHAP_FEATURES as LAW_MAX_SHAP
from llm_tools.prompts.prompt_saudi import ATTRIBUTE_VALUE_MAPPINGS as SAUDI_MAPPINGS, MAX_SHAP_FEATURES as SAUDI_MAX_SHAP
from llm_tools.prompts.prompt_student import ATTRIBUTE_VALUE_MAPPINGS as STUDENT_MAPPINGS, MAX_SHAP_FEATURES as STUDENT_MAX_SHAP

# Configuration
DATASETS = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'shap_file': 'credit_shap.csv',
        'adverse_file': 'credit_adverse.csv',
        'target_col': 'target_credit',
        'protected_attrs': ['age', 'sex', 'foreign_worker'],
        'top_features_in_prompt': CREDIT_MAX_SHAP,
        'mappings': CREDIT_MAPPINGS
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'shap_file': 'law_shap.csv',
        'adverse_file': 'law_adverse.csv',
        'target_col': 'target_law',
        'protected_attrs': ['gender', 'race'],
        'top_features_in_prompt': LAW_MAX_SHAP,
        'mappings': LAW_MAPPINGS
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'shap_file': 'saudi_shap.csv',
        'adverse_file': 'saudi_adverse.csv',
        'target_col': 'target_saudi',
        'protected_attrs': ['Gender', 'Age', 'Health_Issues'],
        'top_features_in_prompt': SAUDI_MAX_SHAP,
        'mappings': SAUDI_MAPPINGS
    },
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'shap_file': 'student_shap.csv',
        'adverse_file': 'student_adverse.csv',
        'target_col': 'target_student',
        'protected_attrs': ['sex', 'age', 'health'],
        'top_features_in_prompt': STUDENT_MAX_SHAP,
        'mappings': STUDENT_MAPPINGS
    }
}

PROVIDERS = ['openai', 'claude', 'gemini', 'grok', 'mistral']
PROMPT_TYPE = 'shap'  # Only SHAP narratives


def get_top_features_from_narrative(shap_row, dataset_name, config):
    """
    Extract the top 3 features that should be mentioned in the narrative based on SHAP values.
    
    Args:
        shap_row: Row from SHAP CSV containing SHAP values
        dataset_name: 'credit' or 'law'
        config: Dataset configuration
        
    Returns:
        List of top feature names with their SHAP signs
    """
    # Extract SHAP columns (all columns that start with 'SHAP_')
    shap_cols = [col for col in shap_row.index if col.startswith('SHAP_')]
    
    # Get feature names and their absolute SHAP values
    features_shap = []
    for col in shap_cols:
        feature_name = col.replace('SHAP_', '')
        shap_value = shap_row[col]
        features_shap.append({
            'feature': feature_name,
            'shap_value': shap_value,
            'abs_shap': abs(shap_value),
            'sign': int(np.sign(shap_value))
        })
    
    # Sort by absolute SHAP value and get top 3
    features_shap = sorted(features_shap, key=lambda x: x['abs_shap'], reverse=True)
    top_features = features_shap[:config['top_features_in_prompt']]
    
    return top_features


def get_feature_value(feature_name, adverse_row):
    """Get feature value from adverse instance."""
    if feature_name in adverse_row.index:
        return adverse_row[feature_name]
    return np.nan


def generate_feature_distributions(dataset_name, config):
    """
    Generate feature value distributions across approved instances.
    
    Returns:
        Dictionary of {feature_name: {value: percentage, ...}, ...}
    """
    # Load training data
    train_path = os.path.join(config['path'], 'train_cleaned.parquet')
    train_df = pd.read_parquet(train_path)
    
    # Filter for approved instances (target == 1)
    approved_df = train_df[train_df[config['target_col']] == 1]
    
    distributions = {}
    
    # Get all feature columns (exclude target and ID columns)
    for col in approved_df.columns:
        if col not in [config['target_col'], 'ID', 'index']:
            if approved_df[col].dtype in ['float64', 'int64']:
                # For numerical features, compute statistics
                mean_val = approved_df[col].mean()
                std_val = approved_df[col].std()
                min_val = approved_df[col].min()
                max_val = approved_df[col].max()
                distributions[col] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(min_val),
                    'max': float(max_val)
                }
            else:
                # For categorical features, compute value percentages
                value_counts = approved_df[col].value_counts(normalize=True)
                distributions[col] = {str(val): float(pct) for val, pct in value_counts.items()}
    
    return distributions


def build_ground_truth_row(instance_idx, dataset_name, provider, config):
    """
    Build ground truth row for a single narrative (SHAP only).
    
    Ground truth contains:
    - Top features: rank (1-3), sign (+1/-1), value, avg_value
    - Other features: value, avg_value
    - Protected attributes: value only
    
    Args:
        instance_idx: Index of the adverse instance
        dataset_name: 'credit' or 'law'
        provider: LLM provider
        config: Dataset configuration
        
    Returns:
        Dictionary with ground truth information
    """
    dataset_path = config['path']
    
    # Load data files
    shap_path = os.path.join(dataset_path, config['shap_file'])
    adverse_path = os.path.join(dataset_path, config['adverse_file'])
    
    shap_df = pd.read_csv(shap_path)
    adverse_df = pd.read_csv(adverse_path)
    
    # Find the specific instance
    shap_row = shap_df[shap_df['instance_index'] == instance_idx]
    adverse_row = adverse_df[adverse_df['instance_index'] == instance_idx]
    
    if len(shap_row) == 0 or len(adverse_row) == 0:
        return None
    
    shap_row = shap_row.iloc[0]
    adverse_row = adverse_row.iloc[0]
    
    # Initialize ground truth row
    gt_row = {
        'instance_index': int(instance_idx),
        'original_test_index': int(adverse_row['original_test_index']) if 'original_test_index' in adverse_row else np.nan,
        'predicted_probability': adverse_row['prediction_score'] if 'prediction_score' in adverse_row else np.nan,
    }
    
    # Get top features based on SHAP
    top_features = get_top_features_from_narrative(shap_row, dataset_name, config)
    
    # Add top features (rank 1-3, sign, value) with SHAP_feature naming (1-indexed)
    for i, top_feat in enumerate(top_features):
        feature_name = top_feat['feature']
        rank = i + 1  # Rank 1, 2, 3 (not 0-2)
        shap_feature_num = i + 1  # SHAP_feature_1, SHAP_feature_2, SHAP_feature_3
        gt_row[f'SHAP_feature_{shap_feature_num}_name'] = feature_name
        gt_row[f'SHAP_feature_{shap_feature_num}_rank'] = rank  # Ground truth: actual rank
        gt_row[f'SHAP_feature_{shap_feature_num}_sign'] = top_feat['sign']  # Ground truth: actual sign
        gt_row[f'SHAP_feature_{shap_feature_num}_value'] = get_feature_value(feature_name, adverse_row)  # Feature value
    
    # Add all other features (value only - no rank/sign) - includes protected attributes
    all_feature_cols = [col for col in shap_row.index if col.startswith('SHAP_')]
    top_feature_names = [f['feature'] for f in top_features]
    
    # Collect all non-top features from SHAP columns
    other_features = []
    for col in all_feature_cols:
        feature_name = col.replace('SHAP_', '')
        if feature_name not in top_feature_names:
            other_features.append(feature_name)
    
    # Add protected attributes explicitly (they may not be in SHAP columns since they weren't used for model training)
    for protected_feat in config['protected_attrs']:
        if protected_feat not in top_feature_names and protected_feat not in other_features:
            other_features.append(protected_feat)
    
    # Add other features (including protected attributes with same structure)
    for i, feature_name in enumerate(other_features):
        gt_row[f'other_feature_{i}_name'] = feature_name
        gt_row[f'other_feature_{i}_value'] = get_feature_value(feature_name, adverse_row)
    
    return gt_row


def get_all_features_from_ground_truth(gt_df, config):
    """
    Extract all unique feature names from the ground truth.
    This includes both SHAP features and other_feature columns.
    Ensures consistent feature ordering across all instances in the template.
    
    Returns:
        List of unique feature names in consistent order (all 20 features)
    """
    all_features = []
    
    # First, collect SHAP features (top 3) - they should also appear in the feature list
    for i in range(config['top_features_in_prompt']):
        shap_feature_num = i + 1
        shap_col = f'SHAP_feature_{shap_feature_num}_name'
        if shap_col in gt_df.columns:
            # Get the most common SHAP feature name (since it's the same across most instances)
            common_names = gt_df[shap_col].dropna().value_counts()
            if len(common_names) > 0:
                shap_feature_name = common_names.index[0]
                if shap_feature_name not in all_features:
                    all_features.append(shap_feature_name)
    
    # Then collect other_feature names from ground truth
    other_feature_name_cols = [col for col in gt_df.columns if col.startswith('other_feature_') and '_name' in col]
    
    # Sort by feature index to maintain order
    other_feature_name_cols.sort(key=lambda x: int(x.split('_')[2]))
    
    # Extract all unique feature names from all rows
    for col in other_feature_name_cols:
        unique_names = gt_df[col].dropna().unique()
        for name in unique_names:
            if name not in all_features:
                all_features.append(name)
    
    return all_features


def normalize_template_features(template_df, gt_df, config):
    """
    Normalize the template so all instances have consistent other_feature columns.
    
    All instances will have the same set of other_feature_X_name values,
    with only mentioned and value columns empty for LLM extraction.
    """
    # Get all unique features from ground truth
    all_features = get_all_features_from_ground_truth(gt_df, config)
    
    # Create new template with normalized columns
    # Start with non-feature columns
    non_feature_cols = [col for col in template_df.columns if not col.startswith('other_feature_')]
    new_template_df = template_df[non_feature_cols].copy()
    
    # Add normalized other_feature columns for all instances
    for feat_idx, feature_name in enumerate(all_features):
        new_template_df[f'other_feature_{feat_idx}_name'] = feature_name
        new_template_df[f'other_feature_{feat_idx}_mentioned'] = np.nan
        new_template_df[f'other_feature_{feat_idx}_value'] = np.nan
    
    return new_template_df


def convert_ground_truth_to_json(gt_df, dataset_name, config):
    """
    Convert ground truth DataFrame to JSON format for each instance.
    
    JSON structure for each instance:
    {
      "predicted_probability": float or "NaN",
      "most_important_features": [
        {"rank": 1, "name": "feature_name", "sign": 1/-1, "value": float or "NaN"},
        ...
      ],
      "features": [
        {"name": "feature_name", "mentioned": 0/1, "value": float or "NaN"},
        ...
      ]
    }
    """
    output_dir = f"results/ground_truth/json/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    num_shap_features = config['top_features_in_prompt']
    converted_count = 0
    errors = []
    
    for idx, row in gt_df.iterrows():
        try:
            instance_idx = int(row["instance_index"])
            
            # Build most_important_features from SHAP columns
            most_important_features = []
            for i in range(1, num_shap_features + 1):
                shap_name_col = f"SHAP_feature_{i}_name"
                shap_rank_col = f"SHAP_feature_{i}_rank"
                shap_sign_col = f"SHAP_feature_{i}_sign"
                shap_value_col = f"SHAP_feature_{i}_value"
                
                # Check if columns exist
                if all(col in row.index for col in [shap_name_col, shap_rank_col, shap_sign_col, shap_value_col]):
                    name = str(row[shap_name_col]) if pd.notna(row[shap_name_col]) else "NaN"
                    rank = int(row[shap_rank_col]) if pd.notna(row[shap_rank_col]) else i
                    sign = int(row[shap_sign_col]) if pd.notna(row[shap_sign_col]) else 0
                    value = row[shap_value_col]
                    
                    # Convert value to appropriate type
                    if pd.isna(value):
                        value = "NaN"
                    elif isinstance(value, (int, float)):
                        if float(value) == int(value):
                            value = int(value)
                        else:
                            value = round(float(value), 2)
                    
                    most_important_features.append({
                        "rank": rank,
                        "name": name,
                        "sign": sign,
                        "value": value
                    })
            
            # Build features from other_feature columns - dynamically find all
            features = []
            i = 0
            while True:
                name_col = f"other_feature_{i}_name"
                value_col = f"other_feature_{i}_value"
                
                if name_col not in row.index or value_col not in row.index:
                    break
                
                name = str(row[name_col]) if pd.notna(row[name_col]) else "NaN"
                value = row[value_col]
                
                # Convert value to appropriate type
                if pd.isna(value):
                    value = "NaN"
                elif isinstance(value, (int, float)):
                    if float(value) == int(value):
                        value = int(value)
                    else:
                        value = round(float(value), 2)
                
                # All features in ground truth are mentioned (1)
                features.append({
                    "name": name,
                    "mentioned": 1,
                    "value": value
                })
                i += 1
            
            # Build predicted probability
            predicted_prob = row.get("predicted_probability", "NaN")
            if pd.notna(predicted_prob):
                predicted_prob = round(float(predicted_prob), 2)
            else:
                predicted_prob = "NaN"
            
            # Build JSON object
            json_obj = {
                "predicted_probability": predicted_prob,
                "most_important_features": most_important_features,
                "features": features
            }
            
            # Save to JSON file
            output_file = f"{output_dir}/instance_{instance_idx}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2)
            
            converted_count += 1
        
        except Exception as e:
            error_msg = f"Instance {idx}: {str(e)}"
            errors.append(error_msg)
    
    if converted_count > 0:
        print(f"  ✅ JSON conversion: {converted_count}/{len(gt_df)} instances saved to {output_dir}")
    
    return converted_count, errors


def save_template_json(template_df, dataset_name, config):
    """
    Save template as JSON with * for fields to fill by LLM.
    
    JSON structure:
    {
      "predicted_probability": "*",
      "most_important_features": [
        {"rank": 1, "name": "*", "sign": "*", "value": "*"},
        ...
      ],
      "features": [
        {"name": "feature_name", "mentioned": "*", "value": "*"},
        ...
      ]
    }
    """
    output_dir = f"results/ground_truth/json/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    num_shap_features = config['top_features_in_prompt']
    
    # Get first template row (all rows have same template structure)
    if len(template_df) == 0:
        return
    
    template_row = template_df.iloc[0]
    
    # Build most_important_features with asterisks
    most_important_features = []
    for i in range(1, num_shap_features + 1):
        most_important_features.append({
            "rank": i,
            "name": "*",
            "sign": "*",
            "value": "*"
        })
    
    # Build features from other_feature columns - dynamically find all
    features = []
    i = 0
    while True:
        name_col = f"other_feature_{i}_name"
        
        if name_col not in template_row.index:
            break
        
        name = str(template_row[name_col]) if pd.notna(template_row[name_col]) else "NaN"
        
        features.append({
            "name": name,
            "mentioned": "*",
            "value": "*"
        })
        i += 1
    
    # Build JSON template
    template_json = {
        "predicted_probability": "*",
        "most_important_features": most_important_features,
        "features": features
    }
    
    # Save to JSON file
    output_file = f"{output_dir}/template_{dataset_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(template_json, f, indent=2)
    
    print(f"  ✅ Template JSON saved: {output_file}")


def generate_tables():
    """
    Generate ground truth and template tables for credit, law, saudi, and student separately (SHAP only).
    One row per instance (provider-agnostic ground truth).
    """
    print("Generating ground truth and template files (SHAP narratives only)...\n")
    os.makedirs('results', exist_ok=True)
    
    for dataset_name, config in DATASETS.items():
        print(f"Processing {dataset_name.upper()} dataset...")
        
        # Load adverse instances to get instance indices
        adverse_path = os.path.join(config['path'], config['adverse_file'])
        adverse_df = pd.read_csv(adverse_path)
        
        instance_indices = adverse_df['instance_index'].unique()
        print(f"  Found {len(instance_indices)} adverse instances")
        
        # Generate ground truth rows (one per instance, provider-agnostic)
        all_rows = []
        
        for instance_idx in instance_indices:
            # Build ground truth row
            gt_row = build_ground_truth_row(instance_idx, dataset_name, None, config)
            
            if gt_row is not None:
                all_rows.append(gt_row)
            
            if len(all_rows) % 50 == 0:
                print(f"  Progress: {len(all_rows)}/{len(instance_indices)} rows processed...")
        
        # Convert to DataFrame
        gt_df = pd.DataFrame(all_rows)
        
        # Save ground truth table (in csv subdirectory)
        gt_csv_dir = f'results/ground_truth/csv/{dataset_name}'
        os.makedirs(gt_csv_dir, exist_ok=True)
        gt_output_path = os.path.join(gt_csv_dir, f'ground_truth_{dataset_name}.csv')
        gt_df.to_csv(gt_output_path, index=False)
        print(f"  ✅ Ground truth saved: {gt_output_path}")
        print(f"     Shape: {gt_df.shape}")
        
        # Convert ground truth to JSON format (in json subdirectory)
        convert_ground_truth_to_json(gt_df, dataset_name, config)
        
        # Create template table for LLM extraction with normalized feature columns
        template_df = gt_df.copy()
        
        # Empty predicted_probability
        template_df['predicted_probability'] = np.nan
        
        # Empty rank, sign, value columns for SHAP features
        for i in range(config['top_features_in_prompt']):
            shap_feature_num = i + 1
            shap_prefix = f'SHAP_feature_{shap_feature_num}'
            
            # Empty the name column (don't tell LLM what features these are)
            if f'{shap_prefix}_name' in template_df.columns:
                template_df[f'{shap_prefix}_name'] = np.nan
            
            # Empty rank, sign, value columns
            if f'{shap_prefix}_rank' in template_df.columns:
                template_df[f'{shap_prefix}_rank'] = np.nan
            if f'{shap_prefix}_sign' in template_df.columns:
                template_df[f'{shap_prefix}_sign'] = np.nan
            if f'{shap_prefix}_value' in template_df.columns:
                template_df[f'{shap_prefix}_value'] = np.nan
            
            # Remove mentioned column for SHAP features (don't ask about this for top features)
            if f'{shap_prefix}_mentioned' in template_df.columns:
                template_df.drop(columns=[f'{shap_prefix}_mentioned'], inplace=True)
        
        # Normalize other_feature columns so all instances have the same features
        template_df = normalize_template_features(template_df, gt_df, config)
        
        # Save template table (CSV)
        template_output_path = f'results/ground_truth/csv/{dataset_name}/template_{dataset_name}.csv'
        template_df.to_csv(template_output_path, index=False)
        print(f"  ✅ Template CSV saved: {template_output_path}")
        print(f"     Shape: {template_df.shape}")
        
        # Save template table (JSON)
        save_template_json(template_df, dataset_name, config)
        print()
    
    print("="*80)
    print("✅ Ground truth and template generation complete!")
    print("="*80)


def create_summary_report(gt_df):
    """Create a summary report of the ground truth table."""
    print(f"\nTable Shape: {gt_df.shape[0]} rows × {gt_df.shape[1]} columns")


def display_sample_rows(gt_df, dataset_name):
    """Display sample rows with important columns only."""
    print(f"\n{'='*80}")
    print(f"SAMPLE GROUND TRUTH ROWS - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Select important columns for display
    display_cols = [
        'instance_index', 'original_test_index', 'predicted_probability',
        'SHAP_feature_1_name', 'SHAP_feature_1_rank', 'SHAP_feature_1_sign', 'SHAP_feature_1_value',
        'SHAP_feature_2_name', 'SHAP_feature_2_rank', 'SHAP_feature_2_value',
        'SHAP_feature_3_name', 'SHAP_feature_3_rank', 'SHAP_feature_3_value',
    ]
    
    # Only show columns that exist
    display_cols = [col for col in display_cols if col in gt_df.columns]
    
    print(gt_df[display_cols].head(3).to_string())


if __name__ == "__main__":
    generate_tables()
    
    # Display sample rows from all datasets
    for dataset_name in ['credit', 'law', 'saudi', 'student']:
        try:
            gt = pd.read_csv(f'results/ground_truth/csv/{dataset_name}/ground_truth_{dataset_name}.csv')
            display_sample_rows(gt, dataset_name)
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found for {dataset_name}")
    
    print("\n" + "="*80)
    print("FILE STRUCTURE SUMMARY")
    print("="*80)
    print("\n1. Ground Truth CSV (results/ground_truth/csv/{dataset}/ground_truth_*.csv):")
    print("   ✓ instance_index, original_test_index, predicted_probability (metadata)")
    print("   ✓ SHAP_feature_X_name: feature name (FILLED, 1-indexed)")
    print("   ✓ top_feature_X: rank (1-3), sign (+1/-1), value (FILLED)")
    print("   ✓ protected_X: value (FILLED)")
    print("   ✓ other_feature_X: name, value (FILLED)")
    print("\n2. Ground Truth JSON (results/ground_truth/json/{dataset}/instance_*.json):")
    print("   ✓ predicted_probability, most_important_features, features")
    print("   ✓ Converted from CSV for each instance")
    print("\n3. Template CSV (results/ground_truth/csv/{dataset}/template_{dataset}.csv):")
    print("   ✓ instance_index, original_test_index (filled)")
    print("   ✓ predicted_probability (EMPTY - LLM fills)")
    print("   ✓ SHAP_feature_1-3: name (filled), rank (EMPTY), sign (EMPTY), value (EMPTY)")
    print("   ✓ other_feature_X: name (filled), value (EMPTY), mentioned (EMPTY) [includes protected attributes]")
    print("\n4. Template JSON (results/ground_truth/json/{dataset}/template_{dataset}.json):")
    print("   ✓ predicted_probability: \"*\"")
    print("   ✓ most_important_features: rank, name (*), sign (*), value (*)")
    print("   ✓ features: name (filled), mentioned (*), value (*)")
    print("\nDatasets processed: credit, law, saudi, student")
