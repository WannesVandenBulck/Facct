import pandas as pd
import numpy as np
import pickle
import os

# Two-level comparison groups per protected attribute; these mirror assign_groups() in
# scripts/fairness_eval/RQ1_total_disparity.py so the sample is balanced on the groups that get tested
def median_split(s):
    return np.where(s < s.median(), 'below_median', 'at_or_above_median')

COMPARISON_GROUPS = {
    'credit': {
        'age': median_split,
        'sex': lambda s: s.map({0: 'male', 1: 'female'}),
        'foreign_worker': lambda s: s.map({1: 'yes', 2: 'no'}),
    },
    'law': {
        'gender': lambda s: s.map({0: 'female', 1: 'male'}),
        'race': lambda s: np.where(s == 0, 'white', 'non_white'),
    },
    'saudi': {
        'Gender': lambda s: s.map({0: 'female', 1: 'male'}),
        'Age': lambda s: np.where(s == 0, '21-30', '31+'),
        'Health_Issues': lambda s: s.map({0: 'no', 1: 'yes'}),
    },
    'student': {
        'sex': lambda s: s.map({0: 'female', 1: 'male'}),
        'age': median_split,
        'health': lambda s: np.where(s <= 3, 'very_bad_bad_fair', 'good_very_good'),
    },
}

# Define dataset configurations
# Target encoding: 1 = adverse/bad class (denied credit, failed exam)
#                 0 = favorable/good class (approved credit, passed exam)
# Each adverse set is reduced to a balanced sample of 100 over the comparison groups (see balanced_sample)
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'target_col': 'target_credit',           
        'output_file': 'credit_adverse',
        'threshold': 0.25,  # Modify this to get more/fewer adverse instances
        'adverse_sample': {'n': 100, 'seed': 1234}
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'target_col': 'target_law',                  
        'output_file': 'law_adverse',
        'threshold': 0.5,
        'adverse_sample': {'n': 100, 'seed': 1234}
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'target_col': 'target_saudi',
        'output_file': 'saudi_adverse',
        'threshold': 0.4,
        'adverse_sample': {'n': 100, 'seed': 1234}
    }, 
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'target_col': 'target_student',
        'output_file': 'student_adverse',
        'threshold': 0.1,  # Modify this to get more/fewer adverse instances
        'adverse_sample': {'n': 100, 'seed': 1234}
    }
}

def balanced_sample(df, n, groups, seed):
    """Sample n rows spread as evenly as possible over the joint cells of the comparison groups.

    Every cell gets an equal share; cells smaller than their share contribute all their rows and
    the shortfall is redistributed over the remaining cells, so small groups are kept whole.
    """
    cells = pd.DataFrame({attr: np.asarray(fn(df[attr])) for attr, fn in groups.items()}, index=df.index)
    cell_key = cells.astype(str).agg('|'.join, axis=1)
    sizes = cell_key.value_counts().sort_index()

    alloc = pd.Series(0, index=sizes.index)
    open_cells, remaining = list(sizes.index), n
    while open_cells and remaining > 0:
        share = remaining // len(open_cells)
        full = [c for c in open_cells if sizes[c] <= share]
        if not full:
            alloc[open_cells] = share
            # spread the remainder over randomly chosen open cells
            extra = np.random.default_rng(seed).choice(open_cells, remaining - share * len(open_cells), replace=False)
            alloc[list(extra)] += 1
            break
        alloc[full] = sizes[full]
        remaining -= sizes[full].sum()
        open_cells = [c for c in open_cells if c not in full]

    parts = [df[cell_key == c].sample(alloc[c], random_state=seed) for c in sizes.index]
    return pd.concat(parts).sort_values('instance_index')

def make_predictions(dataset_name, config):
    """Load model and make predictions for the test set.
    
    Predictions are filtered for class 1 (adverse/bad class) and class 0 (positive/good class).
    The target value 1 represents the adverse outcome for each dataset.
    The target value 0 represents the favorable outcome for each dataset.
    Model is trained without use of defined protected attributes. 
    
    Outputs:
    - {dataset}_adverse.csv: Instances with predicted_class == 1
    - {dataset}_positive.csv: Instances with predicted_class == 0
    """
    
    dataset_path = config['path']
    target_col = config['target_col'] 
    output_file = config['output_file']
    
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load the Random Forest model
    model_path = os.path.join(dataset_path, 'RF.pkl')
    
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
        
    # Load test data
    test_path = os.path.join(dataset_path, 'test_cleaned.parquet')
    test_df = pd.read_parquet(test_path)
    
    # Separate features and target
    test_features = test_df.drop(columns=[target_col])
    test_target = test_df[target_col]
    
    # Create a copy without protected attributes for model prediction
    # but keep original test_features with all columns for saving later
    if dataset_name == 'law':
        protected_attributes = ['gender', 'race']
        features_for_prediction = test_features.drop(columns=protected_attributes)
    elif dataset_name == 'credit':
        protected_attributes = ['age', 'sex', 'foreign_worker']
        features_for_prediction = test_features.drop(columns=protected_attributes)
    elif dataset_name == 'saudi':
        protected_attributes = ['Gender', 'Age', 'Health_Issues']
        features_for_prediction = test_features.drop(columns=protected_attributes)
    else:  # dataset_name == 'student'
        protected_attributes = ['sex', 'age', 'health']
        features_for_prediction = test_features.drop(columns=protected_attributes)
    
    # Make predictions on test set
    test_pred = rf_model.predict(features_for_prediction)
    
    # Get prediction probabilities for class 1 (bad class)
    test_proba = rf_model.predict_proba(features_for_prediction)[:, 1]
    
    # Apply threshold tuning (dataset-specific)
    threshold = config.get('threshold', 0.5)
    adverse_mask = test_proba >= threshold
    positive_mask = test_proba < threshold  # Complementary mask for positive predictions
    
    # Get predicted class based on threshold (1 if >= threshold, 0 otherwise)
    test_pred_threshold = (test_proba >= threshold).astype(int)
    
    # Save adversly classified instances
    test_adverse = test_features[adverse_mask].copy()
    test_adverse['predicted_class'] = test_pred_threshold[adverse_mask]
    test_adverse['prediction_score'] = test_proba[adverse_mask]
    test_adverse[target_col] = test_target[adverse_mask]
    
    # Capture original test set indices before resetting
    original_indices_adverse = test_features[adverse_mask].index.values
    
    # Add explicit instance_index column (sequential 0, 1, 2, ...)
    test_adverse.reset_index(drop=True, inplace=True)
    test_adverse.insert(0, 'instance_index', range(len(test_adverse)))
    test_adverse.insert(1, 'original_test_index', original_indices_adverse)

    # instance_index keeps its full-adverse-set value so IDs stay stable across subsampling
    sample = config.get('adverse_sample')
    if sample and len(test_adverse) < sample['n']:
        print(f"  WARNING: only {len(test_adverse)} adverse instances, fewer than the requested sample of {sample['n']}; lower the threshold")
    elif sample:
        n_before = len(test_adverse)
        groups = COMPARISON_GROUPS[dataset_name]
        test_adverse = balanced_sample(test_adverse, sample['n'], groups, sample['seed'])
        print(f"  Balanced sample over {list(groups)}: {len(test_adverse)} of {n_before} adverse instances")
        for attr, fn in groups.items():
            print(f"    {attr}: {pd.Series(np.asarray(fn(test_adverse[attr]))).value_counts().to_dict()}")

    # Save adverse to CSV without index (instance_index is now an explicit column)
    output_path_adverse = os.path.join(dataset_path, f'{output_file}.csv')
    test_adverse.to_csv(output_path_adverse, index=False)
    
    print(f"  Saved {len(test_adverse)} adverse instances to {output_path_adverse}")
    
    # Save the positively predicted instances
    test_positive = test_features[positive_mask].copy()
    test_positive['predicted_class'] = test_pred_threshold[positive_mask]
    test_positive['prediction_score'] = test_proba[positive_mask]
    test_positive[target_col] = test_target[positive_mask]
    
    # Capture original test set indices before resetting
    original_indices_positive = test_features[positive_mask].index.values
    
    # Add explicit instance_index column (sequential 0, 1, 2, ...)
    test_positive.reset_index(drop=True, inplace=True)
    test_positive.insert(0, 'instance_index', range(len(test_positive)))
    test_positive.insert(1, 'original_test_index', original_indices_positive)
    
    # Save positive to CSV without index (instance_index is now an explicit column)
    # Use 'positive' suffix instead of 'adverse'
    output_file_positive = output_file.replace('adverse', 'positive')
    output_path_positive = os.path.join(dataset_path, f'{output_file_positive}.csv')
    test_positive.to_csv(output_path_positive, index=False)
    
    print(f"  Saved {len(test_positive)} positive instances to {output_path_positive}")
    
    return test_adverse

# Process all datasets
all_results = {}
for dataset_name, config in datasets.items():
    results = make_predictions(dataset_name, config)
    all_results[dataset_name] = results

print("Prediction Summary:")
for dataset_name, results in all_results.items():
    print(f"{dataset_name.upper()}: {len(results)} adverse instances")
