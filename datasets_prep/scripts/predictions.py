import pandas as pd
import numpy as np
import pickle
import os

# Define dataset configurations
# Target encoding: 1 = adverse/bad class (denied credit, failed exam)
#                 0 = favorable/good class (approved credit, passed exam)
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'target_col': 'target_credit',           
        'output_file': 'credit_adverse',
        'threshold': 0.25  # Modify this to get more/fewer adverse instances
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'target_col': 'target_law',                  
        'output_file': 'law_adverse',
        'threshold': 0.5,
        # Keep the 0.5 threshold and draw a race x gender stratified subsample of the adverse set instead
        'adverse_sample': {'n': 100, 'strata': ['race', 'gender'], 'seed': 1234}
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'target_col': 'target_saudi',
        'output_file': 'saudi_adverse',
        'threshold': 0.4
    }, 
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'target_col': 'target_student',
        'output_file': 'student_adverse',
        'threshold': 0.1  # Modify this to get more/fewer adverse instances
    }
}

def stratified_sample(df, n, strata, seed):
    """Proportional stratified sample of n rows; largest-remainder allocation so the total is exactly n."""
    sizes = df.groupby(strata).size()
    exact = sizes * n / len(df)
    alloc = np.floor(exact).astype(int)
    leftover = n - alloc.sum()
    alloc[(exact - alloc).sort_values(ascending=False, kind='stable').index[:leftover]] += 1
    parts = [group.sample(alloc.loc[key], random_state=seed) for key, group in df.groupby(strata)]
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
    if sample:
        n_before = len(test_adverse)
        test_adverse = stratified_sample(test_adverse, sample['n'], sample['strata'], sample['seed'])
        print(f"  Stratified subsample by {sample['strata']}: {len(test_adverse)} of {n_before} adverse instances")
        print(pd.crosstab(test_adverse[sample['strata'][0]], test_adverse[sample['strata'][1]], margins=True).to_string())

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
