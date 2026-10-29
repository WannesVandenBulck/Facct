import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

DATASET_PATH = r"datasets_prep/data/law_dataset"
TARGET_COL = "target_law"
PROTECTED_ATTRS = ["gender", "race"]
OUTPUT_DIR = "shap_plots"


def load():
    adverse_df = pd.read_csv(os.path.join(DATASET_PATH, "law_adverse.csv"))
    with open(os.path.join(DATASET_PATH, "RF.pkl"), "rb") as f:
        model = pickle.load(f)
    train_df = pd.read_parquet(os.path.join(DATASET_PATH, "train_cleaned.parquet"))
    feature_cols = train_df.drop(columns=PROTECTED_ATTRS + [TARGET_COL], errors="ignore").columns.tolist()
    return adverse_df, model, feature_cols


def explain_first(adverse_df, model, feature_cols):
    x = adverse_df[feature_cols].iloc[[0]]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x))
    base = np.atleast_1d(explainer.expected_value)

    # Keep only class 1 (adverse / declined)
    if values.ndim == 3:
        values = values[0, :, 1] if values.shape[-1] == 2 else values[1, 0, :]
    else:
        values = values[0]
    base = base[1] if base.size > 1 else base[0]

    return shap.Explanation(
        values=values,
        base_values=float(base),
        data=x.iloc[0].values,
        feature_names=feature_cols,
    )


def save(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"Saved {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adverse_df, model, feature_cols = load()
    row = adverse_df.iloc[0]
    print(f"First declined applicant: instance_index={row['instance_index']}, "
          f"original_test_index={row['original_test_index']}, score={row['prediction_score']:.4f}")

    exp = explain_first(adverse_df, model, feature_cols)

    shap.plots.waterfall(exp, max_display=len(feature_cols), show=False)
    save("law_first_declined_waterfall.png")

    shap.plots.force(exp, matplotlib=True, show=False)
    save("law_first_declined_force.png")

    shap.plots.bar(exp, max_display=len(feature_cols), show=False)
    save("law_first_declined_bar.png")

    shap.decision_plot(exp.base_values, exp.values, features=exp.data,
                       feature_names=feature_cols, show=False)
    save("law_first_declined_decision.png")


if __name__ == "__main__":
    main()
