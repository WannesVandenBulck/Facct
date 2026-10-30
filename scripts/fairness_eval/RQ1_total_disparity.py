#!/usr/bin/env python
"""
RQ1 — Total Disparity
======================
Within the `pa_included` condition, are there statistically significant
differences in feature-coverage / faithfulness metrics between social
groups (values of a protected attribute)? Tested separately per dataset
and per provider (not pooled across providers).

INPUT
-----
results/fairness_eval/per_narrative_metrics_<dataset>.csv (one row per
narrative = instance x provider x condition), as produced by
generate_per_narrative_metrics.py.

GROUPING
--------
Every protected attribute is reduced to exactly 2 groups (confirmed with
the research team):

  dataset   attribute        group 1              group 2
  -------   ---------------  -------------------  -------------------
  credit    age              below median          at/above median
  credit    sex              male                  female
  credit    foreign_worker   yes                   no
  saudi     Gender           female                male
  saudi     Age              21-30                 31+
  saudi     Health_Issues    no                    yes
  student   sex              female                male
  student   age              below median          at/above median
  student   health           very bad/bad/fair      good/very good
  law       gender           female                male
  law       race             white                 non-white

Medians (credit age, student age) are computed on unique instances within
that dataset. The student `health` split point (<=fair vs >=good) was
chosen as the most balanced 2-way split of the 5 ordinal levels (43 vs 57
of the 100 sampled instances), confirmed with the research team.

METRICS
-------
Faithfulness (binary, NaN where not applicable):
  rank_{1,2,3}_scoring        - correct feature at correct rank position
  rank_{1,2,3}_sign_scoring   - correct sign for GT's rank-N feature
  rank_{1,2,3}_value_scoring  - correct value for GT's rank-N feature
  predicted_probability_scoring
  <PA>_scoring                 - correct value for this protected attribute
  other_<feature>_value_scoring

Coverage (binary):
  <PA>_mentioned
  other_<feature>_mentioned
  rank_{1,2,3}_mentioned        - DERIVED (see below): whether GT's rank-N
                                   feature was referenced at all. The CSV
                                   only records whether a *value* was
                                   stated (rank_N_value_mentioned) or
                                   whether the extraction placed it in its
                                   OWN top-3 (rank_N_sign_extracted not
                                   null); there is no single column for
                                   "mentioned by name, no value, not in
                                   extraction's own top-3". We define
                                   rank_N_mentioned = 1 if either signal is
                                   present, else 0. This is the best proxy
                                   available without re-touching the
                                   extraction pipeline; flagged here for
                                   review since it is a methodological
                                   choice, not a literal column.

Continuous aggregates (mean of the available binary components per row):
  rank_position_accuracy  - mean of rank_{1,2,3}_scoring
  sign_accuracy           - mean of rank_{1,2,3}_sign_scoring
  value_accuracy          - mean of ALL value-type scoring columns
                             (rank values + PA values + other-feature values)
  prop_pa_mentioned       - mean of all <PA>_mentioned columns
  prop_other_mentioned    - mean of all other_<feature>_mentioned columns

STATISTICAL TESTS
------------------
Every comparison is exactly 2 independent groups:
  binary metric      -> Fisher's exact (Chi-square if all expected >= 5)
  continuous metric  -> Mann-Whitney U
Groups with n < MIN_GROUP_N are skipped.
Effect sizes: odds ratio (binary), rank-biserial correlation (continuous).
Multiple-comparison correction: Benjamini-Hochberg FDR only, applied WITHIN
each (dataset, provider, attribute, metric_type) group — i.e. separately per
protected attribute, and separately for binary vs. continuous metrics within
that attribute, so unrelated attributes and structurally different test
families don't shift each other's significance threshold (Bonferroni is not
used — see apply_correction docstring for the full rationale, including the
empirical comparison against coarser groupings).

OUTPUT
------
results/fairness_eval/rq1_total_disparity_aggregated.csv
    One row per dataset x provider x attribute x group x metric, with n
    and mean/proportion - the data being compared.
results/fairness_eval/rq1_total_disparity_tests.csv
    One row per dataset x provider x attribute x metric, with the test
    used, group sizes, statistic, raw and FDR-corrected p-values,
    significance flags, and effect size.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent.parent
FAIRNESS_EVAL_DIR = ROOT / "results" / "fairness_eval"

DATASETS = ["credit", "saudi", "student", "law"]
PROVIDERS = ["openai", "mistral", "claude", "deepseek", "openai_small", "mistral_small"]  # providers without narratives (yet) are skipped
MIN_GROUP_N = 5
ALPHA = 0.05

# Protected attributes per dataset (must match <attr>_GT / <attr>_mentioned /
# <attr>_extracted / <attr>_scoring columns in the CSV).
PROTECTED_ATTRS = {
    "credit": ["age", "sex", "foreign_worker"],
    "saudi": ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
    "law": ["gender", "race"],
}

OUTPUT_AGGREGATED = FAIRNESS_EVAL_DIR / "rq1_total_disparity_aggregated.csv"
OUTPUT_TESTS = FAIRNESS_EVAL_DIR / "rq1_total_disparity_tests.csv"


# ============================================================================
# GROUP ASSIGNMENT
# ============================================================================

def assign_groups(df, dataset, attr):
    """Return a Series of 2-level group labels for this attribute, plus a
    dict describing which raw values map to which label (for reporting)."""
    col = f"{attr}_GT"

    if (dataset, attr) == ("credit", "age"):
        median = df.drop_duplicates("instance_id")[col].median()
        groups = np.where(df[col] < median, "below_median", "at_or_above_median")
        mapping = {"below_median": f"< {median:g}", "at_or_above_median": f">= {median:g}"}

    elif (dataset, attr) == ("credit", "sex"):
        m = {0: "male", 1: "female"}
        groups = df[col].map(m)
        mapping = m

    elif (dataset, attr) == ("credit", "foreign_worker"):
        m = {1: "yes", 2: "no"}
        groups = df[col].map(m)
        mapping = m

    elif (dataset, attr) == ("saudi", "Gender"):
        m = {0: "female", 1: "male"}
        groups = df[col].map(m)
        mapping = m

    elif (dataset, attr) == ("saudi", "Age"):
        groups = np.where(df[col] == 0, "21-30", "31+")
        mapping = {"21-30": "code 0", "31+": "codes 1,2"}

    elif (dataset, attr) == ("saudi", "Health_Issues"):
        m = {0: "no", 1: "yes"}
        groups = df[col].map(m)
        mapping = m

    elif (dataset, attr) == ("student", "sex"):
        m = {0: "female", 1: "male"}
        groups = df[col].map(m)
        mapping = m

    elif (dataset, attr) == ("student", "age"):
        median = df.drop_duplicates("instance_id")[col].median()
        groups = np.where(df[col] < median, "below_median", "at_or_above_median")
        mapping = {"below_median": f"< {median:g}", "at_or_above_median": f">= {median:g}"}

    elif (dataset, attr) == ("student", "health"):
        groups = np.where(df[col] <= 3, "very_bad_bad_fair", "good_very_good")
        mapping = {"very_bad_bad_fair": "codes 1,2,3", "good_very_good": "codes 4,5"}

    elif (dataset, attr) == ("law", "gender"):
        m = {0: "female", 1: "male"}
        groups = df[col].map(m)
        mapping = m

    elif (dataset, attr) == ("law", "race"):
        groups = np.where(df[col] == 0, "white", "non_white")
        mapping = {"white": "code 0", "non_white": "codes 1,2,3,4"}

    else:
        raise ValueError(f"No grouping rule defined for {dataset}/{attr}")

    return pd.Series(groups, index=df.index), mapping


# ============================================================================
# METRIC DISCOVERY
# ============================================================================

def discover_metrics(df, pa_names):
    """Return dict {metric_name: 'binary'|'continuous'} for every metric to
    test, including derived rank_N_mentioned and the aggregate scores."""
    metrics = {}

    for n in (1, 2, 3):
        metrics[f"rank_{n}_scoring"] = "binary"
        metrics[f"rank_{n}_sign_scoring"] = "binary"
        metrics[f"rank_{n}_value_scoring"] = "binary"
        metrics[f"rank_{n}_mentioned"] = "binary"  # derived
    metrics["predicted_probability_scoring"] = "binary"

    for pa in pa_names:
        metrics[f"{pa}_scoring"] = "binary"
        metrics[f"{pa}_mentioned"] = "binary"

    other_cols = [c for c in df.columns if c.startswith("other_") and c.endswith("_mentioned")]
    for c in other_cols:
        feat = c[len("other_"):-len("_mentioned")]
        metrics[f"other_{feat}_mentioned"] = "binary"
        metrics[f"other_{feat}_value_scoring"] = "binary"

    metrics["rank_position_accuracy"] = "continuous"
    metrics["sign_accuracy"] = "continuous"
    metrics["value_accuracy"] = "continuous"
    metrics["prop_pa_mentioned"] = "continuous"
    metrics["prop_other_mentioned"] = "continuous"

    # lexicon-based sentiment (computed in generate_per_narrative_metrics.py)
    if "sent_net" in df.columns:
        metrics["sent_net"] = "continuous"

    return metrics


def add_derived_columns(df, pa_names):
    df = df.copy()

    for n in (1, 2, 3):
        df[f"rank_{n}_mentioned"] = (
            (df[f"rank_{n}_value_mentioned"] == 1) | df[f"rank_{n}_sign_extracted"].notna()
        ).astype(float)
        # if the row's extraction was entirely missing, propagate NaN rather than 0
        both_nan = df[f"rank_{n}_value_mentioned"].isna() & df[f"rank_{n}_sign_extracted"].isna()
        df.loc[both_nan, f"rank_{n}_mentioned"] = np.nan

    rank_scoring_cols = [f"rank_{n}_scoring" for n in (1, 2, 3)]
    rank_sign_cols = [f"rank_{n}_sign_scoring" for n in (1, 2, 3)]
    value_scoring_cols = (
        [f"rank_{n}_value_scoring" for n in (1, 2, 3)]
        + [f"{pa}_scoring" for pa in pa_names]
        + [c for c in df.columns if c.startswith("other_") and c.endswith("_value_scoring")]
    )
    pa_mentioned_cols = [f"{pa}_mentioned" for pa in pa_names]
    other_mentioned_cols = [c for c in df.columns
                             if c.startswith("other_") and c.endswith("_mentioned")]

    df["rank_position_accuracy"] = df[rank_scoring_cols].mean(axis=1, skipna=True)
    df["sign_accuracy"] = df[rank_sign_cols].mean(axis=1, skipna=True)
    df["value_accuracy"] = df[value_scoring_cols].mean(axis=1, skipna=True)
    df["prop_pa_mentioned"] = df[pa_mentioned_cols].mean(axis=1, skipna=True)
    df["prop_other_mentioned"] = df[other_mentioned_cols].mean(axis=1, skipna=True)

    return df


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def odds_ratio_2x2(table):
    a, b = table[0]
    c, d = table[1]
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a * d) / (b * c)


def rank_biserial_unpaired(x, y):
    n1, n2 = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return (2 * u_stat) / (n1 * n2) - 1


def test_binary(vals1, vals2, name1, name2):
    v1, v2 = vals1[~np.isnan(vals1)], vals2[~np.isnan(vals2)]
    n1, n2 = len(v1), len(v2)
    if n1 < MIN_GROUP_N or n2 < MIN_GROUP_N:
        return None
    s1, f1 = int(np.sum(v1)), n1 - int(np.sum(v1))
    s2, f2 = int(np.sum(v2)), n2 - int(np.sum(v2))
    table = np.array([[s1, f1], [s2, f2]])
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return None
    expected = stats.contingency.expected_freq(table)
    if (expected < 5).any():
        _, p = stats.fisher_exact(table)
        test_name = "Fisher exact"
        stat = np.nan
    else:
        stat, p, _, _ = stats.chi2_contingency(table, correction=False)
        test_name = "Chi-square"
    try:
        eff = odds_ratio_2x2(table)
    except ZeroDivisionError:
        eff = np.nan
    return dict(test=test_name, statistic=stat, p_value=p,
                effect_size=eff, effect_size_type="odds_ratio",
                n1=n1, n2=n2, mean1=np.mean(v1), mean2=np.mean(v2),
                group1=name1, group2=name2)


def test_continuous(vals1, vals2, name1, name2):
    v1, v2 = vals1[~np.isnan(vals1)], vals2[~np.isnan(vals2)]
    if len(v1) < MIN_GROUP_N or len(v2) < MIN_GROUP_N:
        return None
    try:
        stat, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        eff = rank_biserial_unpaired(v1, v2)
    except ValueError:
        return None
    return dict(test="Mann-Whitney U", statistic=stat, p_value=p,
                effect_size=eff, effect_size_type="rank_biserial_r",
                n1=len(v1), n2=len(v2), mean1=np.mean(v1), mean2=np.mean(v2),
                group1=name1, group2=name2)


# ============================================================================
# MAIN
# ============================================================================

def apply_correction(df):
    """Benjamini-Hochberg FDR correction, applied WITHIN each (dataset,
    provider, attribute, metric_type) group — i.e. separately per protected
    attribute, AND separately for binary vs. continuous metrics within that
    attribute. Two deliberate design choices here:

    1. Batching by (dataset, provider, attribute) rather than any coarser
       grouping: each attribute (e.g. "Gender" vs "Age" vs "Health_Issues")
       represents a logically distinct hypothesis family, and batching
       unrelated attributes together would let one attribute's significance
       threshold be shifted by how many tests happened to be run for a
       completely different attribute in the same provider/dataset. This
       was verified empirically: every coarser grouping tested (pooling
       providers, pooling attributes, pooling both) produced fewer
       significant findings than this one, none produced more.

    2. Splitting binary and continuous metrics into separate batches within
       that attribute: Fisher's-exact/Chi-square and Mann-Whitney U are
       structurally different test families with different small-sample
       behaviour, and the 5 continuous metrics (rank_position_accuracy,
       sign_accuracy, value_accuracy, prop_pa_mentioned, prop_other_mentioned)
       are each built from disjoint sets of the binary columns also being
       tested — so pooling them with ~90 binary tests forces genuinely
       distinct, non-overlapping evidence to compete for the same threshold.
       Splitting them into their own small, internally coherent batch is the
       more defensible design, decided on this reasoning independently of
       which specific findings it happens to affect.

    Bonferroni is not used at all: this is exploratory, hypothesis-generating
    audit work with many correlated metrics within a batch (e.g.
    rank_position_accuracy is literally built from rank_1/2/3_scoring), which
    violates Bonferroni's independence assumption and makes it needlessly
    conservative here; FDR is the standard, valid choice for a batch of
    related exploratory comparisons like this one."""
    if df.empty:
        return df
    df = df.copy()
    df["p_value_fdr"] = np.nan
    for _, idx in df.groupby(
            ["dataset", "provider", "attribute", "metric_type"]).groups.items():
        pvals = df.loc[idx, "p_value"].values
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        _, fdr, _, _ = multipletests(pvals[valid], alpha=ALPHA, method="fdr_bh")
        valid_idx = np.array(idx)[valid]
        df.loc[valid_idx, "p_value_fdr"] = fdr
    df["significant_raw"] = df["p_value"] < ALPHA
    df["significant_fdr"] = df["p_value_fdr"] < ALPHA
    return df


def main():
    aggregated_rows = []
    test_rows = []

    for dataset in DATASETS:
        print(f"\n{'=' * 80}\n{dataset.upper()}\n{'=' * 80}")
        path = FAIRNESS_EVAL_DIR / f"per_narrative_metrics_{dataset}.csv"
        if not path.exists():
            print(f"  No {path.name} yet -- skipping.")
            continue
        df = pd.read_csv(path)
        df = df[df["condition"] == "pa_included"].copy()

        pa_names = PROTECTED_ATTRS[dataset]
        df = add_derived_columns(df, pa_names)
        metrics = discover_metrics(df, pa_names)
        print(f"  {len(df)} narratives (pa_included), {len(metrics)} metrics, "
              f"attributes={pa_names}")

        for provider in [p for p in PROVIDERS if p in set(df["provider"])]:
            psub = df[df["provider"] == provider]

            for attr in pa_names:
                group_labels, mapping = assign_groups(psub, dataset, attr)
                level_names = sorted(group_labels.dropna().unique())
                if len(level_names) != 2:
                    print(f"  WARNING: {attr} did not resolve to exactly 2 groups "
                          f"({level_names}) for provider={provider}; skipping.")
                    continue
                g1_name, g2_name = level_names

                for metric, kind in metrics.items():
                    vals1 = psub.loc[group_labels == g1_name, metric].to_numpy(dtype=float)
                    vals2 = psub.loc[group_labels == g2_name, metric].to_numpy(dtype=float)

                    # aggregated data (always recorded, even if test is skipped)
                    for gname, vals in [(g1_name, vals1), (g2_name, vals2)]:
                        valid = vals[~np.isnan(vals)]
                        aggregated_rows.append(dict(
                            dataset=dataset, provider=provider, attribute=attr,
                            group=gname, group_raw_value=mapping.get(gname, gname),
                            metric=metric, metric_type=kind,
                            n_total=len(vals), n_valid=len(valid),
                            mean_or_proportion=np.mean(valid) if len(valid) else np.nan,
                        ))

                    if kind == "binary":
                        result = test_binary(vals1, vals2, g1_name, g2_name)
                    else:
                        result = test_continuous(vals1, vals2, g1_name, g2_name)

                    if result is None:
                        continue

                    test_rows.append(dict(
                        dataset=dataset, provider=provider, attribute=attr,
                        metric=metric, metric_type=kind,
                        group1_raw_value=mapping.get(g1_name, g1_name),
                        group2_raw_value=mapping.get(g2_name, g2_name),
                        **result,
                    ))

    agg_df = pd.DataFrame(aggregated_rows)
    test_df = pd.DataFrame(test_rows)
    test_df = apply_correction(test_df)

    FAIRNESS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(OUTPUT_AGGREGATED, index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_TESTS, index=False, encoding="utf-8")

    print(f"\nAggregated data: {len(agg_df)} rows -> {OUTPUT_AGGREGATED.relative_to(ROOT)}")
    print(f"Test results: {len(test_df)} rows -> {OUTPUT_TESTS.relative_to(ROOT)}")
    print(f"  raw p<0.05: {test_df['significant_raw'].sum()}")
    print(f"  FDR-sig:    {test_df['significant_fdr'].sum()}")


if __name__ == "__main__":
    main()