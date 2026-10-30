#!/usr/bin/env python
"""
RQ3 — Disclosure Disparity
=============================
For a FIXED social group and a FIXED provider, does disclosing the protected
attribute in the prompt (pa_included) change a metric compared to withholding
it (pa_excluded)? Tested as a PAIRED comparison: the same instance's
narrative under both conditions.

This is a fundamentally different comparison structure from RQ1 (total
disparity: group vs. group, same condition) and RQ2 (indirect disparity:
group vs. group, PA withheld) — those compare DIFFERENT people (independent
samples). RQ3 compares the SAME person's two narratives against each other
(matched pairs), because the question here is not "do groups differ" but
"does disclosure change things, within a group". Provider comparisons are
explicitly out of scope, per the research team's request — this script never
compares providers to each other, only pa_included vs. pa_excluded within
one provider at a time.

GROUPING
--------
Identical to RQ1/RQ2 — see assign_groups() below, reused verbatim from
rq1_total_disparity.py for consistency (same 2-level splits for every
attribute, same medians/thresholds).

METRICS
-------
Identical set to RQ1/RQ2 (see discover_metrics()/add_derived_columns()),
including the <PA>_mentioned / <PA>_scoring metrics — these are the metrics
most directly about disclosure itself (does the model mention/correctly
state the PA when shown it vs. when it isn't there at all), so they are
deliberately included here, not treated as a special case.

STATISTICAL TESTS (matched pairs, not independent groups)
-------------------------------------------------------------
binary metric      -> McNemar's test (exact binomial if discordant pairs < 25,
                       chi-square approximation with continuity correction
                       otherwise — this selection is handled internally by
                       statsmodels' mcnemar() based on the discordant count)
continuous metric   -> Wilcoxon signed-rank test
Only complete pairs (metric present under BOTH conditions for that instance)
are used; an instance missing the metric in either condition is dropped for
that specific test. Minimum MIN_GROUP_N=5 complete pairs required.
Effect sizes:
  binary      -> McNemar odds ratio (b/c, the ratio of the two discordant-pair
                 counts; Haldane-Anscombe corrected when either is zero) —
                 labelled "odds_ratio", the same label used in RQ1/RQ2 for
                 Fisher/Chi-square, since it is the same underlying concept
                 computed from matched rather than independent data.
  continuous  -> matched rank-biserial correlation ((n_pos - n_neg) /
                 (n_pos + n_neg) over non-zero within-pair differences),
                 the paired analog of the rank-biserial correlation used
                 for Mann-Whitney U in RQ1/RQ2.

MULTIPLE-COMPARISON CORRECTION
--------------------------------
Benjamini-Hochberg FDR, applied WITHIN each
(dataset, provider, attribute, group, metric_type) batch — i.e. separately
per social group's disclosure-effect question, and separately for binary vs.
continuous metrics within that group. This extends the RQ1/RQ2 convention by
one further dimension (group), because under RQ3 a single attribute produces
TWO distinct sub-questions ("does disclosure matter for women?" and "does
disclosure matter for men?") that should not dilute each other's correction
threshold, for the same reason RQ1/RQ2 do not let unrelated attributes share
a batch. Bonferroni is not used, for the same reasons documented in
RQ1/RQ2's apply_correction().

OUTPUT
------
results/fairness_eval/rq3_disclosure_disparity_aggregated.csv
    dataset x provider x attribute x group x condition x metric, n and
    mean/proportion — the data being compared (note: two rows per group,
    one per condition, so differences can be read directly).
results/fairness_eval/rq3_disclosure_disparity_tests.csv
    dataset x provider x attribute x group x metric, paired test used,
    n_pairs, statistic, p-values (raw/FDR), significance flags, effect size.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
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

PROTECTED_ATTRS = {
    "credit": ["age", "sex", "foreign_worker"],
    "saudi": ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
    "law": ["gender", "race"],
}

OUTPUT_AGGREGATED = FAIRNESS_EVAL_DIR / "rq3_disclosure_disparity_aggregated.csv"
OUTPUT_TESTS = FAIRNESS_EVAL_DIR / "rq3_disclosure_disparity_tests.csv"


# ============================================================================
# GROUP ASSIGNMENT (identical to RQ1/RQ2, reused verbatim for consistency)
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
# METRIC DISCOVERY (identical to RQ1/RQ2, reused verbatim)
# ============================================================================

def discover_metrics(df, pa_names):
    metrics = {}
    for n in (1, 2, 3):
        metrics[f"rank_{n}_scoring"] = "binary"
        metrics[f"rank_{n}_sign_scoring"] = "binary"
        metrics[f"rank_{n}_value_scoring"] = "binary"
        metrics[f"rank_{n}_mentioned"] = "binary"
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
# PAIRED STATISTICAL TESTS
# ============================================================================

def mcnemar_odds_ratio(b, c):
    """McNemar odds ratio from the two discordant-pair counts, with
    Haldane-Anscombe correction when either is zero."""
    if b == 0 or c == 0:
        b, c = b + 0.5, c + 0.5
    return b / c


def test_binary_paired(vals_included, vals_excluded):
    """McNemar's test on matched pairs. vals_included/vals_excluded must
    already be aligned (same instance order) and NaN-free."""
    n = len(vals_included)
    if n < MIN_GROUP_N:
        return None

    table = np.zeros((2, 2))
    for a, b in zip(vals_included, vals_excluded):
        table[int(a), int(b)] += 1
    # table[1,0] = included=1,excluded=0 ("lost" under exclusion)
    # table[0,1] = included=0,excluded=1 ("gained" under exclusion)
    b_disc, c_disc = table[1, 0], table[0, 1]
    if b_disc + c_disc == 0:
        return None  # no discordant pairs at all, nothing to test

    result = mcnemar(table, exact=(b_disc + c_disc < 25), correction=True)
    eff = mcnemar_odds_ratio(b_disc, c_disc)
    return dict(test="McNemar", statistic=result.statistic, p_value=result.pvalue,
                effect_size=eff, effect_size_type="odds_ratio",
                n_pairs=n, mean_included=np.mean(vals_included),
                mean_excluded=np.mean(vals_excluded),
                n_discordant=int(b_disc + c_disc))


def test_continuous_paired(vals_included, vals_excluded):
    """Wilcoxon signed-rank test on matched pairs."""
    n = len(vals_included)
    if n < MIN_GROUP_N:
        return None

    diffs = vals_included - vals_excluded
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return None  # every pair identical, nothing to test

    try:
        stat, p = stats.wilcoxon(vals_included, vals_excluded, zero_method="wilcox")
    except ValueError:
        return None

    n_pos = (nonzero > 0).sum()
    n_neg = (nonzero < 0).sum()
    eff = (n_pos - n_neg) / (n_pos + n_neg)
    return dict(test="Wilcoxon signed-rank", statistic=stat, p_value=p,
                effect_size=eff, effect_size_type="matched_rank_biserial_r",
                n_pairs=n, mean_included=np.mean(vals_included),
                mean_excluded=np.mean(vals_excluded),
                n_discordant=int(len(nonzero)))


# ============================================================================
# CORRECTION
# ============================================================================

def apply_correction(df):
    if df.empty:
        return df
    df = df.copy()
    df["p_value_fdr"] = np.nan
    for _, idx in df.groupby(
            ["dataset", "provider", "attribute", "group", "metric_type"]).groups.items():
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


# ============================================================================
# MAIN
# ============================================================================

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

        pa_names = PROTECTED_ATTRS[dataset]
        df = add_derived_columns(df, pa_names)
        metrics = discover_metrics(df, pa_names)
        print(f"  {len(df)} narratives (both conditions), {len(metrics)} metrics, "
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

                for group_name in level_names:
                    gsub = psub[group_labels == group_name]

                    for metric, kind in metrics.items():
                        wide = gsub.pivot_table(index="instance_id", columns="condition",
                                                 values=metric)
                        if "pa_included" not in wide.columns or "pa_excluded" not in wide.columns:
                            continue
                        pair = wide[["pa_included", "pa_excluded"]].dropna()
                        vals_inc = pair["pa_included"].to_numpy(dtype=float)
                        vals_exc = pair["pa_excluded"].to_numpy(dtype=float)

                        # aggregated data (always recorded)
                        for cond_label, vals in [("pa_included", vals_inc), ("pa_excluded", vals_exc)]:
                            aggregated_rows.append(dict(
                                dataset=dataset, provider=provider, attribute=attr,
                                group=group_name, group_raw_value=mapping.get(group_name, group_name),
                                condition=cond_label, metric=metric, metric_type=kind,
                                n_pairs=len(vals),
                                mean_or_proportion=np.mean(vals) if len(vals) else np.nan,
                            ))

                        if kind == "binary":
                            result = test_binary_paired(vals_inc, vals_exc)
                        else:
                            result = test_continuous_paired(vals_inc, vals_exc)

                        if result is None:
                            continue

                        test_rows.append(dict(
                            dataset=dataset, provider=provider, attribute=attr,
                            group=group_name, group_raw_value=mapping.get(group_name, group_name),
                            metric=metric, metric_type=kind,
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