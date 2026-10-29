#!/usr/bin/env python
"""
Provider Disparity
====================
Supplementary analysis, not one of the paper's core disparity RQs (total /
indirect / disclosure / value / direct) -- kept separate and unnumbered so
it isn't confused with RQ4 (direct disparity) or RQ5 (value disparity).

Do the LLM providers used to generate narratives (openai / mistral / claude / deepseek)
differ from each other in feature-coverage / faithfulness metrics? Tested
both OVERALL (pooled across all instances of a dataset/condition) and
WITHIN each social group of every protected attribute (e.g. "is openai
more sign-accurate than grok among women, even if not on average?").

INPUT
-----
results/fairness_eval/per_narrative_metrics_<dataset>.csv (one row per
narrative = instance x provider x condition), as produced by
generate_per_narrative_metrics.py.

PAIRING (why these are PAIRED, not independent-group, tests)
--------------------------------------------------------------
Every instance is narrated by all three providers under the same
condition, so a given instance_id contributes one row per provider that
share the exact same underlying case (same applicant, same GT features).
Provider differences are therefore tested as PAIRED comparisons matched on
instance_id (within dataset x condition), exactly like the condition-effect
tests in the old RQ4 script matched pa_included vs pa_excluded by
instance_id. This is the correct design (as opposed to treating the three
providers as independent samples, which would ignore that they are scored
on identical cases and understate statistical power / misstate the
variance).

Each unordered pair of providers is tested separately:
  every pair of the providers present (with all four: 6 pairs).

GROUPING (identical 2-group split per dataset/attribute as RQ1/RQ2)
--------------------------------------------------------------------
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
  student   health           very bad/bad/fair     good/very good
  law       gender           female                male
  law       race             white                 non-white

Group assignment uses the ground-truth attribute value (<attr>_GT), which
is provider-invariant (same applicant), so an instance falls in the same
group regardless of which provider narrated it.

METRICS (identical set to RQ1/RQ2 — see those scripts for full rationale)
---------------------------------------------------------------------------
Faithfulness (binary, NaN where not applicable):
  rank_{1,2,3}_scoring, rank_{1,2,3}_sign_scoring, rank_{1,2,3}_value_scoring,
  predicted_probability_scoring, <PA>_scoring, other_<feature>_value_scoring
Coverage (binary):
  <PA>_mentioned, other_<feature>_mentioned, rank_{1,2,3}_mentioned (derived)
Continuous aggregates (mean of the available binary components per row):
  rank_position_accuracy, sign_accuracy, value_accuracy,
  prop_pa_mentioned, prop_other_mentioned

Tested separately per CONDITION (pa_included / pa_excluded /
pa_counterfactual), since the conditions are different tasks for the model
(PA shown, hidden, or swapped) and mixing them would blur a provider's
behaviour under one condition with its behaviour under another.

COUNTERFACTUAL CONDITION (pa_counterfactual)
---------------------------------------------
Under the counterfactual condition every instance appears once PER
counterfactual job (one per swapped protected attribute, e.g. a credit
applicant three times: sex, age and foreign_worker swapped). Pooling these
narratives directly would count the same instance several times and
violate the independence of the pairs. Therefore:
  * overall and pooled scopes: per instance and provider, the CONTINUOUS
    metrics are averaged over that instance's counterfactual narratives,
    so every instance again contributes exactly one value per provider and
    is paired on instance_id as in the other conditions. Binary indicators
    are not tested at these scopes (their average is no longer binary).
  * cf_job scope (replaces the group scope for this condition): every
    counterfactual job (e.g. credit "sex_female_to_male") is tested
    separately on ALL metrics (binary and continuous). Within a job each
    instance appears exactly once per provider, and all narratives in the
    job display the same swapped value, so the job itself is the
    displayed-group comparison (e.g. "is Grok more sign-accurate than
    OpenAI for originally-female applicants shown as male?").
    attribute = cf_label, group = "<attr> shown as <value>".
FDR batches are formed exactly as for the other conditions; since condition
is part of every batch key, adding this condition does not change any
pa_included / pa_excluded result.

STATISTICAL TESTS
------------------
Every comparison is a matched pair of the SAME instances scored by two
providers:
  binary metric      -> McNemar's test (exact if #discordant pairs < 25,
                         chi-square with continuity correction otherwise),
                         on the 2x2 table of (provider1, provider2) outcomes
                         restricted to instances where both are non-NaN.
                         Effect size: paired odds ratio
                         ((n where only provider1=1) / (n where only
                         provider2=1), 0.5-smoothed) — >1 means provider1
                         wins more of the discordant cases.
  continuous metric  -> Wilcoxon signed-rank test on the paired differences.
                         Effect size: paired Cohen's d (mean paired
                         difference / std of paired differences).
Pairs with fewer than MIN_GROUP_N matched, non-NaN instances, or with zero
discordant/non-zero differences, are skipped (no evidence to test).
Multiple-comparison correction: Benjamini-Hochberg FDR, applied WITHIN each
(dataset, condition, scope, metric_type) batch, where scope is "overall" or
the specific protected attribute being tested in the group-level scope —
i.e. the three provider-pair tests for a given metric compete for the same
correction batch as every other metric of the same type (binary/continuous)
in that scope, mirroring the batching rationale in RQ1_total_disparity.py.

POOLED SCOPE (all 4 datasets combined)
-----------------------------------------
In addition to per-dataset "overall"/"group" scopes, a "pooled" scope
answers e.g. "is openai generally more faithful than grok across all 4
datasets?" — pairing is on (dataset, instance_id) jointly (plain
instance_id is not unique across datasets), still per condition. Only
metrics with an identical, dataset-independent meaning are poolable this
way: rank_{1,2,3}_scoring/_sign_scoring/_value_scoring/_mentioned,
predicted_probability_scoring, and the 5 continuous aggregates
(rank_position_accuracy, sign_accuracy, value_accuracy, prop_pa_mentioned,
prop_other_mentioned). <PA>_scoring/_mentioned and other_<feature>_* are
NOT pooled, since the protected-attribute and "other feature" names differ
per dataset (e.g. "sex" only exists for credit/student/law, "race" only for
law) — comparing them across datasets would silently mix unrelated
attributes. There is no pooled x group-level scope for the same reason:
protected attributes are not standardized across datasets, so a
cross-dataset "social group" (e.g. combining credit's `sex` with law's
`race`) would not be a coherent group.

OUTPUT
------
results/fairness_eval/provider_disparity_aggregated.csv
    One row per dataset x condition x scope_type x attribute x group x
    provider x metric, with n and mean/proportion. Pooled rows use
    dataset == "ALL".
results/fairness_eval/provider_disparity_tests.csv
    One row per dataset x condition x scope_type x attribute x group x
    metric x provider pair, with the test used, matched-pair counts,
    statistic, raw and FDR-corrected p-values, significance flags, and
    effect size. scope_type == "overall"/"pooled" rows have
    attribute/group == NaN; scope_type == "group" rows report the
    per-social-group comparison (within a single dataset).
"""

import warnings
from itertools import combinations
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
PROVIDERS = ["openai", "mistral", "claude", "deepseek"]  # providers without narratives (yet) are skipped
CONDITIONS = ["pa_included", "pa_excluded"]
CF_CONDITION_RAW = "counterfactual_pa"   # condition value in the metrics CSV
CF_CONDITION = "pa_counterfactual"       # label used in the output files
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

OUTPUT_AGGREGATED = FAIRNESS_EVAL_DIR / "provider_disparity_aggregated.csv"
OUTPUT_TESTS = FAIRNESS_EVAL_DIR / "provider_disparity_tests.csv"

# Metrics with identical meaning/column-name across every dataset, i.e. not
# tied to a dataset-specific protected attribute or "other feature" name
# (<PA>_scoring, other_<feature>_mentioned, etc. are NOT poolable since the
# attribute/feature names differ per dataset). Used for the "pooled" scope,
# which compares providers across ALL datasets combined.
POOLED_METRICS = {}
for _n in (1, 2, 3):
    POOLED_METRICS[f"rank_{_n}_scoring"] = "binary"
    POOLED_METRICS[f"rank_{_n}_sign_scoring"] = "binary"
    POOLED_METRICS[f"rank_{_n}_value_scoring"] = "binary"
    POOLED_METRICS[f"rank_{_n}_mentioned"] = "binary"
POOLED_METRICS["predicted_probability_scoring"] = "binary"
POOLED_METRICS["rank_position_accuracy"] = "continuous"
POOLED_METRICS["sign_accuracy"] = "continuous"
POOLED_METRICS["value_accuracy"] = "continuous"
POOLED_METRICS["prop_pa_mentioned"] = "continuous"
POOLED_METRICS["prop_other_mentioned"] = "continuous"




# ============================================================================
# GROUP ASSIGNMENT (identical to RQ1_total_disparity.py / RQ2_indirect_disparity.py)
# ============================================================================

def assign_groups(df, dataset, attr):
    """Return a Series of 2-level group labels for this attribute, plus a
    dict describing which raw values map to which label (for reporting)."""
    col = f"{attr}_GT"

    if (dataset, attr) == ("credit", "age"):
        # df is always pre-deduplicated by instance_id by the caller.
        median = df[col].median()
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
        median = df[col].median()
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
# METRIC DISCOVERY (identical to RQ1_total_disparity.py / RQ2_indirect_disparity.py)
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
# PAIRED STATISTICAL TESTS
# ============================================================================

def test_paired_binary(vals1, vals2, name1, name2):
    """McNemar's test on the 2x2 table of (provider1, provider2) binary
    outcomes for the same matched instances. Effect size = paired odds
    ratio: (# instances where only provider1 scored 1) / (# instances where
    only provider2 scored 1), 0.5-smoothed; >1 favors provider1."""
    mask = ~(np.isnan(vals1) | np.isnan(vals2))
    a, b = vals1[mask], vals2[mask]
    n = len(a)
    if n < MIN_GROUP_N:
        return None

    both1 = int(np.sum((a == 1) & (b == 1)))
    both0 = int(np.sum((a == 0) & (b == 0)))
    only1 = int(np.sum((a == 1) & (b == 0)))
    only2 = int(np.sum((a == 0) & (b == 1)))
    n_discordant = only1 + only2
    if n_discordant == 0:
        return None  # perfectly concordant -> no evidence of a difference

    table = [[both1, only1], [only2, both0]]
    result = mcnemar(table, exact=(n_discordant < 25), correction=True)
    eff = (only1 + 0.5) / (only2 + 0.5)

    return dict(test="McNemar", statistic=result.statistic, p_value=result.pvalue,
                effect_size=eff, effect_size_type="paired_odds_ratio",
                n_pairs=n, n_discordant=n_discordant, n_only1=only1, n_only2=only2,
                mean1=np.mean(a), mean2=np.mean(b), group1=name1, group2=name2)


def test_paired_continuous(vals1, vals2, name1, name2):
    """Wilcoxon signed-rank test on the paired differences. Effect size =
    paired Cohen's d (mean paired difference / std of paired differences)."""
    mask = ~(np.isnan(vals1) | np.isnan(vals2))
    a, b = vals1[mask], vals2[mask]
    if len(a) < MIN_GROUP_N:
        return None
    diff = a - b
    if not np.any(diff != 0):
        return None
    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        return None
    diff_std = diff.std(ddof=1)
    eff = diff.mean() / diff_std if diff_std > 0 else np.nan
    return dict(test="Wilcoxon signed-rank", statistic=stat, p_value=p,
                effect_size=eff, effect_size_type="paired_cohens_d",
                n_pairs=len(a), mean1=np.mean(a), mean2=np.mean(b),
                mean_diff=diff.mean(), group1=name1, group2=name2)


def run_paired_tests(sub_df, metrics, dataset, condition, scope_type, attribute, group, group_raw_value,
                      id_col="instance_id"):
    """Run every metric x provider-pair paired test on the instances in
    sub_df (already restricted to the relevant condition and, for group
    scope, the relevant social group). Returns (test_rows, aggregated_rows).
    id_col is the column pairs are matched on: "instance_id" within a single
    dataset, or a dataset-qualified composite key for the pooled scope
    (plain instance_id is not unique across datasets)."""
    test_rows = []
    aggregated_rows = []

    provider_frames = {p: sub_df[sub_df["provider"] == p].set_index(id_col) for p in PROVIDERS
                       if (sub_df["provider"] == p).any()}

    for provider, pframe in provider_frames.items():
        for metric_name, metric_type in metrics.items():
            vals = pframe[metric_name].to_numpy(dtype=float)
            valid = vals[~np.isnan(vals)]
            aggregated_rows.append(dict(
                dataset=dataset, condition=condition, scope_type=scope_type,
                attribute=attribute, group=group, group_raw_value=group_raw_value,
                provider=provider, metric=metric_name, metric_type=metric_type,
                n_total=len(vals), n_valid=len(valid),
                mean=np.mean(valid) if len(valid) else np.nan,
            ))

    for p1, p2 in combinations(provider_frames, 2):
        d1, d2 = provider_frames[p1], provider_frames[p2]
        common_ids = d1.index.intersection(d2.index)
        if len(common_ids) < MIN_GROUP_N:
            continue
        d1c, d2c = d1.loc[common_ids], d2.loc[common_ids]

        for metric_name, metric_type in metrics.items():
            v1 = d1c[metric_name].to_numpy(dtype=float)
            v2 = d2c[metric_name].to_numpy(dtype=float)
            if metric_type == "binary":
                result = test_paired_binary(v1, v2, p1, p2)
            else:
                result = test_paired_continuous(v1, v2, p1, p2)
            if result is None:
                continue
            test_rows.append(dict(
                dataset=dataset, condition=condition, scope_type=scope_type,
                attribute=attribute, group=group, group_raw_value=group_raw_value,
                metric=metric_name, metric_type=metric_type, **result,
            ))

    return test_rows, aggregated_rows


# ============================================================================
# MULTIPLE-COMPARISON CORRECTION
# ============================================================================

def apply_correction(df):
    """Benjamini-Hochberg FDR correction, applied WITHIN each (dataset,
    condition, scope_type, attribute, metric_type) batch — i.e. the 3
    provider-pair tests for every metric of a given type (binary/continuous)
    compete for the same threshold within a given scope (overall, or one
    specific protected attribute), mirroring the batching rationale in
    RQ1_total_disparity.py (each attribute is a distinct hypothesis family;
    binary and continuous metrics are structurally different test families
    and the continuous ones are built from the binary ones being tested)."""
    if df.empty:
        return df
    df = df.copy()
    df["_batch_attr"] = df["attribute"].fillna("__overall__")
    df["p_value_fdr"] = np.nan
    for _, idx in df.groupby(
            ["dataset", "condition", "scope_type", "_batch_attr", "metric_type"]).groups.items():
        pvals = df.loc[idx, "p_value"].values
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        _, fdr, _, _ = multipletests(pvals[valid], alpha=ALPHA, method="fdr_bh")
        valid_idx = np.array(idx)[valid]
        df.loc[valid_idx, "p_value_fdr"] = fdr
    df["significant_raw"] = df["p_value"] < ALPHA
    df["significant_fdr"] = df["p_value_fdr"] < ALPHA
    return df.drop(columns=["_batch_attr"])


# ============================================================================
# MAIN
# ============================================================================

def main():
    all_test_rows = []
    all_aggregated_rows = []
    dataset_dfs = {}  # dataset -> derived-columns df, reused for the pooled scope below
    cf_averaged = {}  # dataset -> per-instance averaged counterfactual metrics

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
        dataset_dfs[dataset] = df

        for condition in CONDITIONS:
            csub = df[df["condition"] == condition].copy()
            print(f"  [{condition}] {len(csub)} narratives")

            # ---- overall scope: all instances pooled ----
            test_rows, agg_rows = run_paired_tests(
                csub, metrics, dataset, condition, "overall",
                attribute=np.nan, group=np.nan, group_raw_value=np.nan,
            )
            all_test_rows.extend(test_rows)
            all_aggregated_rows.extend(agg_rows)

            # ---- group scope: restricted to one social group at a time ----
            # GT-based group membership is provider-invariant, so it is
            # computed once per instance (dedup) rather than per provider.
            gt_lookup = csub.drop_duplicates("instance_id").set_index("instance_id")

            for attr in pa_names:
                group_labels, mapping = assign_groups(gt_lookup, dataset, attr)
                level_names = sorted(group_labels.dropna().unique())
                if len(level_names) != 2:
                    print(f"  WARNING: {attr} did not resolve to exactly 2 groups "
                          f"({level_names}) for condition={condition}; skipping.")
                    continue

                for gname in level_names:
                    group_instance_ids = group_labels[group_labels == gname].index
                    gsub = csub[csub["instance_id"].isin(group_instance_ids)]

                    test_rows, agg_rows = run_paired_tests(
                        gsub, metrics, dataset, condition, "group",
                        attribute=attr, group=gname, group_raw_value=mapping.get(gname, gname),
                    )
                    all_test_rows.extend(test_rows)
                    all_aggregated_rows.extend(agg_rows)

        # ---- counterfactual condition ----
        cf = df[df["condition"] == CF_CONDITION_RAW].copy()
        if not cf.empty:
            print(f"  [{CF_CONDITION}] {len(cf)} narratives, "
                  f"{cf['cf_label'].nunique()} counterfactual jobs")
            cont_metrics = {m: t for m, t in metrics.items() if t == "continuous"}

            # overall scope: average each instance's counterfactual narratives
            # per provider, so every instance contributes one value per provider
            cf_avg = (cf.groupby(["provider", "instance_id"])[list(cont_metrics)]
                        .mean().reset_index())
            cf_avg["condition"] = CF_CONDITION
            test_rows, agg_rows = run_paired_tests(
                cf_avg, cont_metrics, dataset, CF_CONDITION, "overall",
                attribute=np.nan, group=np.nan, group_raw_value=np.nan,
            )
            all_test_rows.extend(test_rows)
            all_aggregated_rows.extend(agg_rows)
            cf_avg["dataset"] = dataset
            cf_averaged[dataset] = cf_avg

            # cf_job scope: each counterfactual job separately, all metrics
            for job, jsub in cf.groupby("cf_label"):
                attr_name = jsub["cf_attr"].iloc[0]
                shown = jsub["cf_override_value"].iloc[0]
                test_rows, agg_rows = run_paired_tests(
                    jsub, metrics, dataset, CF_CONDITION, "cf_job",
                    attribute=job, group=f"{attr_name} shown as {shown}",
                    group_raw_value=shown,
                )
                all_test_rows.extend(test_rows)
                all_aggregated_rows.extend(agg_rows)

    # ---- pooled scope: providers compared across ALL datasets combined ----
    print(f"\n{'=' * 80}\nPOOLED (ALL DATASETS)\n{'=' * 80}")
    keep_cols = ["dataset", "provider", "condition", "instance_id"] + list(POOLED_METRICS)
    pooled_df = pd.concat([df_[keep_cols] for df_ in dataset_dfs.values()], ignore_index=True)  # datasets present only
    # instance_id resets per dataset, so pair on a dataset-qualified key.
    pooled_df["pair_key"] = pooled_df["dataset"] + "_" + pooled_df["instance_id"].astype(str)

    for condition in CONDITIONS:
        csub = pooled_df[pooled_df["condition"] == condition]
        print(f"  [{condition}] {len(csub)} narratives across {len(dataset_dfs)} datasets")
        test_rows, agg_rows = run_paired_tests(
            csub, POOLED_METRICS, "ALL", condition, "pooled",
            attribute=np.nan, group=np.nan, group_raw_value=np.nan, id_col="pair_key",
        )
        all_test_rows.extend(test_rows)
        all_aggregated_rows.extend(agg_rows)

    # pooled scope for the counterfactual condition (continuous metrics only)
    if cf_averaged:
        pooled_cont = {m: t for m, t in POOLED_METRICS.items() if t == "continuous"}
        cf_pooled = pd.concat(cf_averaged.values(), ignore_index=True)
        cf_pooled["pair_key"] = cf_pooled["dataset"] + "_" + cf_pooled["instance_id"].astype(str)
        print(f"  [{CF_CONDITION}] {len(cf_pooled)} instance-level averages across "
              f"{len(cf_averaged)} datasets")
        test_rows, agg_rows = run_paired_tests(
            cf_pooled, pooled_cont, "ALL", CF_CONDITION, "pooled",
            attribute=np.nan, group=np.nan, group_raw_value=np.nan, id_col="pair_key",
        )
        all_test_rows.extend(test_rows)
        all_aggregated_rows.extend(agg_rows)

    agg_df = pd.DataFrame(all_aggregated_rows)
    test_df = pd.DataFrame(all_test_rows)
    test_df = apply_correction(test_df)

    FAIRNESS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(OUTPUT_AGGREGATED, index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_TESTS, index=False, encoding="utf-8")

    print(f"\nAggregated data: {len(agg_df)} rows -> {OUTPUT_AGGREGATED.relative_to(ROOT)}")
    print(f"Test results: {len(test_df)} rows -> {OUTPUT_TESTS.relative_to(ROOT)}")
    if not test_df.empty:
        n_sig = int(test_df["significant_fdr"].sum())
        print(f"Significant (FDR) provider-pair differences: {n_sig} / {len(test_df)}")


if __name__ == "__main__":
    main()