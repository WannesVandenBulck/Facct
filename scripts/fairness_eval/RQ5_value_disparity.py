#!/usr/bin/env python
"""
RQ5 — Value Disparity
======================
Holding EVERY displayed protected attribute constant, do narratives differ
because of the applicant's feature values alone?
(Fairness_Methodology_XAI_Narratives.pdf, Section 2: comparing (c)
counterfactual-PA-disclosed with (a) factual-PA-disclosed FOR THE SAME PA.)

Concretely: for a counterfactual job (e.g. credit "sex_female_to_male" --
originally-female instances shown as male), the counterfactual narratives
are compared against the FACTUAL (pa_included) narratives of instances that
ACTUALLY have the displayed value (real men), same provider. The PA label
the LLM sees is identical on both sides; only the feature vectors differ
(female-typical vs. male-typical). Different instances on each side ->
independent samples (unlike RQ3/RQ4, which are paired).

MATCHING (coarsened exact matching on the displayed PA profile)
------------------------------------------------------------------
Each narrative gets a stratum key built from ALL its displayed PAs:
  * the SWAPPED attribute: exact displayed value (e.g. "male" vs "male"),
    except continuous age (credit, student), which uses the RQ1 median
    split (counterfactual ages are fixed values -- 26/40 credit, 16/18
    student -- that almost no real applicant has exactly);
  * every OTHER PA: the RQ1 2-group split (see RQ1_total_disparity.py),
    so strata stay large enough to test.
Counterfactual and factual narratives are only compared WITHIN the same
stratum. Strata containing only one side are dropped (counted in the
matching summary output).

STATISTICAL TESTS (stratified versions of the RQ1/RQ2 tests)
-------------------------------------------------------------
binary metric     -> Cochran-Mantel-Haenszel test (stratified chi-square),
                     no continuity correction (as RQ1 chi-square); if any
                     informative stratum has an expected cell count < 5,
                     the EXACT stratified test is used instead (stratified
                     Fisher: exact conditional distribution of the
                     counterfactual "1" count, i.e. the convolution of the
                     per-stratum hypergeometrics) -- mirroring RQ1's
                     Fisher-when-expected<5 rule.
                     Effect size: Mantel-Haenszel common odds ratio
                     (counterfactual vs. factual), 0.5 added to strata
                     with a zero cell (as RQ4's Haldane-Anscombe).
continuous metric -> van Elteren test (stratified Mann-Whitney U), normal
                     approximation with tie correction, weights 1/(N_h+1).
                     Effect size: rank-biserial correlation per stratum,
                     averaged with weights n1*n2/N_h. Positive = higher in
                     the counterfactual narratives.
With a single stratum these reduce to chi-square and Mann-Whitney U.
Minimum MIN_GROUP_N matched narratives on EACH side.

MULTIPLE-COMPARISON CORRECTION
--------------------------------
Benjamini-Hochberg FDR within each (dataset, provider, cf_label,
metric_type) batch -- same granularity as RQ4.

PROVIDERS
---------
All narrative providers are included: factual and counterfactual narratives are
generated in the same batch run from the same frozen prompts. Providers without
narratives yet are skipped.

OUTPUT
------
results/fairness_eval/rq5_value_disparity_matching.csv
    One row per dataset x provider x cf_label: narratives available and
    kept after matching, number of strata.
results/fairness_eval/rq5_value_disparity_aggregated.csv
    One row per dataset x provider x cf_label x side x metric (matched
    narratives only), with n and mean/proportion.
results/fairness_eval/rq5_value_disparity_tests.csv
    One row per dataset x provider x cf_label x metric: test, sizes,
    statistic, raw and FDR p-values, significance flags, effect size.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent.parent
FAIRNESS_EVAL_DIR = ROOT / "results" / "fairness_eval"

DATASETS = ["credit", "saudi", "student", "law"]
PROVIDERS = ["openai", "mistral", "claude", "deepseek"]  # providers without narratives (yet) are skipped
MIN_GROUP_N = 5
ALPHA = 0.05

PROTECTED_ATTRS = {
    "credit": ["age", "sex", "foreign_worker"],
    "saudi": ["Gender", "Age", "Health_Issues"],
    "student": ["sex", "age", "health"],
    "law": ["gender", "race"],
}

# Continuous PAs: always binned (median split, as in RQ1), even when swapped.
CONTINUOUS_PAS = {("credit", "age"), ("student", "age")}

OUTPUT_MATCHING = FAIRNESS_EVAL_DIR / "rq5_value_disparity_matching.csv"
OUTPUT_AGGREGATED = FAIRNESS_EVAL_DIR / "rq5_value_disparity_aggregated.csv"
OUTPUT_TESTS = FAIRNESS_EVAL_DIR / "rq5_value_disparity_tests.csv"


# ============================================================================
# METRICS (identical to RQ1/RQ3/RQ4)
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
# DISPLAYED PA PROFILE + STRATUM KEYS
# ============================================================================

def add_displayed_pas(df, pa_names):
    """<pa>_shown = value shown in the prompt: the real value, except the
    swapped attribute of a counterfactual row, which shows the override."""
    df = df.copy()
    for pa in pa_names:
        df[f"{pa}_shown"] = df[f"{pa}_GT"].astype(float)
    cf = df["condition"] == "counterfactual_pa"
    for idx, row in df[cf].iterrows():
        df.at[idx, f"{row['cf_attr']}_shown"] = float(row["cf_override_value"])
    return df


def rq1_group(dataset, pa, values, medians):
    """RQ1 2-group split (see RQ1_total_disparity.assign_groups)."""
    v = values.astype(float)
    if (dataset, pa) in CONTINUOUS_PAS:
        return np.where(v < medians[pa], "below_median", "at_or_above_median")
    if (dataset, pa) == ("saudi", "Age"):
        return np.where(v == 0, "21-30", "31+")
    if (dataset, pa) == ("student", "health"):
        return np.where(v <= 3, "very_bad_bad_fair", "good_very_good")
    if (dataset, pa) == ("law", "race"):
        return np.where(v == 0, "white", "non_white")
    return v.map(lambda x: f"{x:g}").to_numpy()  # binary PAs: already 2 groups


def stratum_keys(df, dataset, pa_names, swapped_attr, medians):
    parts = []
    for pa in pa_names:
        shown = df[f"{pa}_shown"]
        if pa == swapped_attr and (dataset, pa) not in CONTINUOUS_PAS:
            parts.append(pa + "=" + shown.map(lambda x: f"{x:g}"))  # exact
        else:
            parts.append(pa + "=" + pd.Series(
                rq1_group(dataset, pa, shown, medians), index=df.index))
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


# ============================================================================
# STRATIFIED TESTS
# ============================================================================

def exact_stratified_p(tables):
    """Exact conditional test of no association across strata: the
    counterfactual "1" count per stratum is hypergeometric under H0 given
    the margins; the total is their convolution. Two-sided p-value =
    P(|T - E[T]| >= |T_obs - E[T]|)."""
    dist = np.array([1.0])
    lo_total = 0
    t_obs, e_t = 0.0, 0.0
    for t in tables:
        n1 = int(t[0].sum())          # counterfactual narratives
        m1 = int(t[:, 0].sum())       # total "1"s
        N = int(t.sum())
        lo, hi = max(0, n1 + m1 - N), min(n1, m1)
        pmf = stats.hypergeom.pmf(np.arange(lo, hi + 1), N, m1, n1)
        dist = np.convolve(dist, pmf)
        lo_total += lo
        t_obs += t[0, 0]
        e_t += n1 * m1 / N
    support = np.arange(lo_total, lo_total + len(dist))
    dev = np.abs(support - e_t)
    return min(1.0, dist[dev >= abs(t_obs - e_t) - 1e-9].sum())


def test_binary_stratified(cf_vals, fac_vals, cf_strata, fac_strata):
    """Cochran-Mantel-Haenszel over strata (exact stratified test when
    sparse); rows = cf/factual, cols = 1/0."""
    if len(cf_vals) < MIN_GROUP_N or len(fac_vals) < MIN_GROUP_N:
        return None
    tables = []
    for s in np.unique(cf_strata):
        a = cf_vals[cf_strata == s]
        b = fac_vals[fac_strata == s]
        t = np.array([[a.sum(), len(a) - a.sum()],
                      [b.sum(), len(b) - b.sum()]], dtype=float)
        if t.sum(axis=1).min() > 0 and t.sum(axis=0).min() > 0:
            tables.append(t)  # strata with a zero margin carry no information
    if not tables:
        return None

    sparse = any((np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()).min() < 5
                 for t in tables)
    if sparse:
        test, stat, p = "Exact stratified (Fisher)", np.nan, exact_stratified_p(tables)
    else:
        res = StratifiedTable(tables).test_null_odds(correction=False)
        test, stat, p = "Cochran-Mantel-Haenszel", res.statistic, res.pvalue
    try:
        eff = StratifiedTable(tables, shift_zeros=True).oddsratio_pooled
    except ZeroDivisionError:
        eff = np.nan
    return dict(test=test, statistic=stat, p_value=p, effect_size=eff,
                effect_size_type="MH_common_odds_ratio",
                n_informative_strata=len(tables))


def test_continuous_stratified(cf_vals, fac_vals, cf_strata, fac_strata):
    """van Elteren test: stratified Wilcoxon rank-sum, weights 1/(N_h+1)."""
    if len(cf_vals) < MIN_GROUP_N or len(fac_vals) < MIN_GROUP_N:
        return None
    num, var, rb_sum, rb_w, n_inf = 0.0, 0.0, 0.0, 0.0, 0
    for s in np.unique(cf_strata):
        a = cf_vals[cf_strata == s]
        b = fac_vals[fac_strata == s]
        n1, n2 = len(a), len(b)
        N = n1 + n2
        if n1 == 0 or n2 == 0:
            continue
        ranks = stats.rankdata(np.concatenate([a, b]))
        W = ranks[:n1].sum()
        _, counts = np.unique(ranks, return_counts=True)
        tie = (counts ** 3 - counts).sum()
        v = n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1)))
        if v <= 0:
            continue  # all values identical in this stratum
        w = 1.0 / (N + 1)
        num += w * (W - n1 * (N + 1) / 2.0)
        var += w ** 2 * v
        U = W - n1 * (n1 + 1) / 2.0
        rb_sum += (2 * U / (n1 * n2) - 1) * (n1 * n2 / N)
        rb_w += n1 * n2 / N
        n_inf += 1
    if var == 0:
        return None
    z = num / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return dict(test="van Elteren", statistic=z, p_value=p,
                effect_size=rb_sum / rb_w,
                effect_size_type="stratified_rank_biserial_r",
                n_informative_strata=n_inf)


def apply_correction(df):
    if df.empty:
        return df
    df = df.copy()
    df["p_value_fdr"] = np.nan
    for _, idx in df.groupby(["dataset", "provider", "cf_label", "metric_type"]).groups.items():
        pvals = df.loc[idx, "p_value"].values
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        _, fdr, _, _ = multipletests(pvals[valid], alpha=ALPHA, method="fdr_bh")
        df.loc[np.array(idx)[valid], "p_value_fdr"] = fdr
    df["significant_raw"] = df["p_value"] < ALPHA
    df["significant_fdr"] = df["p_value_fdr"] < ALPHA
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    matching_rows, aggregated_rows, test_rows = [], [], []

    for dataset in DATASETS:
        print(f"\n{'=' * 80}\n{dataset.upper()}\n{'=' * 80}")
        path = FAIRNESS_EVAL_DIR / f"per_narrative_metrics_{dataset}.csv"
        if not path.exists():
            print(f"  No {path.name} yet -- skipping.")
            continue
        df = pd.read_csv(path, low_memory=False)
        if "cf_label" not in df.columns or df["cf_label"].notna().sum() == 0:
            print("  No counterfactual_pa rows -- skipping.")
            continue

        pa_names = PROTECTED_ATTRS[dataset]
        df = add_derived_columns(df, pa_names)
        df = add_displayed_pas(df, pa_names)
        metrics = discover_metrics(df, pa_names)

        # medians on unique instances' REAL values (identical to RQ1)
        uniq = df.drop_duplicates("instance_id")
        medians = {pa: uniq[f"{pa}_GT"].astype(float).median()
                   for pa in pa_names if (dataset, pa) in CONTINUOUS_PAS}
        cf_labels = sorted(df.loc[df["condition"] == "counterfactual_pa", "cf_label"].dropna().unique())

        for provider in [p for p in PROVIDERS if p in set(df["provider"])]:
            psub = df[df["provider"] == provider]
            factual_all = psub[psub["condition"] == "pa_included"]

            for cf_label in cf_labels:
                cf_all = psub[(psub["condition"] == "counterfactual_pa") &
                              (psub["cf_label"] == cf_label)]
                if cf_all.empty:
                    continue
                attr = cf_all["cf_attr"].iloc[0]
                override = cf_all["cf_override_value"].iloc[0]

                cf_key = stratum_keys(cf_all, dataset, pa_names, attr, medians)
                fac_key = stratum_keys(factual_all, dataset, pa_names, attr, medians)
                shared = set(cf_key) & set(fac_key)
                cf = cf_all[cf_key.isin(shared)]
                fac = factual_all[fac_key.isin(shared)]
                cf_s = cf_key[cf_key.isin(shared)].to_numpy()
                fac_s = fac_key[fac_key.isin(shared)].to_numpy()

                matching_rows.append(dict(
                    dataset=dataset, provider=provider, cf_label=cf_label,
                    attribute=attr, override_value=override,
                    n_cf_total=len(cf_all), n_cf_matched=len(cf),
                    n_factual_matched=len(fac), n_strata=len(shared)))
                print(f"  {provider:8}{cf_label:28} cf {len(cf):4}/{len(cf_all):<4} "
                      f"factual {len(fac):4}  strata {len(shared)}")

                for metric, kind in metrics.items():
                    if metric not in df.columns:
                        continue
                    m_cf = cf[metric].notna().to_numpy()
                    m_fac = fac[metric].notna().to_numpy()
                    v_cf = cf[metric].to_numpy(dtype=float)[m_cf]
                    v_fac = fac[metric].to_numpy(dtype=float)[m_fac]
                    s_cf, s_fac = cf_s[m_cf], fac_s[m_fac]

                    for side, vals in [("counterfactual", v_cf), ("factual", v_fac)]:
                        aggregated_rows.append(dict(
                            dataset=dataset, provider=provider, cf_label=cf_label,
                            attribute=attr, override_value=override, condition=side,
                            metric=metric, metric_type=kind, n=len(vals),
                            mean_or_proportion=np.mean(vals) if len(vals) else np.nan))

                    if kind == "binary":
                        res = test_binary_stratified(v_cf, v_fac, s_cf, s_fac)
                    else:
                        res = test_continuous_stratified(v_cf, v_fac, s_cf, s_fac)
                    if res is None:
                        continue
                    test_rows.append(dict(
                        dataset=dataset, provider=provider, cf_label=cf_label,
                        attribute=attr, override_value=override,
                        metric=metric, metric_type=kind,
                        n_counterfactual=len(v_cf), n_factual=len(v_fac),
                        mean_counterfactual=np.mean(v_cf), mean_factual=np.mean(v_fac),
                        **res))

    match_df = pd.DataFrame(matching_rows)
    agg_df = pd.DataFrame(aggregated_rows)
    test_df = apply_correction(pd.DataFrame(test_rows))

    FAIRNESS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(OUTPUT_MATCHING, index=False, encoding="utf-8")
    agg_df.to_csv(OUTPUT_AGGREGATED, index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_TESTS, index=False, encoding="utf-8")

    print(f"\nMatching summary -> {OUTPUT_MATCHING.relative_to(ROOT)}")
    print(f"Aggregated: {len(agg_df)} rows -> {OUTPUT_AGGREGATED.relative_to(ROOT)}")
    print(f"Tests: {len(test_df)} rows -> {OUTPUT_TESTS.relative_to(ROOT)}")
    if not test_df.empty:
        print(f"  raw p<0.05: {test_df['significant_raw'].sum()}")
        print(f"  FDR-sig:    {test_df['significant_fdr'].sum()}")


if __name__ == "__main__":
    main()