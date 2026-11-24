from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _portfolio_stats(returns: pd.DataFrame) -> Dict:
    port = returns.mean(axis=1)
    vol = port.std() * np.sqrt(252)
    mean = port.mean() * 252
    corr = returns.corr().to_numpy()
    n = corr.shape[0]
    avg_corr = (corr.sum() - n) / (n * (n - 1))
    return {
        "ann_vol": float(vol),
        "ann_return": float(mean),
        "return_over_vol": float(mean / vol) if vol > 0 else np.nan,
        "avg_pairwise_corr": float(avg_corr),
    }


def central_vs_peripheral(
    z_scores: pd.Series,
    next_returns: pd.DataFrame,
    n_pick: int,
    seed: int,
    n_random: int = 1000,
) -> Tuple[Dict, Dict, Dict]:
    common = [t for t in z_scores.index if t in next_returns.columns]
    z = z_scores.loc[common].sort_values()
    k = min(n_pick, len(z) // 4)
    core = list(z.index[:k])
    periphery = list(z.index[-k:])
    core_stats = _portfolio_stats(next_returns[core])
    peri_stats = _portfolio_stats(next_returns[periphery])

    rng = np.random.default_rng(seed)
    pool = np.array(common)
    vols = []
    corrs = []
    for _ in range(n_random):
        pick = rng.choice(pool, size=k, replace=False)
        s = _portfolio_stats(next_returns[list(pick)])
        vols.append(s["ann_vol"])
        corrs.append(s["avg_pairwise_corr"])
    vols = np.asarray(vols)
    corrs = np.asarray(corrs)
    baseline = {
        "random_vol_mean": float(vols.mean()),
        "core_vol_percentile": float((vols < core_stats["ann_vol"]).mean()),
        "periphery_vol_percentile": float((vols < peri_stats["ann_vol"]).mean()),
        "random_corr_mean": float(corrs.mean()),
        "core_corr_percentile": float((corrs < core_stats["avg_pairwise_corr"]).mean()),
        "periphery_corr_percentile": float((corrs < peri_stats["avg_pairwise_corr"]).mean()),
        "n_pick": int(k),
        "n_candidates": int(len(common)),
    }
    return core_stats, peri_stats, baseline


def persistence(z_now: pd.Series, z_next: pd.Series) -> Dict:
    common = z_now.index.intersection(z_next.index)
    if len(common) < 10:
        return {"common": int(len(common)), "spearman": np.nan, "p": np.nan}
    res = stats.spearmanr(z_now.loc[common], z_next.loc[common])
    return {"common": int(len(common)), "spearman": float(res.statistic), "p": float(res.pvalue)}


def next_period_traits(z_scores: pd.Series, next_returns: pd.DataFrame) -> Dict:
    common = [t for t in z_scores.index if t in next_returns.columns]
    z = z_scores.loc[common]
    vol = next_returns[common].std() * np.sqrt(252)
    market = next_returns[common].mean(axis=1)
    beta = next_returns[common].apply(lambda s: np.cov(s, market)[0, 1] / market.var())
    r_vol = stats.spearmanr(z, vol)
    r_beta = stats.spearmanr(z, beta)
    return {
        "spearman_z_vs_next_vol": float(r_vol.statistic),
        "p_vol": float(r_vol.pvalue),
        "spearman_z_vs_next_beta": float(r_beta.statistic),
        "p_beta": float(r_beta.pvalue),
    }
