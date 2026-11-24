from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR


def select_lag(
    returns: pd.DataFrame, max_lag: int, sample_size: int, seed: int
) -> Tuple[int, dict]:
    rng = np.random.default_rng(seed)
    columns = returns.columns.to_numpy()
    size = min(sample_size, len(columns))
    sample = rng.choice(columns, size=size, replace=False)
    order = VAR(returns[sample].to_numpy()).select_order(maxlags=max_lag)
    selected = {key: int(value) for key, value in order.selected_orders.items()}
    lag = max(1, selected.get("bic", 1))
    return lag, selected


def _lag_matrix(values: np.ndarray, lag: int) -> np.ndarray:
    t_total, n_assets = values.shape
    blocks = [values[lag - k : t_total - k] for k in range(1, lag + 1)]
    stacked = np.stack(blocks, axis=2)
    return stacked.reshape(t_total - lag, n_assets * lag)


def pairwise_granger(
    returns: pd.DataFrame, lag: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    values = returns.to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    t_total, n_assets = values.shape

    target = values[lag:]
    lagged = _lag_matrix(values, lag)
    t_eff = target.shape[0]
    design = np.hstack([np.ones((t_eff, 1)), lagged])

    gram = design.T @ design
    cross = design.T @ target
    yy = np.einsum("ij,ij->j", target, target)

    f_stat = np.full((n_assets, n_assets), np.nan)
    p_value = np.full((n_assets, n_assets), np.nan)

    lag_range = np.arange(lag)
    source_cols = 1 + np.arange(n_assets)[:, None] * lag + lag_range[None, :]
    dof_unrestricted = t_eff - (1 + 2 * lag)

    for j in range(n_assets):
        own = np.concatenate(([0], 1 + j * lag + lag_range))

        g_r = gram[np.ix_(own, own)]
        c_r = cross[own, j]
        beta_r = np.linalg.solve(g_r, c_r)
        rss_r = yy[j] - c_r @ beta_r

        safe_sources = source_cols.copy()
        safe_sources[j] = source_cols[(j + 1) % n_assets]
        idx = np.concatenate(
            [np.broadcast_to(own, (n_assets, own.size)), safe_sources], axis=1
        )
        g_u = gram[idx[:, :, None], idx[:, None, :]]
        c_u = cross[idx, j]
        beta_u = np.linalg.solve(g_u, c_u[:, :, None])[:, :, 0]
        rss_u = yy[j] - np.einsum("ik,ik->i", c_u, beta_u)

        stat = ((rss_r - rss_u) / lag) / (rss_u / dof_unrestricted)
        pval = stats.f.sf(stat, lag, dof_unrestricted)

        stat[j] = np.nan
        pval[j] = np.nan
        f_stat[:, j] = stat
        p_value[:, j] = pval

    return f_stat, p_value, dof_unrestricted


def fdr_adjacency(
    p_value: np.ndarray, alpha: float, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    from statsmodels.stats.multitest import multipletests

    n_assets = p_value.shape[0]
    mask = ~np.eye(n_assets, dtype=bool)
    flat = p_value[mask]
    reject, adjusted, _, _ = multipletests(flat, alpha=alpha, method=method)
    adjacency = np.zeros_like(p_value, dtype=bool)
    adjacency[mask] = reject
    q_value = np.full_like(p_value, np.nan)
    q_value[mask] = adjusted
    return adjacency, q_value


def raw_adjacency(p_value: np.ndarray, alpha: float) -> np.ndarray:
    adjacency = np.zeros_like(p_value, dtype=bool)
    valid = ~np.isnan(p_value)
    adjacency[valid] = p_value[valid] < alpha
    return adjacency


def reference_granger_pvalue(
    returns: pd.DataFrame, source: str, target: str, lag: int
) -> Optional[float]:
    from statsmodels.tsa.stattools import grangercausalitytests

    pair = returns[[target, source]].to_numpy()
    result = grangercausalitytests(pair, maxlag=[lag])
    return float(result[lag][0]["ssr_ftest"][1])
