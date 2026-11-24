from typing import Iterable, List

import numpy as np
import pandas as pd

from granger import fdr_adjacency, pairwise_granger, raw_adjacency


def lag_sensitivity(
    returns: pd.DataFrame,
    lags: Iterable[int],
    alpha: float,
    fdr_method: str,
    base_lag: int,
    top_k: int = 15,
) -> pd.DataFrame:
    tickers = np.asarray(returns.columns)
    n_assets = len(tickers)
    mask = ~np.eye(n_assets, dtype=bool)
    base_f, base_p, _ = pairwise_granger(returns, base_lag)
    base_adj = raw_adjacency(base_p, alpha)
    base_top = set(tickers[np.argsort(-base_adj.sum(axis=1))[:top_k]])

    rows: List[dict] = []
    for lag in lags:
        f_stat, p_value, _ = pairwise_granger(returns, lag)
        raw = raw_adjacency(p_value, alpha)
        fdr, _ = fdr_adjacency(p_value, alpha, fdr_method)
        intersection = (raw & base_adj)[mask].sum()
        union = (raw | base_adj)[mask].sum()
        top = set(tickers[np.argsort(-raw.sum(axis=1))[:top_k]])
        rows.append(
            {
                "lag": int(lag),
                "raw05_edges": int(raw[mask].sum()),
                "raw05_density": float(raw[mask].mean()),
                "fdr05_edges": int(fdr[mask].sum()),
                "fdr05_density": float(fdr[mask].mean()),
                "jaccard_raw_vs_base": float(intersection / union) if union else np.nan,
                "top_out_overlap_vs_base": float(len(top & base_top) / top_k),
                "out_degree_spearman_vs_base": float(
                    pd.Series(raw.sum(axis=1)).corr(pd.Series(base_adj.sum(axis=1)), method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)
