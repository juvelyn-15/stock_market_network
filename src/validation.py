from typing import Dict

import numpy as np
import pandas as pd

from granger import reference_granger_pvalue


def validate_against_statsmodels(
    returns: pd.DataFrame,
    f_stat: np.ndarray,
    p_value: np.ndarray,
    lag: int,
    n_pairs: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed + 1)
    tickers = list(returns.columns)
    n_assets = len(tickers)
    diffs = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_assets, size=2, replace=False)
        reference = reference_granger_pvalue(returns, tickers[i], tickers[j], lag)
        diffs.append(abs(reference - p_value[i, j]))
    return {
        "pairs_checked": int(n_pairs),
        "max_abs_p_diff": float(np.max(diffs)),
        "mean_abs_p_diff": float(np.mean(diffs)),
    }
