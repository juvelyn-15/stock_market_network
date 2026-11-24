from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class WindowData:
    returns: pd.DataFrame
    info: Dict


def load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], format="%d/%m/%Y")
    frame = frame.sort_values("datetime").drop_duplicates("datetime")
    prices = frame.set_index("datetime")
    prices = prices.dropna(axis=1, how="all")
    prices = prices.where(prices > 0)
    return prices.astype(float)


def prepare_window(
    prices: pd.DataFrame,
    start: str,
    end: str,
    min_coverage: float,
    max_zero_share: float,
    winsor_abs: float,
) -> WindowData:
    window = prices.loc[start:end]
    n_days = len(window)
    n_listed = int(window.notna().any().sum())
    coverage = window.notna().mean()
    kept = coverage[coverage >= min_coverage].index
    window = window[kept]

    returns = np.log(window).diff().iloc[1:]
    returns = returns.fillna(0.0)

    zero_share = (returns == 0.0).mean()
    liquid = zero_share[zero_share <= max_zero_share].index
    dropped_illiquid = len(kept) - len(liquid)
    returns = returns[liquid]

    clipped_share = float((returns.abs() > winsor_abs).to_numpy().mean())
    returns = returns.clip(lower=-winsor_abs, upper=winsor_abs)

    std = returns.std()
    returns = returns.loc[:, std > 0]

    info = {
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "trading_days": int(len(returns)),
        "price_rows": int(n_days),
        "tickers_with_any_price": n_listed,
        "tickers_after_coverage_filter": int(len(kept)),
        "tickers_dropped_illiquid": int(dropped_illiquid),
        "tickers_final": int(returns.shape[1]),
        "share_returns_winsorized": clipped_share,
    }
    return WindowData(returns=returns, info=info)


def stationarity_summary(returns: pd.DataFrame, alpha: float = 0.05) -> Dict:
    pvalues = []
    for column in returns.columns:
        series = returns[column].to_numpy()
        pvalues.append(adfuller(series, autolag="AIC", result_object=False)[1])
    pvalues = np.asarray(pvalues)
    return {
        "adf_share_stationary": float((pvalues < alpha).mean()),
        "adf_max_pvalue": float(pvalues.max()),
    }
