from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


def load_sector_map(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8").set_index("ticker")


def sector_summary(
    adjacency: np.ndarray, tickers: np.ndarray, sectors: pd.Series
) -> pd.DataFrame:
    sec = sectors.reindex(tickers).fillna("Đa ngành & Khác").to_numpy()
    n = len(tickers)
    total_edges = int(adjacency.sum())
    out_deg = adjacency.sum(axis=1)
    in_deg = adjacency.sum(axis=0)
    rows = []
    for name in sorted(set(sec)):
        mask = sec == name
        n_s = int(mask.sum())
        cross_out = adjacency[np.ix_(mask, ~mask)].sum()
        cross_in = adjacency[np.ix_(~mask, mask)].sum()
        within = adjacency[np.ix_(mask, mask)].sum()
        share_stocks = n_s / n
        rows.append(
            {
                "sector": name,
                "stocks": n_s,
                "share_stocks": share_stocks,
                "mean_out_degree": out_deg[mask].mean(),
                "mean_in_degree": in_deg[mask].mean(),
                "mean_net_degree": (out_deg[mask] - in_deg[mask]).mean(),
                "share_out_edges": out_deg[mask].sum() / total_edges if total_edges else np.nan,
                "share_in_edges": in_deg[mask].sum() / total_edges if total_edges else np.nan,
                "lead_ratio": (out_deg[mask].sum() / total_edges) / share_stocks if total_edges else np.nan,
                "follow_ratio": (in_deg[mask].sum() / total_edges) / share_stocks if total_edges else np.nan,
                "cross_net_per_stock": (cross_out - cross_in) / n_s,
                "within_edges": int(within),
            }
        )
    frame = pd.DataFrame(rows)
    frame["lead_minus_follow"] = frame["lead_ratio"] - frame["follow_ratio"]
    return frame.sort_values("lead_minus_follow", ascending=False).reset_index(drop=True)


def sector_flow_matrix(
    adjacency: np.ndarray, tickers: np.ndarray, sectors: pd.Series
) -> pd.DataFrame:
    sec = sectors.reindex(tickers).fillna("Đa ngành & Khác").to_numpy()
    names = sorted(set(sec))
    n = len(tickers)
    overall = adjacency.sum() / (n * (n - 1))
    lift = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        ma = sec == a
        for b in names:
            mb = sec == b
            pairs = ma.sum() * mb.sum() - (ma.sum() if a == b else 0)
            if pairs <= 0 or overall == 0:
                lift.loc[a, b] = np.nan
                continue
            lift.loc[a, b] = adjacency[np.ix_(ma, mb)].sum() / pairs / overall
    return lift


def overrepresentation(
    top_tickers, universe, sectors: pd.Series
) -> pd.DataFrame:
    sec_all = sectors.reindex(universe).fillna("Đa ngành & Khác")
    sec_top = sectors.reindex(list(top_tickers)).fillna("Đa ngành & Khác")
    n_all, n_top = len(sec_all), len(sec_top)
    rows = []
    for name, total in sec_all.value_counts().items():
        observed = int((sec_top == name).sum())
        expected = n_top * total / n_all
        p_value = stats.hypergeom.sf(observed - 1, n_all, total, n_top) if observed > 0 else 1.0
        rows.append(
            {
                "sector": name,
                "universe": int(total),
                "top_count": observed,
                "expected": expected,
                "ratio": observed / expected if expected else np.nan,
                "p_hypergeom": float(p_value),
            }
        )
    return pd.DataFrame(rows).sort_values(["ratio", "top_count"], ascending=False).reset_index(drop=True)


def _nmi(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    table = pd.crosstab(labels_a, labels_b).to_numpy().astype(float)
    total = table.sum()
    pij = table / total
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    nonzero = pij > 0
    mutual = (pij[nonzero] * np.log(pij[nonzero] / (pi @ pj)[nonzero])).sum()
    h_a = -(pi[pi > 0] * np.log(pi[pi > 0])).sum()
    h_b = -(pj[pj > 0] * np.log(pj[pj > 0])).sum()
    return float(2 * mutual / (h_a + h_b)) if (h_a + h_b) > 0 else np.nan


def homophily(
    adjacency: np.ndarray,
    tickers: np.ndarray,
    sectors: pd.Series,
    communities: pd.Series,
    seed: int,
    permutations: int = 500,
) -> Dict:
    sec = sectors.reindex(tickers).fillna("Đa ngành & Khác").to_numpy()
    n = len(tickers)
    total = adjacency.sum()
    same = sec[:, None] == sec[None, :]
    np.fill_diagonal(same, False)
    within = adjacency[same].sum()
    expected_share = same.sum() / (n * (n - 1))
    observed_share = within / total if total else np.nan

    graph = nx.from_numpy_array(((adjacency + adjacency.T) > 0).astype(int))
    nx.set_node_attributes(graph, {i: sec[i] for i in range(n)}, "sector")
    assort = nx.attribute_assortativity_coefficient(graph, "sector")

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        perm = rng.permutation(sec)
        s = perm[:, None] == perm[None, :]
        np.fill_diagonal(s, False)
        null.append(adjacency[s].sum() / total if total else np.nan)
    null = np.asarray(null)
    p_perm = float((np.sum(null >= observed_share) + 1) / (permutations + 1))

    comm = communities.reindex(tickers).fillna(0).to_numpy()
    return {
        "within_sector_edge_share": float(observed_share),
        "expected_within_share_random": float(expected_share),
        "within_ratio": float(observed_share / expected_share) if expected_share else np.nan,
        "permutation_p": p_perm,
        "sector_assortativity": float(assort),
        "nmi_louvain_vs_sector": _nmi(comm, sec),
    }
