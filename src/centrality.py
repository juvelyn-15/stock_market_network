from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def undirected_weights(adjacency: np.ndarray, f_stat: np.ndarray) -> np.ndarray:
    directed = np.where(adjacency.astype(bool), np.nan_to_num(f_stat), 0.0)
    return directed + directed.T


def giant_component(weights: np.ndarray) -> np.ndarray:
    _, labels = connected_components(csr_matrix(weights > 0), directed=False)
    counts = np.bincount(labels)
    return np.nonzero(labels == counts.argmax())[0]


def _rank(values: np.ndarray, descending: bool) -> np.ndarray:
    return pd.Series(values).rank(ascending=not descending, method="average").to_numpy()


def pozzi_measures(weights: np.ndarray, tickers: np.ndarray) -> pd.DataFrame:
    n = weights.shape[0]
    binary = (weights > 0).astype(float)
    safe = np.where(weights > 0, weights, 1.0)
    distance = np.where(weights > 0, 1.0 / safe, 0.0)

    g_u = nx.from_numpy_array(binary)
    g_w = nx.from_numpy_array(weights)
    for _, _, data in g_w.edges(data=True):
        data["distance"] = 1.0 / data["weight"]

    bc_u = nx.betweenness_centrality(g_u, normalized=True)
    bc_w = nx.betweenness_centrality(g_w, weight="distance", normalized=True)
    ec_u = nx.eigenvector_centrality_numpy(g_u)
    ec_w = nx.eigenvector_centrality_numpy(g_w, weight="weight")

    d_u = shortest_path(csr_matrix(binary), directed=False, unweighted=True)
    d_w = shortest_path(csr_matrix(distance), directed=False)

    frame = pd.DataFrame(index=tickers)
    frame["degree_u"] = binary.sum(axis=1)
    frame["strength_w"] = weights.sum(axis=1)
    frame["betweenness_u"] = [bc_u[i] for i in range(n)]
    frame["betweenness_w"] = [bc_w[i] for i in range(n)]
    frame["eigenvector_u"] = np.abs([ec_u[i] for i in range(n)])
    frame["eigenvector_w"] = np.abs([ec_w[i] for i in range(n)])
    frame["eccentricity_u"] = d_u.max(axis=1)
    frame["eccentricity_w"] = d_w.max(axis=1)
    frame["closeness_u"] = d_u.sum(axis=1) / (n - 1)
    frame["closeness_w"] = d_w.sum(axis=1) / (n - 1)

    desc = ["degree_u", "strength_w", "betweenness_u", "betweenness_w", "eigenvector_u", "eigenvector_w"]
    asc = ["eccentricity_u", "eccentricity_w", "closeness_u", "closeness_w"]
    for column in desc:
        frame[f"rank_{column}"] = _rank(frame[column].to_numpy(), True)
    for column in asc:
        frame[f"rank_{column}"] = _rank(frame[column].to_numpy(), False)

    x_sum = (
        frame["rank_strength_w"]
        + frame["rank_degree_u"]
        + frame["rank_betweenness_w"]
        + frame["rank_betweenness_u"]
    )
    y_sum = (
        frame["rank_eccentricity_w"]
        + frame["rank_eccentricity_u"]
        + frame["rank_closeness_w"]
        + frame["rank_closeness_u"]
        + frame["rank_eigenvector_w"]
        + frame["rank_eigenvector_u"]
    )
    frame["X"] = (x_sum - 4) / (4 * (n - 1))
    frame["Y"] = (y_sum - 6) / (6 * (n - 1))
    frame["Z"] = (frame["X"] + frame["Y"]) / 2
    low, high = frame["Z"].quantile([0.2, 0.8])
    frame["zone"] = np.select(
        [frame["Z"] <= low, frame["Z"] >= high], ["lõi", "ngoại vi"], default="trung gian"
    )
    frame.index.name = "ticker"
    return frame.sort_values("Z")


def measures_for_network(
    adjacency: np.ndarray, f_stat: np.ndarray, tickers: np.ndarray, min_nodes: int
) -> Tuple[pd.DataFrame, dict]:
    weights = undirected_weights(adjacency, f_stat)
    keep = giant_component(weights)
    info = {
        "nodes_total": int(len(tickers)),
        "nodes_giant": int(len(keep)),
        "giant_share": float(len(keep) / len(tickers)),
    }
    if len(keep) < min_nodes:
        return pd.DataFrame(), info
    sub = weights[np.ix_(keep, keep)]
    return pozzi_measures(sub, tickers[keep]), info
