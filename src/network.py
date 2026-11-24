from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


def build_graph(
    tickers: List[str],
    adjacency: np.ndarray,
    f_stat: np.ndarray,
    p_value: np.ndarray,
    q_value: np.ndarray,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(tickers)
    sources, targets = np.nonzero(adjacency)
    for i, j in zip(sources, targets):
        graph.add_edge(
            tickers[i],
            tickers[j],
            weight=float(f_stat[i, j]),
            pvalue=float(p_value[i, j]),
            qvalue=float(q_value[i, j]),
        )
    return graph


def edge_table(graph: nx.DiGraph) -> pd.DataFrame:
    rows = [
        {
            "source": u,
            "target": v,
            "f_stat": d["weight"],
            "p_value": d["pvalue"],
            "q_value": d["qvalue"],
        }
        for u, v, d in graph.edges(data=True)
    ]
    columns = ["source", "target", "f_stat", "p_value", "q_value"]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values("f_stat", ascending=False).reset_index(drop=True)


def node_metrics(graph: nx.DiGraph, seed: int) -> pd.DataFrame:
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    in_str = dict(graph.in_degree(weight="weight"))
    out_str = dict(graph.out_degree(weight="weight"))

    frame = pd.DataFrame(index=nodes)
    frame["in_degree"] = pd.Series(in_deg)
    frame["out_degree"] = pd.Series(out_deg)
    frame["degree"] = frame["in_degree"] + frame["out_degree"]
    frame["net_degree"] = frame["out_degree"] - frame["in_degree"]
    frame["in_degree_norm"] = frame["in_degree"] / (n_nodes - 1)
    frame["out_degree_norm"] = frame["out_degree"] / (n_nodes - 1)
    frame["in_strength"] = pd.Series(in_str)
    frame["out_strength"] = pd.Series(out_str)
    frame["net_strength"] = frame["out_strength"] - frame["in_strength"]

    recip = []
    for node in nodes:
        successors = set(graph.successors(node))
        predecessors = set(graph.predecessors(node))
        recip.append(len(successors & predecessors))
    frame["reciprocal_links"] = recip

    frame["betweenness"] = pd.Series(nx.betweenness_centrality(graph, normalized=True))
    frame["closeness_in"] = pd.Series(nx.closeness_centrality(graph))
    frame["closeness_out"] = pd.Series(nx.closeness_centrality(graph.reverse(copy=False)))
    frame["pagerank"] = pd.Series(nx.pagerank(graph, alpha=0.85, weight="weight"))

    if graph.number_of_edges() > 0:
        hubs, authorities = nx.hits(graph, max_iter=2000, tol=1e-10, normalized=True)
    else:
        hubs = {n: 0.0 for n in nodes}
        authorities = {n: 0.0 for n in nodes}
    frame["hub_score"] = pd.Series(hubs)
    frame["authority_score"] = pd.Series(authorities)

    frame["clustering"] = pd.Series(nx.clustering(graph))

    undirected = graph.to_undirected(reciprocal=False)
    for u, v in undirected.edges():
        w = 0.0
        if graph.has_edge(u, v):
            w += graph[u][v]["weight"]
        if graph.has_edge(v, u):
            w += graph[v][u]["weight"]
        undirected[u][v]["weight"] = w
    communities: List[set] = []
    if undirected.number_of_edges() > 0:
        communities = nx.community.louvain_communities(
            undirected, weight="weight", seed=seed
        )
    communities = sorted(communities, key=len, reverse=True)
    membership = {}
    for index, members in enumerate(communities, start=1):
        for node in members:
            membership[node] = index
    frame["community"] = pd.Series(membership).reindex(nodes).fillna(0).astype(int)

    frame["role"] = np.select(
        [
            (frame["out_degree"] > 0) & (frame["net_degree"] >= frame["degree"] * 0.5),
            (frame["in_degree"] > 0) & (frame["net_degree"] <= -frame["degree"] * 0.5),
        ],
        ["leader", "follower"],
        default="mixed",
    )
    frame.loc[frame["degree"] == 0, "role"] = "isolated"
    frame.index.name = "ticker"
    return frame.sort_values("degree", ascending=False)


def _giant_component_stats(graph: nx.DiGraph) -> Dict:
    if graph.number_of_edges() == 0:
        return {"avg_path_length_giant_scc": np.nan, "diameter_giant_scc": np.nan}
    components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    giant = graph.subgraph(components[0])
    if giant.number_of_nodes() < 2:
        return {"avg_path_length_giant_scc": np.nan, "diameter_giant_scc": np.nan}
    return {
        "avg_path_length_giant_scc": float(nx.average_shortest_path_length(giant)),
        "diameter_giant_scc": int(nx.diameter(giant)),
    }


def network_metrics(graph: nx.DiGraph, nodes: pd.DataFrame, seed: int) -> Dict:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    metrics: Dict = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": float(nx.density(graph)),
        "avg_degree_in_out": float(n_edges / n_nodes) if n_nodes else np.nan,
        "max_in_degree": int(nodes["in_degree"].max()),
        "max_out_degree": int(nodes["out_degree"].max()),
        "isolated_nodes": int((nodes["degree"] == 0).sum()),
        "reciprocity": float(nx.reciprocity(graph)) if n_edges else np.nan,
        "avg_clustering": float(nodes["clustering"].mean()),
        "transitivity_undirected": float(nx.transitivity(graph.to_undirected())),
    }

    weak = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    strong = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    metrics["weak_components"] = len(weak)
    metrics["giant_wcc_share"] = len(weak[0]) / n_nodes
    metrics["strong_components"] = len(strong)
    metrics["giant_scc_share"] = len(strong[0]) / n_nodes
    metrics.update(_giant_component_stats(graph))

    if n_edges > 2:
        metrics["assortativity_out_in"] = float(
            nx.degree_assortativity_coefficient(graph, x="out", y="in")
        )
    else:
        metrics["assortativity_out_in"] = np.nan

    n_comm = int(nodes["community"].max())
    metrics["communities"] = n_comm
    if n_comm > 0 and n_edges > 0:
        undirected = graph.to_undirected()
        groups = [set(nodes.index[nodes["community"] == c]) for c in range(1, n_comm + 1)]
        metrics["modularity"] = float(
            nx.community.modularity(undirected, groups)
        )
    else:
        metrics["modularity"] = np.nan

    out_c = nodes["out_degree"].to_numpy(dtype=float)
    in_c = nodes["in_degree"].to_numpy(dtype=float)
    denom = (n_nodes - 1) * (n_nodes - 1)
    metrics["out_centralization"] = float((out_c.max() - out_c).sum() / denom)
    metrics["in_centralization"] = float((in_c.max() - in_c).sum() / denom)
    metrics["corr_in_out_degree_spearman"] = float(
        stats.spearmanr(in_c, out_c).statistic
    )
    metrics["roles"] = {k: int(v) for k, v in nodes["role"].value_counts().items()}
    return metrics


def attach_stock_traits(nodes: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    traits = pd.DataFrame(index=returns.columns)
    traits["volatility_ann"] = returns.std() * np.sqrt(252)
    traits["mean_return_ann"] = returns.mean() * 252
    market = returns.mean(axis=1)
    traits["beta_to_ew_market"] = returns.apply(
        lambda s: np.cov(s, market)[0, 1] / market.var()
    )
    traits["zero_return_share"] = (returns == 0).mean()
    traits["autocorr_lag1"] = returns.apply(lambda s: s.autocorr(1))
    traits.index.name = "ticker"
    return nodes.join(traits, how="left")


def trait_correlations(nodes: pd.DataFrame) -> Dict:
    out: Dict = {}
    for trait in ["volatility_ann", "beta_to_ew_market", "zero_return_share"]:
        for measure in ["out_degree", "in_degree"]:
            rho = stats.spearmanr(nodes[trait], nodes[measure])
            out[f"{measure}_vs_{trait}"] = {
                "rho": float(rho.statistic),
                "p": float(rho.pvalue),
            }
    return out


def top_nodes(nodes: pd.DataFrame, column: str, n: int) -> List[Tuple[str, float]]:
    ranked = nodes[column].sort_values(ascending=False).head(n)
    return [(idx, float(val)) for idx, val in ranked.items()]
