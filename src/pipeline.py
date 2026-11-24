import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from config import Period, Settings
from data_loader import load_prices, prepare_window, stationarity_summary
from granger import fdr_adjacency, pairwise_granger, raw_adjacency, select_lag
from network import (
    attach_stock_traits,
    build_graph,
    edge_table,
    network_metrics,
    node_metrics,
    trait_correlations,
)
from sensitivity import lag_sensitivity
from validation import validate_against_statsmodels
from visualization import (
    plot_adjacency,
    plot_degree_distributions,
    plot_in_out_scatter,
    plot_leader_persistence,
    plot_network,
    plot_period_comparison,
    plot_top_bars,
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _write_json(payload: Dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _pi0(p_value: np.ndarray) -> float:
    mask = ~np.eye(p_value.shape[0], dtype=bool)
    flat = p_value[mask]
    return float(min(1.0, (flat > 0.5).mean() / 0.5))


def _adjacency_for_rule(rule, raw, fdr):
    return raw if rule == "raw05" else fdr


def run_period(prices: pd.DataFrame, period: Period, settings: Settings) -> Dict:
    folder = settings.output_dir / period.name
    folder.mkdir(parents=True, exist_ok=True)

    window = prepare_window(
        prices,
        period.start,
        period.end,
        settings.min_coverage,
        settings.max_zero_share,
        settings.winsor_abs,
    )
    returns = window.returns
    tickers: List[str] = list(returns.columns)

    stationarity = stationarity_summary(returns)

    if settings.fixed_lag is not None:
        lag, lag_orders = settings.fixed_lag, {"fixed": settings.fixed_lag}
    else:
        lag, lag_orders = select_lag(returns, settings.max_lag, settings.lag_sample_size, settings.seed)

    f_stat, p_value, dof = pairwise_granger(returns, lag)
    adj_fdr, q_value = fdr_adjacency(p_value, settings.alpha, settings.fdr_method)
    adj_raw = raw_adjacency(p_value, settings.alpha)
    n_pairs = len(tickers) * (len(tickers) - 1)

    validation = validate_against_statsmodels(
        returns, f_stat, p_value, lag, settings.validation_pairs, settings.seed
    )
    sensitivity = lag_sensitivity(
        returns, settings.sensitivity_lags, settings.alpha, settings.fdr_method, lag
    )
    sensitivity.insert(0, "period", period.name)

    pd.DataFrame(f_stat, index=tickers, columns=tickers).to_csv(folder / "granger_F_matrix.csv", float_format="%.5f")
    pd.DataFrame(p_value, index=tickers, columns=tickers).to_csv(folder / "granger_p_matrix.csv", float_format="%.6g")

    pi0 = _pi0(p_value)
    context = {
        "period": period.name,
        "label": period.label,
        "data": window.info,
        "stationarity": stationarity,
        "lag": lag,
        "lag_selection_orders": lag_orders,
        "dof_unrestricted": dof,
        "alpha": settings.alpha,
        "fdr_method": settings.fdr_method,
        "validation": validation,
        "pi0_storey": pi0,
        "expected_false_raw_edges": settings.alpha * n_pairs,
        "observed_raw_edges": int(adj_raw.sum()),
        "observed_fdr_edges": int(adj_fdr.sum()),
    }
    _write_json(context, folder / "context.json")

    by_rule: Dict[str, Dict] = {}
    for rule in settings.edge_rules:
        rule_dir = folder / rule
        figures = rule_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        adjacency = _adjacency_for_rule(rule, adj_raw, adj_fdr)
        rule_label = "p<0.05 (mạng chuẩn)" if rule == "raw05" else "FDR 5% (mạng lõi)"

        graph = build_graph(tickers, adjacency, f_stat, p_value, q_value)
        nodes = node_metrics(graph, settings.seed)
        nodes = attach_stock_traits(nodes, returns)
        metrics = network_metrics(graph, nodes, settings.seed)
        metrics["trait_correlations"] = trait_correlations(nodes)
        edges = edge_table(graph)

        nodes.to_csv(rule_dir / "node_metrics.csv", float_format="%.6f")
        edges.to_csv(rule_dir / "edges.csv", index=False, float_format="%.6g")
        pd.DataFrame(adjacency.astype(int), index=tickers, columns=tickers).to_csv(rule_dir / "adjacency_matrix.csv")
        nx.write_graphml(graph, rule_dir / "network.graphml")
        _write_json({"context": context, "rule": rule, "network": metrics}, rule_dir / "summary.json")

        title = f"{period.label} | {rule_label}"
        plot_degree_distributions(nodes, title, figures / "degree_distribution.png")
        plot_top_bars(nodes, settings.top_n, title, figures / "top_nodes.png")
        plot_in_out_scatter(nodes, title, figures / "in_out_scatter.png")
        plot_network(
            graph,
            nodes,
            settings.graph_top_nodes,
            settings.graph_max_edges,
            f"Mạng Granger HOSE {title}",
            figures / "network.png",
            settings.seed,
        )
        if adjacency.sum() > 0:
            plot_adjacency(adjacency, tickers, nodes, f"{title}: ma trận kề theo cộng đồng", figures / "adjacency.png")

        by_rule[rule] = {"network": metrics, "nodes": nodes}

    return {"context": context, "by_rule": by_rule, "sensitivity": sensitivity}


def _rule_outputs(results: List[Dict], rule: str, settings: Settings) -> pd.DataFrame:
    rows = []
    for item in results:
        ctx, net, data = item["context"], item["by_rule"][rule]["network"], item["context"]["data"]
        rows.append(
            {
                "period": ctx["period"],
                "label": ctx["label"],
                "start": data["start"],
                "end": data["end"],
                "trading_days": data["trading_days"],
                "nodes": net["nodes"],
                "lag": ctx["lag"],
                "edges": net["edges"],
                "density": net["density"],
                "reciprocity": net["reciprocity"],
                "avg_clustering": net["avg_clustering"],
                "transitivity": net["transitivity_undirected"],
                "giant_wcc_share": net["giant_wcc_share"],
                "giant_scc_share": net["giant_scc_share"],
                "avg_path_length_giant_scc": net["avg_path_length_giant_scc"],
                "diameter_giant_scc": net["diameter_giant_scc"],
                "communities": net["communities"],
                "modularity": net["modularity"],
                "assortativity_out_in": net["assortativity_out_in"],
                "out_centralization": net["out_centralization"],
                "in_centralization": net["in_centralization"],
                "isolated_nodes": net["isolated_nodes"],
                "max_out_degree": net["max_out_degree"],
                "max_in_degree": net["max_in_degree"],
                "pi0_storey": ctx["pi0_storey"],
                "adf_share_stationary": ctx["stationarity"]["adf_share_stationary"],
                "validation_max_abs_p_diff": ctx["validation"]["max_abs_p_diff"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(settings.output_dir / f"period_comparison_{rule}.csv", index=False, float_format="%.6g")
    plot_period_comparison(
        summary,
        settings.output_dir / f"period_comparison_{rule}.png",
        "p<0.05 (mạng chuẩn)" if rule == "raw05" else "FDR 5% (mạng lõi)",
    )

    frames = []
    for item in results:
        nodes = item["by_rule"][rule]["nodes"]
        name = item["context"]["period"]
        frames.append(nodes[["out_degree", "in_degree", "net_degree", "pagerank", "betweenness"]].add_suffix(f"_{name}"))
    wide = pd.concat(frames, axis=1)
    wide.index.name = "ticker"
    wide.to_csv(settings.output_dir / f"node_metrics_all_periods_{rule}.csv", float_format="%.6f")

    top_flags = {}
    norm_cols = {}
    for item in results:
        name = item["context"]["period"]
        n_nodes = item["by_rule"][rule]["network"]["nodes"]
        col = wide[f"out_degree_{name}"]
        top_index = col.dropna().sort_values(ascending=False).head(settings.top_n).index
        top_flags[name] = wide.index.isin(top_index)
        norm_cols[name] = col / (n_nodes - 1)
    flags = pd.DataFrame(top_flags, index=wide.index)
    norms = pd.DataFrame(norm_cols, index=wide.index)
    persistence = pd.DataFrame(
        {
            "periods_in_top": flags.sum(axis=1),
            "periods_present": norms.notna().sum(axis=1),
            "mean_out_degree_norm": norms.mean(axis=1),
        }
    ).sort_values(["periods_in_top", "mean_out_degree_norm"], ascending=False)
    persistence.to_csv(settings.output_dir / f"leader_persistence_{rule}.csv", float_format="%.5f")
    plot_leader_persistence(persistence, settings.output_dir / f"leader_persistence_{rule}.png")
    return summary


def run_all(settings: Settings) -> Dict:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_prices(settings.data_path)

    results = [run_period(prices, period, settings) for period in settings.periods]

    summaries = {rule: _rule_outputs(results, rule, settings) for rule in settings.edge_rules}

    pd.concat([r["sensitivity"] for r in results]).to_csv(
        settings.output_dir / "sensitivity_lag.csv", index=False, float_format="%.5g"
    )

    _write_json(
        {
            "settings": {
                "min_coverage": settings.min_coverage,
                "max_zero_share": settings.max_zero_share,
                "winsor_abs": settings.winsor_abs,
                "max_lag": settings.max_lag,
                "alpha": settings.alpha,
                "fdr_method": settings.fdr_method,
                "edge_rules": list(settings.edge_rules),
                "seed": settings.seed,
            },
            "periods": [r["context"] for r in results],
        },
        settings.output_dir / "run_summary.json",
    )
    return {"summaries": summaries, "results": results}
