from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from centrality import measures_for_network
from config import Settings
from data_loader import load_prices, prepare_window
from portfolio import central_vs_peripheral, next_period_traits, persistence
from sectors import (
    homophily,
    load_sector_map,
    overrepresentation,
    sector_flow_matrix,
    sector_summary,
)
from visualization import (
    plot_portfolio,
    plot_sector_flow,
    plot_sector_lead,
    plot_xy,
)


def _read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def run_centrality(settings: Settings, sector_path: Path, min_nodes: int = 30, top_k: int = 20) -> Dict:
    prices = load_prices(settings.data_path)
    sector_table = load_sector_map(sector_path)
    sectors = sector_table["sector"]
    confidence = sector_table["confidence"]
    base = settings.output_dir / "centrality"
    base.mkdir(parents=True, exist_ok=True)

    returns_by_period = {}
    for period in settings.periods:
        window = prepare_window(
            prices, period.start, period.end,
            settings.min_coverage, settings.max_zero_share, settings.winsor_abs,
        )
        returns_by_period[period.name] = window.returns

    summary_rows: List[dict] = []
    homophily_rows: List[dict] = []
    top_rows: List[pd.DataFrame] = []
    overrep_rows: List[pd.DataFrame] = []
    sector_rows: List[pd.DataFrame] = []
    z_store: Dict[str, Dict[str, pd.Series]] = {}

    for rule in settings.edge_rules:
        rule_dir = base / rule
        fig_dir = rule_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        z_store[rule] = {}
        rule_label = "p<0.05 (mạng chuẩn)" if rule == "raw05" else "FDR 5% (mạng lõi)"

        for period in settings.periods:
            folder = settings.output_dir / period.name
            f_df = _read_matrix(folder / "granger_F_matrix.csv")
            adj_df = _read_matrix(folder / rule / "adjacency_matrix.csv")
            tickers = np.asarray(adj_df.index)
            adjacency = adj_df.to_numpy().astype(bool)
            f_stat = f_df.loc[tickers, tickers].to_numpy()

            nodes = pd.read_csv(folder / rule / "node_metrics.csv", index_col=0)
            measures, info = measures_for_network(adjacency, f_stat, tickers, min_nodes)

            row = {"rule": rule, "period": period.name, **info}
            if measures.empty:
                row["status"] = "bỏ qua: thành phần liên thông lớn nhất quá nhỏ"
                summary_rows.append(row)
                continue

            measures["sector"] = sectors.reindex(measures.index).fillna("Đa ngành & Khác")
            measures["sector_confidence"] = confidence.reindex(measures.index).fillna("thấp")
            measures = measures.join(nodes[["in_degree", "out_degree", "net_degree", "community"]], how="left")
            measures.to_csv(rule_dir / f"{period.name}_centrality.csv", float_format="%.6f")
            z_store[rule][period.name] = measures["Z"]

            plot_xy(
                measures,
                f"{period.label} | {rule_label}: vị trí X–Y (góc trên-phải của đồ thị = trung tâm)",
                fig_dir / f"{period.name}_xy.png",
            )

            top = measures.head(top_k).copy()
            top.insert(0, "period", period.name)
            top.insert(0, "rule", rule)
            top_rows.append(top.reset_index())

            over = overrepresentation(measures.head(top_k).index, measures.index, sectors)
            over.insert(0, "period", period.name)
            over.insert(0, "rule", rule)
            overrep_rows.append(over)

            sec_sum = sector_summary(adjacency, tickers, sectors)
            sec_sum.insert(0, "period", period.name)
            sec_sum.insert(0, "rule", rule)
            sector_rows.append(sec_sum)
            plot_sector_lead(
                sec_sum,
                f"{period.label} | {rule_label}: ngành dẫn dắt và bị dẫn dắt",
                fig_dir / f"{period.name}_sector_lead.png",
            )
            lift = sector_flow_matrix(adjacency, tickers, sectors)
            lift.to_csv(rule_dir / f"{period.name}_sector_flow_lift.csv", float_format="%.4f")
            if rule == "fdr05" or period.name in ("P1", "P5"):
                plot_sector_flow(
                    lift,
                    f"{period.label} | {rule_label}: dòng chảy giữa các ngành",
                    fig_dir / f"{period.name}_sector_flow.png",
                )

            hom = homophily(adjacency, tickers, sectors, nodes["community"], settings.seed)
            homophily_rows.append({"rule": rule, "period": period.name, **hom})

            row.update(
                {
                    "status": "ok",
                    "z_corr_out_degree": float(measures["Z"].corr(measures["out_degree"], method="spearman")),
                    "z_corr_in_degree": float(measures["Z"].corr(measures["in_degree"], method="spearman")),
                    "core_share_high_conf_sector": float(
                        (measures.head(top_k)["sector_confidence"] == "cao").mean()
                    ),
                }
            )
            summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(base / "centrality_status.csv", index=False, float_format="%.5g")
    if top_rows:
        pd.concat(top_rows).to_csv(base / "top_central_stocks.csv", index=False, float_format="%.5f")
    if overrep_rows:
        pd.concat(overrep_rows).to_csv(base / "sector_overrepresentation_top_central.csv", index=False, float_format="%.5g")
    if sector_rows:
        pd.concat(sector_rows).to_csv(base / "sector_lead_follow.csv", index=False, float_format="%.5g")
    if homophily_rows:
        pd.DataFrame(homophily_rows).to_csv(base / "sector_homophily.csv", index=False, float_format="%.5g")

    portfolio_rows = []
    persistence_rows = []
    trait_rows = []
    names = [p.name for p in settings.periods]
    for rule in settings.edge_rules:
        for current, nxt in zip(names[:-1], names[1:]):
            if current not in z_store[rule]:
                continue
            z_now = z_store[rule][current]
            nxt_returns = returns_by_period[nxt]
            core, peri, base_stats = central_vs_peripheral(z_now, nxt_returns, 20, settings.seed)
            portfolio_rows.append(
                {
                    "rule": rule,
                    "transition": f"{current}→{nxt}",
                    "core_vol": core["ann_vol"],
                    "periphery_vol": peri["ann_vol"],
                    "random_vol": base_stats["random_vol_mean"],
                    "core_corr": core["avg_pairwise_corr"],
                    "periphery_corr": peri["avg_pairwise_corr"],
                    "random_corr": base_stats["random_corr_mean"],
                    "core_return_over_vol": core["return_over_vol"],
                    "periphery_return_over_vol": peri["return_over_vol"],
                    "core_vol_percentile": base_stats["core_vol_percentile"],
                    "periphery_vol_percentile": base_stats["periphery_vol_percentile"],
                    "core_corr_percentile": base_stats["core_corr_percentile"],
                    "periphery_corr_percentile": base_stats["periphery_corr_percentile"],
                    "n_pick": base_stats["n_pick"],
                    "n_candidates": base_stats["n_candidates"],
                }
            )
            if nxt in z_store[rule]:
                persistence_rows.append({"rule": rule, "transition": f"{current}→{nxt}", **persistence(z_now, z_store[rule][nxt])})
            trait_rows.append({"rule": rule, "transition": f"{current}→{nxt}", **next_period_traits(z_now, nxt_returns)})

    if portfolio_rows:
        port = pd.DataFrame(portfolio_rows)
        port.to_csv(base / "portfolio_core_vs_periphery.csv", index=False, float_format="%.5g")
        for rule in settings.edge_rules:
            subset = port[port["rule"] == rule]
            if len(subset) >= 2:
                plot_portfolio(subset, base / rule / "figures" / "portfolio_core_vs_periphery.png")
    if persistence_rows:
        pd.DataFrame(persistence_rows).to_csv(base / "centrality_persistence.csv", index=False, float_format="%.5g")
    if trait_rows:
        pd.DataFrame(trait_rows).to_csv(base / "z_vs_next_period_traits.csv", index=False, float_format="%.5g")
    return {"base": base}
