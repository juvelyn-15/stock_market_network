from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm, colors

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.titleweight": "bold",
    }
)

PALETTE = ["#2a6f97", "#e07a5f", "#3d9970", "#9b5de5", "#f2a541", "#577590", "#c44569", "#6c757d"]


def plot_degree_distributions(nodes: pd.DataFrame, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    maxdeg = int(max(nodes["in_degree"].max(), nodes["out_degree"].max(), 1))
    bins = np.linspace(0, maxdeg, 30)
    axes[0].hist(nodes["in_degree"], bins=bins, color=PALETTE[0], alpha=0.85)
    axes[0].set_title("Phân phối in-degree")
    axes[0].set_xlabel("In-degree")
    axes[0].set_ylabel("Số cổ phiếu")
    axes[1].hist(nodes["out_degree"], bins=bins, color=PALETTE[1], alpha=0.85)
    axes[1].set_title("Phân phối out-degree")
    axes[1].set_xlabel("Out-degree")
    for axis, column, color in [(axes[2], "in_degree", PALETTE[0]), (axes[2], "out_degree", PALETTE[1])]:
        values = np.sort(nodes[column].to_numpy())[::-1]
        ccdf = np.arange(1, len(values) + 1) / len(values)
        mask = values > 0
        axis.loglog(values[mask], ccdf[mask], marker="o", ms=3, lw=1, color=color, label=column)
    axes[2].set_title("CCDF (log-log)")
    axes[2].set_xlabel("Bậc")
    axes[2].set_ylabel("P(X ≥ x)")
    axes[2].legend(frameon=False)
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_top_bars(nodes: pd.DataFrame, n: int, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    specs = [
        ("out_degree", "Top out-degree (dẫn dắt)", PALETTE[1]),
        ("in_degree", "Top in-degree (bị ảnh hưởng)", PALETTE[0]),
        ("net_degree", "Top net = out − in (người dẫn dắt ròng)", PALETTE[2]),
    ]
    for axis, (column, label, color) in zip(axes, specs):
        top = nodes[column].sort_values(ascending=False).head(n)[::-1]
        axis.barh(top.index, top.values, color=color)
        axis.set_title(label, fontsize=10)
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_in_out_scatter(nodes: pd.DataFrame, title: str, path: Path, label_n: int = 12) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 6))
    sizes = 20 + 400 * nodes["pagerank"] / nodes["pagerank"].max()
    sc = axis.scatter(
        nodes["in_degree"],
        nodes["out_degree"],
        s=sizes,
        c=nodes["betweenness"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    top = nodes.assign(score=nodes["degree"]).sort_values("score", ascending=False).head(label_n)
    for ticker, row in top.iterrows():
        axis.annotate(ticker, (row["in_degree"], row["out_degree"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    limit = max(nodes["in_degree"].max(), nodes["out_degree"].max()) * 1.05
    axis.plot([0, limit], [0, limit], ls="--", color="grey", lw=0.8)
    axis.set_xlabel("In-degree")
    axis.set_ylabel("Out-degree")
    axis.set_title(title, fontsize=11)
    cbar = fig.colorbar(sc, ax=axis)
    cbar.set_label("Betweenness")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_network(
    graph: nx.DiGraph,
    nodes: pd.DataFrame,
    top_nodes: int,
    max_edges: int,
    title: str,
    path: Path,
    seed: int,
) -> None:
    half = top_nodes // 2
    leaders = nodes.sort_values("out_degree", ascending=False).head(half).index
    followers = nodes.drop(index=leaders).sort_values("in_degree", ascending=False).head(top_nodes - half).index
    chosen = list(leaders) + list(followers)
    sub = graph.subgraph(chosen).copy()
    if sub.number_of_edges() > max_edges:
        ranked = sorted(sub.edges(data="weight"), key=lambda e: e[2], reverse=True)[:max_edges]
        keep = nx.DiGraph()
        keep.add_nodes_from(sub.nodes())
        keep.add_weighted_edges_from(ranked)
        sub = keep
    if sub.number_of_edges() == 0:
        return

    leader_set = set(leaders)
    inner = [n for n in sub.nodes() if n in leader_set]
    outer = [n for n in sub.nodes() if n not in leader_set]
    inner.sort(key=lambda n: -nodes.loc[n, "out_degree"])
    outer.sort(key=lambda n: -nodes.loc[n, "in_degree"])
    pos = {}
    for k, n in enumerate(inner):
        angle = 2 * np.pi * k / max(len(inner), 1)
        pos[n] = (0.42 * np.cos(angle), 0.42 * np.sin(angle))
    for k, n in enumerate(outer):
        angle = 2 * np.pi * (k + 0.5) / max(len(outer), 1)
        pos[n] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))

    net = nodes.loc[list(sub.nodes()), "net_degree"].to_numpy(dtype=float)
    vmax = max(np.abs(net).max(), 1)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sizes = 90 + 900 * nodes.loc[list(sub.nodes()), "out_degree_norm"].to_numpy(dtype=float)

    weights = np.array([d["weight"] for _, _, d in sub.edges(data=True)])
    widths = 0.3 + 1.6 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-12)

    fig, axis = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(
        sub,
        pos,
        ax=axis,
        width=widths,
        edge_color="#8d99ae",
        alpha=0.35,
        arrows=True,
        arrowsize=7,
        connectionstyle="arc3,rad=0.12",
        node_size=sizes,
    )
    nx.draw_networkx_nodes(
        sub,
        pos,
        ax=axis,
        node_size=sizes,
        node_color=net,
        cmap="coolwarm_r",
        vmin=-vmax,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(sub, pos, ax=axis, font_size=6.5)
    sm = cm.ScalarMappable(norm=norm, cmap="coolwarm_r")
    cbar = fig.colorbar(sm, ax=axis, shrink=0.6)
    cbar.set_label("Net degree = out − in (xanh: dẫn dắt, đỏ: bị dẫn dắt)")
    axis.set_title(title, fontsize=12)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_adjacency(adjacency: np.ndarray, tickers: List[str], nodes: pd.DataFrame, title: str, path: Path) -> None:
    order = nodes.loc[tickers].sort_values(["community", "out_degree"], ascending=[True, False]).index
    index = [tickers.index(t) for t in order]
    ordered = adjacency[np.ix_(index, index)]
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(ordered, cmap="Greys", interpolation="nearest", aspect="equal")
    bounds = nodes.loc[order, "community"].to_numpy()
    cuts = np.nonzero(np.diff(bounds))[0] + 0.5
    for cut in cuts:
        axis.axhline(cut, color="#e07a5f", lw=0.5)
        axis.axvline(cut, color="#e07a5f", lw=0.5)
    axis.set_xlabel("Cổ phiếu bị ảnh hưởng (target)")
    axis.set_ylabel("Cổ phiếu dẫn dắt (source)")
    axis.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_period_comparison(summary: pd.DataFrame, path: Path, rule_label: str = "") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    ticks = summary["start"].str[:4] + "–" + summary["end"].str[2:4]
    specs = [
        ("density", "Mật độ mạng", PALETTE[0], None),
        ("edges", "Số cạnh", PALETTE[1], None),
        ("reciprocity", "Độ tương hỗ (reciprocity)", PALETTE[2], None),
        ("giant_scc_share", "Tỷ trọng SCC lớn nhất", PALETTE[3], (0, 1.05)),
        ("avg_clustering", "Hệ số cụm trung bình", PALETTE[4], None),
        ("modularity", "Modularity (Louvain)", PALETTE[5], None),
    ]
    for axis, (column, label, color, limits) in zip(axes.ravel(), specs):
        axis.plot(ticks, summary[column], marker="o", color=color, lw=2)
        axis.set_title(label, fontsize=10)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(alpha=0.25)
        if limits:
            axis.set_ylim(*limits)
    if rule_label:
        fig.suptitle(f"So sánh cấu trúc mạng theo giai đoạn: {rule_label}", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_leader_persistence(table: pd.DataFrame, path: Path, top: int = 20) -> None:
    data = table.head(top)[::-1]
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.barh(data.index, data["periods_in_top"], color=PALETTE[1])
    axis.set_xlabel("Số giai đoạn nằm trong top out-degree")
    axis.set_title("Mức độ bền vững của nhóm dẫn dắt", fontsize=11)
    axis.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _sector_colors(sectors: List[str]) -> Dict[str, str]:
    cmap = plt.get_cmap("tab20")
    return {name: (cmap(i) if i < 20 else "#222222") for i, name in enumerate(sorted(set(sectors)))}


def plot_xy(frame: pd.DataFrame, title: str, path: Path, label_n: int = 14) -> None:
    colors_map = _sector_colors(list(frame["sector"]))
    fig, axis = plt.subplots(figsize=(10, 7.5))
    for name, group in frame.groupby("sector"):
        axis.scatter(group["Y"], group["X"], s=38, color=colors_map[name], label=f"{name} ({len(group)})", edgecolor="white", linewidth=0.4, alpha=0.9)
    for ticker, row in frame.sort_values("Z").head(label_n).iterrows():
        axis.annotate(ticker, (row["Y"], row["X"]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Y: chỉ số khoảng cách (thấp = gần tâm)")
    axis.set_ylabel("X: chỉ số kết nối (thấp = nhiều kết nối)")
    axis.set_title(title, fontsize=11)
    axis.invert_xaxis()
    axis.invert_yaxis()
    axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sector_lead(summary: pd.DataFrame, title: str, path: Path) -> None:
    data = summary.sort_values("lead_minus_follow")
    fig, axis = plt.subplots(figsize=(9, 7))
    y = np.arange(len(data))
    axis.barh(y + 0.2, data["lead_ratio"], height=0.4, color=PALETTE[1], label="Tỷ lệ dẫn dắt (cạnh ra / tỷ trọng số mã)")
    axis.barh(y - 0.2, data["follow_ratio"], height=0.4, color=PALETTE[0], label="Tỷ lệ bị dẫn dắt (cạnh vào / tỷ trọng số mã)")
    axis.axvline(1, color="grey", ls="--", lw=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(data["sector"], fontsize=8)
    axis.set_title(title, fontsize=11)
    axis.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sector_flow(lift: pd.DataFrame, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 8.5))
    values = lift.to_numpy(dtype=float)
    image = axis.imshow(values, cmap="RdBu_r", norm=colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2.0, np.nanmax(values))))
    axis.set_xticks(range(len(lift.columns)))
    axis.set_yticks(range(len(lift.index)))
    axis.set_xticklabels(lift.columns, rotation=75, ha="right", fontsize=7)
    axis.set_yticklabels(lift.index, fontsize=7)
    axis.set_xlabel("Ngành bị ảnh hưởng (target)")
    axis.set_ylabel("Ngành dẫn dắt (source)")
    axis.set_title(title, fontsize=11)
    cbar = fig.colorbar(image, ax=axis, shrink=0.7)
    cbar.set_label("Lift = mật độ cạnh giữa hai ngành / mật độ toàn mạng")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = table["transition"]
    x = np.arange(len(labels))
    for axis, col, name in [(axes[0], "vol", "Biến động năm hóa của danh mục (kỳ sau)"), (axes[1], "corr", "Tương quan cặp trung bình (kỳ sau)")]:
        axis.bar(x - 0.27, table[f"core_{col}"], width=0.27, color=PALETTE[1], label="Lõi (Z thấp)")
        axis.bar(x, table[f"random_{col}"], width=0.27, color="#adb5bd", label="Ngẫu nhiên")
        axis.bar(x + 0.27, table[f"periphery_{col}"], width=0.27, color=PALETTE[2], label="Ngoại vi (Z cao)")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_title(name, fontsize=10)
        axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
