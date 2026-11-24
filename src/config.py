from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str
    label: str


DEFAULT_PERIODS: Tuple[Period, ...] = (
    Period("P1", "2010-01-01", "2013-12-31", "2010-2013 Hậu khủng hoảng, tái cơ cấu"),
    Period("P2", "2014-01-01", "2017-12-31", "2014-2017 Phục hồi và tăng trưởng"),
    Period("P3", "2018-01-01", "2019-12-31", "2018-2019 Chiến tranh thương mại"),
    Period("P4", "2020-01-01", "2021-12-31", "2020-2021 COVID-19"),
    Period("P5", "2022-01-01", "2023-12-31", "2022-2023 Thị trường gấu, trái phiếu"),
    Period("P6", "2024-01-01", "2026-04-02", "2024-2026 Phục hồi"),
)


@dataclass
class Settings:
    data_path: Path = Path("data/hose_stock_data.csv")
    output_dir: Path = Path("outputs")
    periods: Tuple[Period, ...] = DEFAULT_PERIODS
    min_coverage: float = 0.95
    max_zero_share: float = 0.30
    winsor_abs: float = 0.10
    max_lag: int = 10
    fixed_lag: Optional[int] = None
    lag_sample_size: int = 30
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"
    edge_rules: Tuple[str, ...] = ("raw05", "fdr05")
    sensitivity_lags: Tuple[int, ...] = (1, 2, 3, 5)
    seed: int = 42
    top_n: int = 15
    graph_top_nodes: int = 50
    graph_max_edges: int = 350
    validation_pairs: int = 25
    extra: dict = field(default_factory=dict)
