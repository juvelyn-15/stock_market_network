import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import Settings
from centrality_pipeline import run_centrality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "hose_stock_data.csv"))
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--sectors", default=str(ROOT / "data" / "sector_map.csv"))
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    settings = Settings(data_path=Path(args.data), output_dir=Path(args.output))
    run_centrality(settings, Path(args.sectors), args.min_nodes, args.top_k)


if __name__ == "__main__":
    main()
