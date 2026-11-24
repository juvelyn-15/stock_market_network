import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import DEFAULT_PERIODS, Period, Settings
from pipeline import run_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "hose_stock_data.csv"))
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--name", default="CUSTOM")
    parser.add_argument("--lag", type=int)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--max-zero-share", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    periods = DEFAULT_PERIODS
    if args.start and args.end:
        periods = (Period(args.name, args.start, args.end, f"{args.start} den {args.end}"),)
    settings = Settings(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        periods=periods,
        fixed_lag=args.lag,
        max_lag=args.max_lag,
        alpha=args.alpha,
        min_coverage=args.min_coverage,
        max_zero_share=args.max_zero_share,
        seed=args.seed,
    )
    run_all(settings)


if __name__ == "__main__":
    main()
