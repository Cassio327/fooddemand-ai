from __future__ import annotations

import argparse
import json
from pathlib import Path

from fooddemand.data import load_competition_data


def build_summary(data_dir: str | Path, max_rows: int | None = None) -> dict:
    data = load_competition_data(data_dir, max_rows=max_rows)
    sales = data.sales.copy()
    return {
        "rows": int(len(sales)),
        "date_start": str(sales["date"].min().date()),
        "date_end": str(sales["date"].max().date()),
        "stores": int(sales["store_nbr"].nunique()),
        "families": int(sales["family"].nunique()),
        "zero_sales_share": float((sales["sales"] == 0).mean()),
        "promotion_share": float((sales["onpromotion"] > 0).mean()),
        "sales_mean": float(sales["sales"].mean()),
        "sales_median": float(sales["sales"].median()),
    }


def save_figures(data_dir: str | Path, output_dir: str | Path, max_rows: int | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    data = load_competition_data(data_dir, max_rows=max_rows)
    sales = data.sales.copy()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    daily = sales.groupby("date", as_index=False)["sales"].sum()
    fig = plt.figure(figsize=(10, 4))
    plt.plot(daily["date"], daily["sales"])
    plt.title("Total sales over time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    fig.savefig(output / "daily_sales.png", dpi=150)
    plt.close(fig)

    top = sales.groupby("family")["sales"].sum().nlargest(10).sort_values()
    fig = plt.figure(figsize=(8, 5))
    top.plot(kind="barh")
    plt.title("Top 10 product families by total sales")
    plt.xlabel("Sales")
    plt.tight_layout()
    fig.savefig(output / "top_families.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA summary and portfolio figures.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/eda_summary.json"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args.data_dir, max_rows=args.max_rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_figures(args.data_dir, args.figures_dir, max_rows=args.max_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
