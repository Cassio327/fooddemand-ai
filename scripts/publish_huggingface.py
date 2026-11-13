from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install the publish extra: pip install -e '.[publish]'") from exc

    parser = argparse.ArgumentParser(description="Publish the trained artifact and model card.")
    parser.add_argument("--repo-id", required=True, help="Example: Cassio327/fooddemand-ai")
    parser.add_argument("--model", type=Path, default=Path("models/fooddemand.joblib"))
    parser.add_argument("--card", type=Path, default=Path("reports/MODEL_CARD.md"))
    parser.add_argument("--report", type=Path, default=Path("reports/latest_metrics.json"))
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    if not args.model.exists() or not args.card.exists() or not args.report.exists():
        raise SystemExit("Train the project before publishing")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("data_source") != "kaggle-store-sales":
        raise SystemExit(
            "Publishing blocked: only verified Kaggle runs may be published. "
            "Set --data-source kaggle-store-sales when running against official data."
        )

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(args.model),
        path_in_repo="fooddemand.joblib",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(args.card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Published to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
