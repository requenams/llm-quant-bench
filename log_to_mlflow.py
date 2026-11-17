"""Log existing benchmark results to MLflow.

Example:
    python log_to_mlflow.py --results results
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # then open http://127.0.0.1:5000
"""

import argparse

from quantbench.tracking import log_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tracking-uri", default=None,
                    help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    args = ap.parse_args()
    ids = log_folder(args.results, args.tracking_uri)
    for variant, run_id in ids.items():
        print(f"{variant}: run {run_id}")


if __name__ == "__main__":
    main()
