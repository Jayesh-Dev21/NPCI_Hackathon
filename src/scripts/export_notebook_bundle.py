from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib


def _extract_feature_cols(notebook_text: str) -> list[str] | None:
    # Extract likely feature names from ROLL_COLS and known engineered columns.
    # This is a best-effort helper for integration and can be overridden manually.
    roll_match = re.search(r'ROLL_COLS\s*=\s*\[(.*?)\]', notebook_text, flags=re.DOTALL)
    cols: list[str] = []
    if roll_match:
        raw = roll_match.group(1)
        cols.extend(re.findall(r'"([^"]+)"', raw))

    extras = [
        "logon_count",
        "unique_pcs",
        "logon_after_hours",
        "logon_weekend",
        "logon_mean_hour",
        "logon_std_hour",
        "file_count",
        "unique_files",
        "file_after_hours",
        "file_copy",
        "file_delete",
        "file_write",
        "device_count",
        "device_after_hours",
        "usb_connect",
        "email_count",
        "email_external",
        "email_after_hours",
        "email_attachments",
        "http_count",
        "http_after_hours",
        "unique_urls",
        "http_suspicious",
        "total_activity",
        "logon_ah_ratio",
        "email_ext_ratio",
        "file_del_ratio",
        "file_copy_ratio",
        "usb_per_logon",
        "http_sus_ratio",
        "day_of_week",
        "month",
        "is_weekend",
    ]
    for c in extras:
        if c not in cols:
            cols.append(c)

    return cols if cols else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create web-compatible model bundle from trained models")
    parser.add_argument("--xgb", required=True, help="Path to trained XGBoost model (.joblib/.pkl)")
    parser.add_argument("--lgb", required=True, help="Path to trained LightGBM model (.joblib/.pkl)")
    parser.add_argument("--output", default="models/notebook_ensemble_bundle.joblib", help="Output bundle path")
    parser.add_argument("--threshold", type=float, default=0.05, help="Alert threshold from notebook")
    parser.add_argument("--xgb-weight", type=float, default=0.55, help="Ensemble weight for XGB")
    parser.add_argument("--lgb-weight", type=float, default=0.45, help="Ensemble weight for LGB")
    parser.add_argument(
        "--feature-cols-json",
        default="",
        help="Optional JSON list file containing exact feature column order",
    )
    parser.add_argument(
        "--notebook",
        default="main copy.ipynb",
        help="Notebook path for best-effort feature extraction",
    )
    args = parser.parse_args()

    xgb_model = joblib.load(args.xgb)
    lgb_model = joblib.load(args.lgb)

    feature_cols = None
    if args.feature_cols_json:
        feature_cols = json.loads(Path(args.feature_cols_json).read_text(encoding="utf-8"))
    else:
        nb_path = Path(args.notebook)
        if nb_path.exists():
            feature_cols = _extract_feature_cols(nb_path.read_text(encoding="utf-8"))

    bundle = {
        "xgb_model": xgb_model,
        "lgb_model": lgb_model,
        "weights": {"xgb": args.xgb_weight, "lgb": args.lgb_weight},
        "threshold": args.threshold,
        "feature_cols": feature_cols,
        "source": "main copy.ipynb",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Wrote bundle: {out_path}")
    if feature_cols:
        print(f"Feature cols attached: {len(feature_cols)}")
    else:
        print("Feature cols not attached; backend will use runtime dataframe columns.")


if __name__ == "__main__":
    main()
