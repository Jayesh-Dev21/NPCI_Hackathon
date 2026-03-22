from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from webui.features import FeatureBuilder
from webui.model_bridge import ApiModelPredictor, LocalModelPredictor


DEFAULT_FILE_CSV = "r4.2/file.csv"
DEFAULT_HTTP_CSV = "r4.2/http.csv"


@dataclass
class AppConfig:
    model_source: str
    local_model_path: str
    api_endpoint: str
    api_key: str
    file_csv: str
    http_csv: str


def load_config() -> AppConfig:
    return AppConfig(
        model_source=os.getenv("MODEL_SOURCE", "local"),
        local_model_path=os.getenv("MODEL_PATH", "models/xgboost.pkl"),
        api_endpoint=os.getenv("MODEL_API_ENDPOINT", "http://localhost:8000/predict"),
        api_key=os.getenv("MODEL_API_KEY", ""),
        file_csv=os.getenv("FILE_CSV", DEFAULT_FILE_CSV),
        http_csv=os.getenv("HTTP_CSV", DEFAULT_HTTP_CSV),
    )


def _build_predictor(cfg: AppConfig):
    if cfg.model_source == "api":
        return ApiModelPredictor(endpoint=cfg.api_endpoint, api_key=cfg.api_key or None)
    return LocalModelPredictor(model_path=cfg.local_model_path)


def _make_model_input(features: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {"user", "day"}
    cols = [c for c in features.columns if c not in drop_cols]
    return features[cols].copy()


def _fallback_scores(features: pd.DataFrame) -> np.ndarray:
    z = (
        0.35 * features.get("delete_ratio", 0)
        + 0.35 * features.get("suspicious_ratio", 0)
        + 0.30 * (features.get("total_activity", 0) / (features.get("total_activity", 0).max() + 1e-6))
    )
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    return z


def main() -> None:
    st.set_page_config(page_title="NPCI Insider Threat UI", page_icon="🛡️", layout="wide")

    cfg = load_config()

    st.title("NPCI Insider Threat Detection")
    st.caption("Web UI linked to your ML model for user-day risk scoring.")

    with st.sidebar:
        st.subheader("Configuration")
        model_source = st.selectbox("Model Source", ["local", "api"], index=0 if cfg.model_source == "local" else 1)
        local_model_path = st.text_input("Local Model Path", value=cfg.local_model_path)
        api_endpoint = st.text_input("API Endpoint", value=cfg.api_endpoint)
        api_key = st.text_input("API Key (optional)", value=cfg.api_key, type="password")
        file_csv = st.text_input("file.csv path", value=cfg.file_csv)
        http_csv = st.text_input("http.csv path", value=cfg.http_csv)
        run = st.button("Run Scoring", type="primary")

    if not run:
        st.info("Set configuration and click Run Scoring.")
        return

    try:
        with st.spinner("Building features from logs..."):
            feature_builder = FeatureBuilder(file_csv=file_csv, http_csv=http_csv)
            features = feature_builder.build_user_day_features()

        st.success(f"Built {len(features):,} user-day rows.")

        model_input = _make_model_input(features)

        used_fallback = False
        try:
            predictor = (
                ApiModelPredictor(endpoint=api_endpoint, api_key=api_key or None)
                if model_source == "api"
                else LocalModelPredictor(model_path=local_model_path)
            )
            scores = predictor.predict_scores(model_input)
        except Exception as model_error:
            used_fallback = True
            st.warning(
                "Model unavailable. Using heuristic fallback scores for UI preview. "
                f"Details: {model_error}"
            )
            scores = _fallback_scores(features)

        result = features[["user", "day"]].copy()
        result["risk_score"] = np.asarray(scores, dtype=float)
        result["risk_score"] = np.clip(result["risk_score"], 0.0, 1.0)

        threshold = st.slider("Alert threshold", min_value=0.1, max_value=0.99, value=0.7, step=0.01)
        result["alert"] = (result["risk_score"] >= threshold).astype(int)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Scored", f"{len(result):,}")
        with col2:
            st.metric("Alerts", int(result["alert"].sum()))
        with col3:
            st.metric("Mean Risk", f"{result['risk_score'].mean():.3f}")

        st.subheader("Top Risky User-Day Alerts")
        top = result.sort_values("risk_score", ascending=False).head(200)
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.subheader("Risk Trend")
        trend = result.groupby("day", as_index=False)["risk_score"].mean()
        st.line_chart(trend.set_index("day"))

        st.subheader("Per-User Drilldown")
        selected_user = st.selectbox("Select user", sorted(result["user"].astype(str).unique().tolist()))
        user_view = result[result["user"].astype(str) == selected_user].sort_values("day")
        st.dataframe(user_view, use_container_width=True, hide_index=True)
        st.area_chart(user_view.set_index("day")["risk_score"])

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download scored results CSV",
            data=csv_bytes,
            file_name="scored_user_day_risk.csv",
            mime="text/csv",
        )

        if not used_fallback:
            st.success("Scoring completed using your ML model.")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
