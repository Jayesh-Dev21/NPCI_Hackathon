from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from webui.features import FeatureBuilder
from webui.model_bridge import ApiModelPredictor, LocalModelPredictor
from webui.analytics import get_analytics
from webui.gemini_explainer import GeminiExplainer


DEFAULT_FILE_CSV = "r4.2/file.csv"
DEFAULT_HTTP_CSV = "r4.2/http.csv"
DEFAULT_LOGON_CSV = "r4.2/logon.csv"
DEFAULT_DEVICE_CSV = "r4.2/device.csv"
DEFAULT_EMAIL_CSV = "r4.2/email.csv"
DEFAULT_PSYCHOMETRIC_CSV = "r4.2/psychometric.csv"
FALLBACK_FILE_CSV = "data/r4.2/file.csv"
FALLBACK_HTTP_CSV = "data/r4.2/http.csv"
FALLBACK_LOGON_CSV = "data/r4.2/logon.csv"
FALLBACK_DEVICE_CSV = "data/r4.2/device.csv"
FALLBACK_EMAIL_CSV = "data/r4.2/email.csv"
FALLBACK_PSYCHOMETRIC_CSV = "data/r4.2/psychometric.csv"


class ScoreRequest(BaseModel):
    model_source: str = Field(default="local", pattern="^(local|api)$")
    local_model_path: str = "models/xgboost.pkl"
    api_endpoint: str = "http://localhost:8000/predict"
    api_key: str | None = None
    file_csv: str = DEFAULT_FILE_CSV
    http_csv: str = DEFAULT_HTTP_CSV
    file_nrows: int | None = 200000
    http_nrows: int | None = 200000
    threshold: float = 0.7
    strict_model_features: bool = False


class UserActivityRequest(BaseModel):
    user: str
    source: str = Field(default="http", pattern="^(http|file|logon|device|email|psychometric)$")
    max_rows: int = 500
    file_csv: str = DEFAULT_FILE_CSV
    http_csv: str = DEFAULT_HTTP_CSV
    logon_csv: str = DEFAULT_LOGON_CSV
    device_csv: str = DEFAULT_DEVICE_CSV
    email_csv: str = DEFAULT_EMAIL_CSV
    psychometric_csv: str = DEFAULT_PSYCHOMETRIC_CSV


def _resolve_source_path(req: UserActivityRequest) -> str:
    source_to_path = {
        "http": req.http_csv,
        "file": req.file_csv,
        "logon": req.logon_csv,
        "device": req.device_csv,
        "email": req.email_csv,
        "psychometric": req.psychometric_csv,
    }
    return source_to_path[req.source]


def _resolve_data_path(preferred: str, fallback: str) -> str:
    p_preferred = Path(preferred)
    if p_preferred.exists():
        return str(p_preferred)
    p_fallback = Path(fallback)
    if p_fallback.exists():
        return str(p_fallback)
    return preferred


def _read_csv_columns(path: str) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _pick_usecols(header: list[str], source: str) -> list[str]:
    base = ["id", "date", "user", "pc"]
    source_specific = {
        "http": ["url", "content"],
        "file": ["filename", "activity", "content"],
        "logon": ["activity"],
        "device": ["activity"],
        "email": ["to", "cc", "bcc", "from", "size", "attachments", "content"],
        "psychometric": ["user_id", "O", "C", "E", "A", "N"],
    }
    wanted = base + source_specific.get(source, [])
    return [c for c in wanted if c in header]


def _safe_iso_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return str(value)


def _collect_users_from_csv(path: str, max_users: int = 1000) -> set[str]:
    users: set[str] = set()
    header = _read_csv_columns(path)
    user_col = "user" if "user" in header else ("user_id" if "user_id" in header else None)
    if user_col is None:
        return users

    for chunk in pd.read_csv(path, usecols=[user_col], chunksize=200000):
        users.update(
            chunk[user_col].dropna().astype(str).str.strip().str.upper().tolist()
        )
        if len(users) >= max_users:
            break
    return users


def _stream_user_rows(path: str, user: str, source: str, max_rows: int) -> list[dict[str, Any]]:
    max_rows = max(1, min(max_rows, 5000))
    header = _read_csv_columns(path)
    usecols = _pick_usecols(header, source)
    user_norm = user.strip().lower()

    out: list[dict[str, Any]] = []
    chunk_iter = pd.read_csv(path, usecols=usecols, chunksize=200000)
    for chunk in chunk_iter:
        user_col = "user" if "user" in chunk.columns else ("user_id" if "user_id" in chunk.columns else None)
        if user_col is None:
            break
        mask = chunk[user_col].astype(str).str.strip().str.lower() == user_norm
        if not mask.any():
            continue
        matched = chunk.loc[mask].copy()
        if "user_id" in matched.columns and "user" not in matched.columns:
            matched["user"] = matched["user_id"]

        if "date" in matched.columns:
            matched["date"] = pd.to_datetime(matched["date"], errors="coerce")
            matched = matched.sort_values("date", ascending=False)
            matched["date"] = matched["date"].map(_safe_iso_date)

        records = matched.to_dict(orient="records")
        out.extend(records)
        if len(out) >= max_rows:
            break

    return out[:max_rows]


def _fallback_scores(features: pd.DataFrame) -> np.ndarray:
    delete_ratio = np.asarray(features.get("delete_ratio", 0.0), dtype=float)
    suspicious_ratio = np.asarray(features.get("suspicious_ratio", 0.0), dtype=float)
    total_activity = np.asarray(features.get("total_activity", 0.0), dtype=float)
    denom = float(np.max(total_activity)) + 1e-6

    z = (
        0.35 * delete_ratio
        + 0.35 * suspicious_ratio
        + 0.30 * (total_activity / denom)
    )
    return np.clip(np.asarray(z, dtype=float), 0.0, 1.0)


def _predict_with_model(req: ScoreRequest, model_input: pd.DataFrame) -> tuple[np.ndarray, bool, str | None]:
    if req.model_source == "api":
        predictor = ApiModelPredictor(endpoint=req.api_endpoint, api_key=req.api_key)
    else:
        predictor = LocalModelPredictor(model_path=req.local_model_path)

    scores = predictor.predict_scores(model_input)
    return np.asarray(scores, dtype=float), False, None


def create_app() -> FastAPI:
    app = FastAPI(title="NPCI Insider Threat UI", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html_path = CURRENT_DIR / "templates" / "overview.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="Overview template missing")
        return html_path.read_text(encoding="utf-8")

    @app.get("/data-intelligence", response_class=HTMLResponse)
    def data_intelligence() -> str:
        html_path = CURRENT_DIR / "templates" / "data_intelligence.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="Data Intelligence template missing")
        return html_path.read_text(encoding="utf-8")

    @app.get("/model-performance", response_class=HTMLResponse)
    def model_performance() -> str:
        html_path = CURRENT_DIR / "templates" / "model_performance.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="Model Performance template missing")
        return html_path.read_text(encoding="utf-8")

    @app.get("/threat-investigator", response_class=HTMLResponse)
    def threat_investigator() -> str:
        html_path = CURRENT_DIR / "templates" / "threat_investigator.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="Threat Investigator template missing")
        return html_path.read_text(encoding="utf-8")

    @app.get("/users", response_class=HTMLResponse)
    def users_page() -> str:
        html_path = CURRENT_DIR / "templates" / "users.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="Users UI template missing")
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/users")
    def users_list(
        file_csv: str = DEFAULT_FILE_CSV,
        http_csv: str = DEFAULT_HTTP_CSV,
        logon_csv: str = DEFAULT_LOGON_CSV,
        device_csv: str = DEFAULT_DEVICE_CSV,
        email_csv: str = DEFAULT_EMAIL_CSV,
        psychometric_csv: str = DEFAULT_PSYCHOMETRIC_CSV,
        include_http: bool = False,
    ) -> JSONResponse:
        file_csv = _resolve_data_path(file_csv, FALLBACK_FILE_CSV)
        http_csv = _resolve_data_path(http_csv, FALLBACK_HTTP_CSV)
        logon_csv = _resolve_data_path(logon_csv, FALLBACK_LOGON_CSV)
        device_csv = _resolve_data_path(device_csv, FALLBACK_DEVICE_CSV)
        email_csv = _resolve_data_path(email_csv, FALLBACK_EMAIL_CSV)
        psychometric_csv = _resolve_data_path(psychometric_csv, FALLBACK_PSYCHOMETRIC_CSV)

        users: set[str] = set()

        source_paths = [psychometric_csv, logon_csv, file_csv, device_csv, email_csv]
        if include_http:
            source_paths.append(http_csv)

        for path in source_paths:
            p = Path(path)
            if not p.exists():
                continue
            try:
                users.update(_collect_users_from_csv(path, max_users=1000))
                if len(users) >= 1000:
                    break
            except Exception:
                continue

        sorted_users = sorted([u for u in users if u])
        return JSONResponse({"users": sorted_users, "count": len(sorted_users)})

    @app.post("/api/user-activity")
    def user_activity(req: UserActivityRequest) -> JSONResponse:
        # Prefer newly extracted ./data/r4.2 files, fallback to older ./data/raw/r4.2.
        if req.file_csv == DEFAULT_FILE_CSV:
            req.file_csv = _resolve_data_path(DEFAULT_FILE_CSV, FALLBACK_FILE_CSV)
        if req.http_csv == DEFAULT_HTTP_CSV:
            req.http_csv = _resolve_data_path(DEFAULT_HTTP_CSV, FALLBACK_HTTP_CSV)
        if req.logon_csv == DEFAULT_LOGON_CSV:
            req.logon_csv = _resolve_data_path(DEFAULT_LOGON_CSV, FALLBACK_LOGON_CSV)
        if req.device_csv == DEFAULT_DEVICE_CSV:
            req.device_csv = _resolve_data_path(DEFAULT_DEVICE_CSV, FALLBACK_DEVICE_CSV)
        if req.email_csv == DEFAULT_EMAIL_CSV:
            req.email_csv = _resolve_data_path(DEFAULT_EMAIL_CSV, FALLBACK_EMAIL_CSV)
        if req.psychometric_csv == DEFAULT_PSYCHOMETRIC_CSV:
            req.psychometric_csv = _resolve_data_path(DEFAULT_PSYCHOMETRIC_CSV, FALLBACK_PSYCHOMETRIC_CSV)

        path = _resolve_source_path(req)
        if not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"Source file not found: {path}")

        try:
            rows = _stream_user_rows(path=path, user=req.user, source=req.source, max_rows=req.max_rows)
            return JSONResponse(
                {
                    "user": req.user.strip().upper(),
                    "source": req.source,
                    "count": len(rows),
                    "rows": rows,
                }
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch user activity: {e}")

    @app.post("/api/score")
    def score(req: ScoreRequest) -> JSONResponse:
        try:
            if req.file_csv == DEFAULT_FILE_CSV:
                req.file_csv = _resolve_data_path(DEFAULT_FILE_CSV, FALLBACK_FILE_CSV)
            if req.http_csv == DEFAULT_HTTP_CSV:
                req.http_csv = _resolve_data_path(DEFAULT_HTTP_CSV, FALLBACK_HTTP_CSV)

            feature_builder = FeatureBuilder(
                file_csv=req.file_csv,
                http_csv=req.http_csv,
                file_nrows=req.file_nrows,
                http_nrows=req.http_nrows,
            )
            features = feature_builder.build_user_day_features()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature build failed: {e}")

        model_input = features[[c for c in features.columns if c not in {"user", "day"}]].copy()

        used_fallback = False
        fallback_reason: str | None = None
        model_features_used: list[str] = list(model_input.columns)
        model_threshold_used = req.threshold
        try:
            if req.model_source == "api":
                predictor = ApiModelPredictor(endpoint=req.api_endpoint, api_key=req.api_key)
                scores = predictor.predict_scores(model_input)
            else:
                predictor = LocalModelPredictor(model_path=req.local_model_path)
                expected_cols = predictor.get_feature_cols()
                if expected_cols:
                    aligned = model_input.reindex(columns=expected_cols)
                    if aligned.isnull().any().any():
                        missing = [
                            c for c in expected_cols if c not in model_input.columns
                        ]
                        if req.strict_model_features:
                            raise ValueError(
                                "Missing required model features: " + ", ".join(missing)
                            )
                        aligned = aligned.fillna(0.0)
                    model_input = aligned
                    model_features_used = expected_cols

                saved_threshold = predictor.get_threshold()
                if saved_threshold is not None:
                    model_threshold_used = float(saved_threshold)

                scores = predictor.predict_scores(model_input)
        except Exception as e:
            used_fallback = True
            fallback_reason = str(e)
            scores = _fallback_scores(features)

        result = features[["user", "day"]].copy()
        result["risk_score"] = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        result["alert"] = (result["risk_score"] >= model_threshold_used).astype(int)

        top = result.sort_values("risk_score", ascending=False).head(200)
        trend = result.groupby("day", as_index=False)["risk_score"].mean()
        trend_alerts = (
            result.groupby("day", as_index=False)
            .agg(mean_risk=("risk_score", "mean"), alert_count=("alert", "sum"))
            .sort_values("day")
        )

        response: dict[str, Any] = {
            "metrics": {
                "rows_scored": int(len(result)),
                "alerts": int(result["alert"].sum()),
                "mean_risk": float(result["risk_score"].mean()),
            },
            "threshold_used": float(model_threshold_used),
            "model_features_used": model_features_used,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "top_alerts": top.assign(day=top["day"].astype(str)).to_dict(orient="records"),
            "trend": trend.assign(day=trend["day"].astype(str)).to_dict(orient="records"),
            "trend_alerts": trend_alerts.assign(day=trend_alerts["day"].astype(str)).to_dict(orient="records"),
        }
        return JSONResponse(response)

    @app.get("/api/model-metrics")
    def model_metrics() -> JSONResponse:
        """Return real model performance metrics from notebook evaluation."""
        try:
            analytics = get_analytics()
            metrics = analytics.get_model_metrics()
            return JSONResponse(metrics)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")

    @app.get("/api/roc-curve")
    def roc_curve() -> JSONResponse:
        """Return ROC curve data for all models."""
        try:
            analytics = get_analytics()
            roc_data = analytics.generate_roc_curve_data()
            return JSONResponse(roc_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate ROC curve: {e}")

    @app.get("/api/pr-curve")
    def pr_curve() -> JSONResponse:
        """Return Precision-Recall curve data for all models."""
        try:
            analytics = get_analytics()
            pr_data = analytics.generate_pr_curve_data()
            return JSONResponse(pr_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate PR curve: {e}")

    @app.post("/api/user-timeline")
    def user_timeline(req: ScoreRequest) -> JSONResponse:
        """
        Generate risk timeline for a specific user using real model predictions.
        """
        try:
            # Build features for this user
            if req.file_csv == DEFAULT_FILE_CSV:
                req.file_csv = _resolve_data_path(DEFAULT_FILE_CSV, FALLBACK_FILE_CSV)
            if req.http_csv == DEFAULT_HTTP_CSV:
                req.http_csv = _resolve_data_path(DEFAULT_HTTP_CSV, FALLBACK_HTTP_CSV)

            feature_builder = FeatureBuilder(
                file_csv=req.file_csv,
                http_csv=req.http_csv,
                file_nrows=req.file_nrows,
                http_nrows=req.http_nrows,
            )
            features = feature_builder.build_user_day_features()
            
            # Get predictions from real model
            analytics = get_analytics()
            model_features = features[[c for c in features.columns if c not in {"user", "day"}]]
            
            # Use ensemble predictions
            risk_scores = analytics.predict_ensemble(model_features)
            
            # Attach scores to user-day pairs
            result = features[["user", "day"]].copy()
            result["risk_score"] = risk_scores
            
            # Group by user
            user_timelines = {}
            for user in result["user"].unique():
                user_data = result[result["user"] == user].sort_values("day")
                user_timelines[user] = {
                    "dates": user_data["day"].astype(str).tolist(),
                    "risk_scores": user_data["risk_score"].tolist(),
                    "max_risk": float(user_data["risk_score"].max()),
                    "mean_risk": float(user_data["risk_score"].mean()),
                    "total_days": len(user_data)
                }
            
            return JSONResponse({"timelines": user_timelines})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Timeline generation failed: {e}")

    @app.get("/api/investigate-user/{user_id}")
    def investigate_user(user_id: str) -> JSONResponse:
        """
        Get detailed investigation data for a specific user including timeline and top incidents.
        """
        try:
            # Build features
            file_csv = _resolve_data_path(DEFAULT_FILE_CSV, FALLBACK_FILE_CSV)
            http_csv = _resolve_data_path(DEFAULT_HTTP_CSV, FALLBACK_HTTP_CSV)
            
            feature_builder = FeatureBuilder(
                file_csv=file_csv,
                http_csv=http_csv,
                file_nrows=200000,
                http_nrows=200000,
            )
            features = feature_builder.build_user_day_features()
            
            # Filter to this user
            user_features = features[features["user"].str.upper() == user_id.upper()]
            
            if len(user_features) == 0:
                return JSONResponse({
                    "error": f"User {user_id} not found in dataset",
                    "found": False
                })
            
            # Get predictions
            analytics = get_analytics()
            model_features = user_features[[c for c in user_features.columns if c not in {"user", "day"}]]
            risk_scores = analytics.predict_ensemble(model_features)
            
            # Build timeline
            timeline_data = user_features[["user", "day"]].copy()
            timeline_data["risk_score"] = risk_scores
            timeline_data = timeline_data.sort_values("day")
            
            # Get top risk incidents
            top_incidents = timeline_data.nlargest(5, "risk_score")
            
            # Compute SHAP for highest risk day
            top_idx = timeline_data["risk_score"].idxmax()
            top_day_features = model_features.loc[[top_idx]]
            
            shap_attributions = analytics.get_top_shap_features(top_day_features, sample_idx=0)
            
            response = {
                "found": True,
                "user": user_id.upper(),
                "timeline": {
                    "dates": timeline_data["day"].astype(str).tolist(),
                    "risk_scores": timeline_data["risk_score"].tolist()
                },
                "stats": {
                    "total_days": len(timeline_data),
                    "max_risk": float(timeline_data["risk_score"].max()),
                    "mean_risk": float(timeline_data["risk_score"].mean()),
                    "high_risk_days": int((timeline_data["risk_score"] > 0.7).sum())
                },
                "top_incidents": [
                    {
                        "date": str(row["day"]),
                        "risk_score": float(row["risk_score"])
                    }
                    for _, row in top_incidents.iterrows()
                ],
                "top_incident_shap": shap_attributions[:10] if shap_attributions else []
            }
            
            return JSONResponse(response)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Investigation failed: {e}")

    @app.get("/api/explain-incident/{user_id}/{date}")
    def explain_incident(user_id: str, date: str) -> JSONResponse:
        """
        Get AI-powered natural language explanation for a specific user-day incident.
        """
        try:
            # Build features
            file_csv = _resolve_data_path(DEFAULT_FILE_CSV, FALLBACK_FILE_CSV)
            http_csv = _resolve_data_path(DEFAULT_HTTP_CSV, FALLBACK_HTTP_CSV)
            
            feature_builder = FeatureBuilder(
                file_csv=file_csv,
                http_csv=http_csv,
                file_nrows=200000,
                http_nrows=200000,
            )
            features = feature_builder.build_user_day_features()
            
            # Filter to this user and date
            user_features = features[
                (features["user"].str.upper() == user_id.upper()) &
                (features["day"].astype(str) == date)
            ]
            
            if len(user_features) == 0:
                return JSONResponse({
                    "error": f"No data found for user {user_id} on {date}",
                    "found": False
                })
            
            # Get predictions
            analytics = get_analytics()
            model_features = user_features[[c for c in user_features.columns if c not in {"user", "day"}]]
            risk_scores = analytics.predict_ensemble(model_features)
            
            # Get SHAP attributions for this day
            shap_attributions = analytics.get_top_shap_features(model_features, sample_idx=0)
            
            # Get Gemini explanation
            gemini = GeminiExplainer()
            explanation = gemini.explain_incident(
                user_id=user_id.upper(),
                date=date,
                risk_score=float(risk_scores[0]),
                shap_features=shap_attributions[:10],
            )
            
            response = {
                "found": True,
                "user": user_id.upper(),
                "date": date,
                "risk_score": float(risk_scores[0]),
                "explanation": explanation,
                "top_features": shap_attributions[:10]
            }
            
            return JSONResponse(response)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Explanation generation failed: {e}")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.server:app", host=host, port=port, reload=True)
