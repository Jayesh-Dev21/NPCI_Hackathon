from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import joblib
import numpy as np
import pandas as pd
import requests


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LocalModelPredictor:
    model_path: str

    def __post_init__(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".joblib", ".jl"}:
            self.model = joblib.load(path)
        else:
            with open(path, "rb") as f:
                self.model = pickle.load(f)

        self.model_meta = self._extract_meta(self.model)

    def _extract_meta(self, obj: object) -> dict:
        meta: dict = {}
        if isinstance(obj, dict):
            if "feature_cols" in obj and isinstance(obj["feature_cols"], (list, tuple)):
                meta["feature_cols"] = [str(c) for c in obj["feature_cols"]]
            if "weights" in obj and isinstance(obj["weights"], dict):
                meta["weights"] = dict(obj["weights"])
            if "threshold" in obj:
                try:
                    meta["threshold"] = float(obj["threshold"])
                except Exception:
                    pass
        return meta

    def _predict_from_bundle(self, features: pd.DataFrame) -> np.ndarray:
        if not isinstance(self.model, dict):
            raise ValueError("Model object is not a bundle")

        xgb_model = self.model.get("xgb_model")
        lgb_model = self.model.get("lgb_model")
        sklearn_model = self.model.get("model")
        weights = self.model.get("weights", {"xgb": 0.55, "lgb": 0.45})

        # Notebook-style ensemble bundle: xgb_model + lgb_model
        if xgb_model is not None and lgb_model is not None:
            xgb_prob = np.asarray(xgb_model.predict_proba(features))[:, 1]
            lgb_prob = np.asarray(lgb_model.predict_proba(features))[:, 1]
            w_xgb = float(weights.get("xgb", 0.55))
            w_lgb = float(weights.get("lgb", 0.45))
            w_sum = w_xgb + w_lgb if (w_xgb + w_lgb) > 0 else 1.0
            return (w_xgb * xgb_prob + w_lgb * lgb_prob) / w_sum

        # Generic wrapped single model bundle: {"model": ...}
        if sklearn_model is not None:
            prepared = self._prepare_input_for_model(sklearn_model, features)
            if hasattr(sklearn_model, "predict_proba"):
                probs = sklearn_model.predict_proba(prepared)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    return probs[:, 1]
                return np.asarray(probs).ravel()
            if hasattr(sklearn_model, "decision_function"):
                raw = sklearn_model.decision_function(prepared)
                return _sigmoid(np.asarray(raw))
            return np.asarray(sklearn_model.predict(prepared), dtype=float)

        raise ValueError(
            "Unsupported model bundle. Expected keys: "
            "(xgb_model and lgb_model) or (model)."
        )

    def get_feature_cols(self) -> Optional[list[str]]:
        cols = self.model_meta.get("feature_cols")
        return list(cols) if isinstance(cols, list) else None

    def get_threshold(self) -> Optional[float]:
        threshold = self.model_meta.get("threshold")
        return float(threshold) if isinstance(threshold, (float, int)) else None

    def _prepare_input_for_model(self, model: object, features: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        # If model has explicit feature names, align to that schema.
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            names = [str(c) for c in list(feature_names)]
            # Many models trained on numpy expose synthetic names like Column_0..Column_N.
            # In that case use positional numeric input instead of name reindexing.
            generic = all(re.fullmatch(r"Column_\d+", n) for n in names)
            if not generic:
                aligned = features.reindex(columns=names).fillna(0.0)
                return aligned

        # If model only knows feature count, trim/pad deterministically.
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None:
            try:
                expected = int(n_features)
            except Exception:
                expected = -1
        else:
            expected = -1

        if expected > 0:
            numeric = features.select_dtypes(include=[np.number]).copy()
            arr = numeric.to_numpy(dtype=float)
            if arr.shape[1] > expected:
                arr = arr[:, :expected]
            elif arr.shape[1] < expected:
                pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=float)
                arr = np.concatenate([arr, pad], axis=1)
            return arr

        return features

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        if isinstance(self.model, dict):
            return self._predict_from_bundle(features)

        prepared = self._prepare_input_for_model(self.model, features)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(prepared)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs.ravel()

        if hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(prepared)
            return _sigmoid(np.asarray(raw))

        preds = self.model.predict(prepared)
        return np.asarray(preds, dtype=float)


@dataclass
class ApiModelPredictor:
    endpoint: str
    timeout_seconds: int = 30
    api_key: Optional[str] = None

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        payload = {"records": features.to_dict(orient="records")}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            self.endpoint,
            data=json.dumps(payload),
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        body = response.json()
        if "scores" in body:
            return np.asarray(body["scores"], dtype=float)
        if "predictions" in body:
            return np.asarray(body["predictions"], dtype=float)

        raise ValueError(
            "API response must include `scores` or `predictions` list."
        )
