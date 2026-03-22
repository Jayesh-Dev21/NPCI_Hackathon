from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class FeatureBuilder:
    file_csv: Optional[str] = None
    http_csv: Optional[str] = None
    file_nrows: Optional[int] = None
    http_nrows: Optional[int] = None

    def _load_csv(
        self,
        path: str,
        usecols: list[str],
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing required data file: {p}")

        header_cols = pd.read_csv(path, nrows=0).columns.tolist()
        available = [c for c in usecols if c in header_cols]

        if "date" not in available or "user" not in available:
            raise ValueError(
                f"CSV at {path} must contain at least 'date' and 'user' columns. "
                f"Found: {header_cols}"
            )

        df = pd.read_csv(path, usecols=available, nrows=nrows)
        for col in usecols:
            if col not in df.columns:
                df[col] = None
        return df

    def build_user_day_features(self) -> pd.DataFrame:
        if not self.file_csv and not self.http_csv:
            raise ValueError("At least one of file_csv or http_csv is required.")

        frames: list[pd.DataFrame] = []

        if self.file_csv:
            file_df = self._load_csv(
                self.file_csv,
                usecols=["date", "user", "activity", "filename"],
                nrows=self.file_nrows,
            )
            file_df["date"] = pd.to_datetime(file_df["date"], errors="coerce")
            file_df = file_df.dropna(subset=["date", "user"])
            file_df["user"] = file_df["user"].astype(str).str.strip().str.lower()
            file_df["day"] = file_df["date"].dt.date

            grouped = file_df.groupby(["user", "day"], as_index=False).agg(
                file_count=("date", "size"),
                unique_files=("filename", "nunique"),
            )

            file_df["is_delete"] = file_df["activity"].astype(str).str.contains(
                "delete", case=False, na=False
            )
            file_df["is_copy"] = file_df["activity"].astype(str).str.contains(
                "copy", case=False, na=False
            )
            grouped_extra = file_df.groupby(["user", "day"], as_index=False).agg(
                file_delete_count=("is_delete", "sum"),
                file_copy_count=("is_copy", "sum"),
            )
            grouped = grouped.merge(grouped_extra, on=["user", "day"], how="left")

            # Notebook-compatible aliases
            grouped["file_delete"] = grouped["file_delete_count"]
            grouped["file_copy"] = grouped["file_copy_count"]
            grouped["file_after_hours"] = 0.0
            grouped["file_write"] = 0.0
            frames.append(grouped)

        if self.http_csv:
            http_df = self._load_csv(
                self.http_csv,
                usecols=["date", "user", "url"],
                nrows=self.http_nrows,
            )
            http_df["date"] = pd.to_datetime(http_df["date"], errors="coerce")
            http_df = http_df.dropna(subset=["date", "user"])
            http_df["user"] = http_df["user"].astype(str).str.strip().str.lower()
            http_df["day"] = http_df["date"].dt.date

            risky_patterns = [
                "dropbox",
                "drive.google",
                "mega",
                "wikileaks",
                "pastebin",
                "job",
                "resume",
            ]
            regex = "|".join(risky_patterns)
            http_df["is_suspicious"] = http_df["url"].astype(str).str.contains(
                regex, case=False, na=False
            )

            grouped = http_df.groupby(["user", "day"], as_index=False).agg(
                http_count=("url", "count"),
                unique_urls=("url", "nunique"),
                http_suspicious_count=("is_suspicious", "sum"),
            )
            grouped["http_suspicious"] = grouped["http_suspicious_count"]
            grouped["http_after_hours"] = 0.0
            frames.append(grouped)

        if not frames:
            raise ValueError("No features could be built from provided sources.")

        base = frames[0]
        for frame in frames[1:]:
            base = base.merge(frame, on=["user", "day"], how="outer")

        base = base.fillna(0)
        base["day"] = pd.to_datetime(base["day"]).dt.date

        required_defaults = {
            "logon_count": 0.0,
            "unique_pcs": 0.0,
            "logon_after_hours": 0.0,
            "logon_weekend": 0.0,
            "logon_mean_hour": 0.0,
            "logon_std_hour": 0.0,
            "file_count": 0.0,
            "unique_files": 0.0,
            "file_after_hours": 0.0,
            "file_copy_count": 0.0,
            "file_delete_count": 0.0,
            "file_copy": 0.0,
            "file_delete": 0.0,
            "file_write": 0.0,
            "device_count": 0.0,
            "device_after_hours": 0.0,
            "usb_connect": 0.0,
            "email_count": 0.0,
            "email_external": 0.0,
            "email_after_hours": 0.0,
            "email_attachments": 0.0,
            "http_count": 0.0,
            "http_after_hours": 0.0,
            "unique_urls": 0.0,
            "http_suspicious_count": 0.0,
            "http_suspicious": 0.0,
        }
        for col, default in required_defaults.items():
            if col not in base:
                base[col] = default

        base["delete_ratio"] = base["file_delete"] / (base["file_count"] + 1)
        base["suspicious_ratio"] = base["http_suspicious"] / (base["http_count"] + 1)
        base["total_activity"] = base["file_count"] + base["http_count"]

        # Notebook-compatible ratio names
        base["logon_ah_ratio"] = base["logon_after_hours"] / (base["logon_count"] + 1)
        base["email_ext_ratio"] = base["email_external"] / (base["email_count"] + 1)
        base["file_del_ratio"] = base["file_delete"] / (base["file_count"] + 1)
        base["file_copy_ratio"] = base["file_copy"] / (base["file_count"] + 1)
        base["usb_per_logon"] = base["usb_connect"] / (base["logon_count"] + 1)
        base["http_sus_ratio"] = base["http_suspicious"] / (base["http_count"] + 1)

        dt = pd.to_datetime(base["day"])
        base["day_of_week"] = dt.dt.dayofweek.astype(float)
        base["month"] = dt.dt.month.astype(float)
        base["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

        # Lightweight rolling features compatible with notebook naming
        base = base.sort_values(["user", "day"]).reset_index(drop=True)
        roll_cols = [
            "logon_count",
            "file_count",
            "device_count",
            "email_count",
            "http_count",
            "usb_connect",
            "email_external",
            "file_delete",
            "http_suspicious",
        ]
        for col in roll_cols:
            base[f"{col}_7d_mean"] = (
                base.groupby("user")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
            )
            base[f"{col}_7d_std"] = (
                base.groupby("user")[col]
                .transform(lambda s: s.rolling(7, min_periods=1).std())
                .fillna(0.0)
            )
            user_mean = base.groupby("user")[col].transform("mean")
            user_std = base.groupby("user")[col].transform("std").fillna(1.0)
            base[f"{col}_zscore"] = (base[col] - user_mean) / (user_std + 1e-6)

        numeric_cols = [
            c
            for c in base.columns
            if c not in {"user", "day"} and pd.api.types.is_numeric_dtype(base[c])
        ]
        base[numeric_cols] = base[numeric_cols].astype(float)

        return base.sort_values(["day", "user"]).reset_index(drop=True)
