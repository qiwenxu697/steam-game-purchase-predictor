from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import ast
import pandas as pd
import numpy as np

def safe_literal_eval(x: str) -> Any:
    """Parse python-literal strings safely (many Steam dumps are python dict strings)."""
    return ast.literal_eval(x)


def to_list_or_empty(x: Any) -> list:
    """Convert NaN/None/scalars into a list, keep list as-is."""
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    return [x]


def sentiment_to_score(x: Any) -> float:
    """
    Convert sentiment label to numeric.
    Adjust mapping if your dataset uses different labels.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0

    s = str(x).strip().lower()
    mapping = {
        "positive": 1.0,
        "recommended": 1.0,
        "mostly positive": 0.7,
        "mixed": 0.0,
        "mostly negative": -0.7,
        "negative": -1.0,
        "not recommended": -1.0,
    }
    return mapping.get(s, 0.0)


def coerce_price(series: pd.Series) -> pd.Series:
    """Convert price column to numeric with safe fallback."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def add_user_game_count(full_df: pd.DataFrame, user_items: pd.DataFrame) -> pd.DataFrame:
    """Adds user_game_count feature based on user_items history."""
    counts = user_items.groupby("user_id")["item_id"].nunique()
    out = full_df.copy()
    out["user_game_count"] = out["user_id"].map(counts).fillna(0).astype(int)
    return out

def to_list_or_empty(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return []

def sentiment_to_score(x):
    mapping = {
        "Overwhelmingly Positive": 5,
        "Very Positive": 4,
        "Positive": 3,
        "Mostly Positive": 2,
        "Mixed": 1,
        "Negative": -1,
        "Mostly Negative": -2,
        "Very Negative": -3,
    }
    return mapping.get(x, 0)

def add_game_age(df: pd.DataFrame, current_year=2025):
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year
    df["game_age"] = current_year - df["release_year"]
    df["game_age"] = df["game_age"].fillna(df["game_age"].median())
    return df

def add_log_playtime(df: pd.DataFrame):
    df["playtime_log"] = np.log1p(df["playtime_forever"])
    return df
