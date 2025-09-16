import pandas as pd
from typing import Tuple, List

FEATURE_COLUMNS: List[str] = [
    "danceability","energy","key","loudness","mode","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","time_signature",
    "duration_ms","explicit"
]

TARGET_REG = "popularity"
TARGET_CLS = "is_hit"

def make_features(df: pd.DataFrame, hit_threshold: int = 75) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    df[TARGET_CLS] = (df[TARGET_REG] >= hit_threshold).astype(int)
    X = df[FEATURE_COLUMNS].fillna(0.0)
    y_reg = df[TARGET_REG]
    y_cls = df[TARGET_CLS]
    return X, y_reg, y_cls
