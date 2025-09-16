import os
import pandas as pd

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
