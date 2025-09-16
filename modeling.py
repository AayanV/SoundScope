from typing import Dict, Any
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from .features import FEATURE_COLUMNS, make_features, TARGET_REG, TARGET_CLS

def train_and_evaluate(df: pd.DataFrame, outdir: str, random_state: int = 42, hit_threshold: int = 75) -> Dict[str, Any]:
    X, y_reg, y_cls = make_features(df, hit_threshold=hit_threshold)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=random_state)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=random_state)

    numeric_features = list(range(X.shape[1]))
    pre = ColumnTransformer([("scale", StandardScaler(), numeric_features)], remainder="drop")

    reg = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=random_state))])
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)

    cls = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=400, random_state=random_state))])
    cls.fit(X_train_c, y_train_c)
    y_pred_c = cls.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)

    dump(reg, f"{outdir}/regression_model.joblib")
    dump(cls, f"{outdir}/classifier_model.joblib")

    pi = permutation_importance(reg, X_test_r, y_test_r, n_repeats=10, random_state=random_state)
    importances = pi.importances_mean
    order = np.argsort(importances)[::-1]
    top_idx = order[:10]
    top_features = [FEATURE_COLUMNS[i] for i in top_idx]
    top_scores = importances[top_idx]

    plt.figure()
    plt.barh(range(len(top_features))[::-1], top_scores[::-1])
    plt.yticks(range(len(top_features))[::-1], top_features[::-1])
    plt.xlabel("Permutation importance (mean)")
    plt.title("Top 10 Features Influencing Popularity")
    plt.tight_layout()
    plt.savefig(f"{outdir}/feature_importance.png", dpi=150)
    plt.close()

    metrics = {
        "regression": {"r2": float(r2), "mae": float(mae)},
        "classification": {"accuracy": float(acc)},
        "top_features": [{"feature": f, "importance": float(s)} for f, s in zip(top_features, top_scores)]
    }
    with open(f"{outdir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
