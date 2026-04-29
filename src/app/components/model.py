import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import TOP_FEATURES

ROOT = os.path.join(os.path.dirname(__file__), "../../..")
MODEL_CACHE_VERSION = "strategy_c_20_features_v2"


def _build_models(pos_w):
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1,
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=20,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=40, max_depth=2, learning_rate=0.1,
        subsample=0.6, colsample_bytree=0.5,
        reg_alpha=5.0, reg_lambda=10.0, min_child_weight=30,
        scale_pos_weight=pos_w, eval_metric="logloss", random_state=42,
    )
    lgbm = LGBMClassifier(
        n_estimators=40, max_depth=2, learning_rate=0.1,
        subsample=0.6, colsample_bytree=0.5,
        reg_alpha=5.0, reg_lambda=10.0, min_child_samples=50,
        scale_pos_weight=pos_w, random_state=42, verbose=-1,
    )
    return lr, rf, xgb, lgbm


def _ensemble_proba(lr, rf, xgb, lgbm, X):
    return (
        lr.predict_proba(X)[:, 1] +
        rf.predict_proba(X)[:, 1] +
        xgb.predict_proba(X)[:, 1] +
        lgbm.predict_proba(X)[:, 1]
    ) / 4


@st.cache_resource(show_spinner="Strategy C 앙상블 모델 학습 중... (첫 실행만 소요됩니다)")
def load_model_and_predict(cache_version: str = MODEL_CACHE_VERSION):
    _ = cache_version

    train_df = pd.read_csv(os.path.join(ROOT, "data/modeling/train_dataset.csv"))
    pred_df  = pd.read_csv(os.path.join(ROOT, "data/modeling/predict_dataset_2026.csv"))

    train_df["date"] = pd.to_datetime(train_df["date"])
    pred_df["date"]  = pd.to_datetime(pred_df["date"])

    cols = [c for c in TOP_FEATURES if c in train_df.columns and c in pred_df.columns]

    X_train = train_df[cols]
    y_train = train_df["postseason"]

    s_min, s_max = train_df["season"].min(), train_df["season"].max()
    sw = (0.3 + 0.7 * (train_df["season"] - s_min) / (s_max - s_min)).values
    pos_w = (y_train == 0).sum() / (y_train == 1).sum()

    lr, rf, xgb, lgbm = _build_models(pos_w)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train, sample_weight=sw)
    xgb.fit(X_train, y_train, sample_weight=sw)
    lgbm.fit(X_train, y_train, sample_weight=sw)

    X_pred = pred_df[cols]
    prob_raw = _ensemble_proba(lr, rf, xgb, lgbm, X_pred)

    pred_df = pred_df.copy()
    pred_df["prob_raw"]  = prob_raw
    pred_df["prob_norm"] = pred_df.groupby("date")["prob_raw"].transform(
        lambda x: (x / x.sum() * 5).clip(upper=1.0)
    )

    rank_df = (
        pred_df
        .sort_values(["date", "prob_norm"], ascending=[True, False])
        .assign(pred_rank=lambda d: d.groupby("date").cumcount() + 1)
    )

    imp_xgb  = pd.Series(xgb.feature_importances_,  index=cols)
    imp_lgbm = pd.Series(lgbm.feature_importances_, index=cols)
    imp_rf   = pd.Series(rf.feature_importances_,   index=cols)
    importance = (
        imp_xgb  / imp_xgb.sum() +
        imp_lgbm / imp_lgbm.sum() +
        imp_rf   / imp_rf.sum()
    ) / 3

    return pred_df, rank_df, importance, cols


@st.cache_data(show_spinner="LOSO-CV 검증 실행 중... (첫 실행만 소요됩니다)")
def run_loso_cv(cache_version: str = MODEL_CACHE_VERSION):
    _ = cache_version
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss,
    )

    train_df = pd.read_csv(os.path.join(ROOT, "data/modeling/train_dataset.csv"))
    train_df["date"] = pd.to_datetime(train_df["date"])
    cols = [c for c in TOP_FEATURES if c in train_df.columns]
    test_seasons = sorted(s for s in train_df["season"].unique() if s >= 2018)

    metrics, oof_probs, oof_labels, cp_rows = [], [], [], []

    for test_season in test_seasons:
        tr = train_df[train_df["season"] != test_season]
        te = train_df[train_df["season"] == test_season]
        X_tr, y_tr = tr[cols], tr["postseason"]
        X_te, y_te = te[cols], te["postseason"]

        pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()
        s_min, s_max = tr["season"].min(), tr["season"].max()
        sw = (0.3 + 0.7 * (tr["season"] - s_min) / (s_max - s_min)).values

        lr, rf, xgb, lgbm = _build_models(pos_w)
        lr.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr, sample_weight=sw)
        xgb.fit(X_tr, y_tr, sample_weight=sw)
        lgbm.fit(X_tr, y_tr, sample_weight=sw)

        te_p = _ensemble_proba(lr, rf, xgb, lgbm, X_te)
        tr_p = _ensemble_proba(lr, rf, xgb, lgbm, X_tr)
        te_bin = (te_p >= 0.5).astype(int)

        metrics.append({
            "season":    int(test_season),
            "train_auc": float(roc_auc_score(y_tr, tr_p)),
            "test_auc":  float(roc_auc_score(y_te, te_p)),
            "f1":        float(f1_score(y_te, te_bin, zero_division=0)),
            "precision": float(precision_score(y_te, te_bin, zero_division=0)),
            "recall":    float(recall_score(y_te, te_bin, zero_division=0)),
            "brier":     float(brier_score_loss(y_te, te_p)),
        })
        oof_probs.extend(te_p.tolist())
        oof_labels.extend(y_te.tolist())

        actual_top5 = set(te[te["postseason"] == 1]["team"].unique())
        for cp_ratio, cp_label in [(0.5, "50%"), (0.75, "75%"), (0.9, "90%"), (1.0, "최종")]:
            snap = (
                te[te["games_played_ratio"] <= cp_ratio]
                .sort_values("date")
                .groupby("team")
                .last()
                .reset_index()
            )
            if snap.empty:
                continue
            snap_p = _ensemble_proba(lr, rf, xgb, lgbm, snap[cols])
            pred_top5 = set(snap.assign(p=snap_p).nlargest(5, "p")["team"])
            cp_rows.append({
                "season":     int(test_season),
                "checkpoint": cp_label,
                "hit":        len(pred_top5 & actual_top5),
            })

    return (
        pd.DataFrame(metrics),
        np.array(oof_probs),
        np.array(oof_labels),
        pd.DataFrame(cp_rows),
    )
