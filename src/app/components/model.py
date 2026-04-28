import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.config import FEATURE_COLS

ROOT = os.path.join(os.path.dirname(__file__), "../../..")


@st.cache_resource(show_spinner="앙상블 모델 학습 중... (첫 실행만 소요됩니다)")
def load_model_and_predict():
    train_df = pd.read_csv(os.path.join(ROOT, "data/modeling/train_dataset.csv"))
    pred_df  = pd.read_csv(os.path.join(ROOT, "data/modeling/predict_dataset_2026.csv"))

    train_df["date"] = pd.to_datetime(train_df["date"])
    pred_df["date"]  = pd.to_datetime(pred_df["date"])

    cols = [c for c in FEATURE_COLS if c in train_df.columns and c in pred_df.columns]

    X_train = train_df[cols]
    y_train = train_df["postseason"]

    s_min, s_max = train_df["season"].min(), train_df["season"].max()
    sw = (0.3 + 0.7 * (train_df["season"] - s_min) / (s_max - s_min)).values
    pos_w = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
        scale_pos_weight=pos_w, eval_metric="logloss", random_state=42,
    )
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_child_samples=20,
        scale_pos_weight=pos_w, random_state=42, verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    xgb.fit(X_train, y_train, sample_weight=sw)
    lgbm.fit(X_train, y_train, sample_weight=sw)
    rf.fit(X_train, y_train, sample_weight=sw)

    X_pred = pred_df[cols]
    prob_raw = (
        xgb.predict_proba(X_pred)[:, 1] +
        lgbm.predict_proba(X_pred)[:, 1] +
        rf.predict_proba(X_pred)[:, 1]
    ) / 3

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
