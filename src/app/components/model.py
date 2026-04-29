import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.join(os.path.dirname(__file__), "../../..")

TOP_FEATURES = [
    "rank", "win_rate", "games_behind_5th",
    "prev_pythagorean_win_rate", "prev_team_era", "prev_ops_concentration",
    "prev_bb_rate", "prev_top5_hitter_ops_avg", "prev_ace_era",
    "prev_run_differential", "prev_k_bb_ratio", "wins_to_5th",
    "games_behind", "home_win_rate", "dyn_run_differential",
    "prev_iso", "away_win_rate", "dyn_bb_rate",
    "dyn_pythagorean_win_rate", "dyn_k_bb_ratio",
]


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
def load_model_and_predict():
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
