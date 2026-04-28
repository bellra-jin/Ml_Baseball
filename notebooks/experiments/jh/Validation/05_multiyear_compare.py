"""
05_multiyear_compare.py

3년 누적 vs 5년 누적 피처 비교 실험.
BASE(prev_만) / BASE+3yr+trend / BASE+3yr_only 세 가지를 같은 조건으로 학습 후 비교한다.

실행: uv run python "notebooks/experiments/jh /05_multiyear_compare.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.utils.config import FEATURE_COLS as BASE_COLS

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(os.path.dirname(__file__), "../../..")

# ──────────────────────────────────────────────
# 1. 피처 정의
# ──────────────────────────────────────────────
MULTI_YEAR_KEYS = [
    "pythagorean_win_rate",
    "run_differential",
    "team_era",
    "k_bb_ratio",
    "top5_hitter_ops_avg",
    "ace_era",
    "iso",
    "ops_concentration",
    "bb_rate",
]

TREND_COLS = [f"trend_{k}" for k in MULTI_YEAR_KEYS]
AVG3_COLS  = [f"avg3yr_{k}" for k in MULTI_YEAR_KEYS]

COLS_3YR      = BASE_COLS + AVG3_COLS + TREND_COLS   # avg3yr + trend
COLS_3YR_ONLY = BASE_COLS + AVG3_COLS                # avg3yr만 (trend 없음)

print(f"BASE:        {len(BASE_COLS)}개")
print(f"3yr+trend:   {len(COLS_3YR)}개  (avg3yr × {len(AVG3_COLS)} + trend × {len(TREND_COLS)})")
print(f"3yr_only:    {len(COLS_3YR_ONLY)}개  (avg3yr × {len(AVG3_COLS)})\n")


# ──────────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

train = df[df["season"] != 2025].copy()
test  = df[df["season"] == 2025].copy()

s_min, s_max = train["season"].min(), train["season"].max()
sw = (0.3 + 0.7 * (train["season"] - s_min) / (s_max - s_min)).values

print(f"학습: {train.shape}  테스트: {test.shape}\n")


# ──────────────────────────────────────────────
# 3. 모델 학습 / 예측 함수
# ──────────────────────────────────────────────
def build_ensemble(X_tr, y_tr, sw):
    pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()
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
    xgb.fit(X_tr, y_tr, sample_weight=sw)
    lgbm.fit(X_tr, y_tr, sample_weight=sw)
    rf.fit(X_tr, y_tr, sample_weight=sw)
    return xgb, lgbm, rf


def predict_prob(models, X):
    xgb, lgbm, rf = models
    return (
        xgb.predict_proba(X)[:, 1] +
        lgbm.predict_proba(X)[:, 1] +
        rf.predict_proba(X)[:, 1]
    ) / 3


def normalize(series):
    return series.transform(lambda x: (x / x.sum() * 5).clip(upper=1.0))


# ──────────────────────────────────────────────
# 4. 세 가지 피처셋 학습 및 예측
# ──────────────────────────────────────────────
y_train = train["postseason"]
y_test  = test["postseason"]
test    = test.copy()

configs = [
    ("BASE",     BASE_COLS),
    ("3yr",      COLS_3YR),
    ("3yr_only", COLS_3YR_ONLY),
]

trained_models = {}
for label, cols in configs:
    print(f"[{label}] 학습 중...")
    models = build_ensemble(train[cols], y_train, sw)
    trained_models[label] = (models, cols)
    prob_raw = predict_prob(models, test[cols])
    test[f"prob_{label}"]      = prob_raw
    test[f"prob_{label}_norm"] = normalize(test.groupby("date")[f"prob_{label}"])


# ──────────────────────────────────────────────
# 5. 시점별 적중률 비교
# ──────────────────────────────────────────────
checkpoints = {
    "50%  (72경기)":  0.50,
    "75% (108경기)":  0.75,
    "90% (130경기)":  0.90,
    "최종 (144경기)":  1.00,
}

actual_top5 = set(test[test["postseason"] == 1]["team"].unique())
print(f"\n[실제 2025 포스트시즌] {sorted(actual_top5)}\n")

header = f"{'시점':<16} {'BASE':>6} {'3yr':>8} {'3yr_only':>10}"
print(header)
print("─" * len(header))

for label, ratio in checkpoints.items():
    snap   = test[test["games_played_ratio"] <= ratio]
    latest = snap.sort_values("date").groupby("team").last().reset_index()

    hits = {}
    teams = {}
    for cfg_label, _ in configs:
        top5 = set(latest.nlargest(5, f"prob_{cfg_label}_norm")["team"])
        hits[cfg_label]  = len(top5 & actual_top5)
        teams[cfg_label] = sorted(top5)

    print(f"{label:<16} {hits['BASE']}/5  {hits['3yr']}/5  {hits['3yr_only']}/5")
    print(f"  BASE:     {teams['BASE']}")
    print(f"  3yr:      {teams['3yr']}")
    print(f"  3yr_only: {teams['3yr_only']}")
    print()

# ROC-AUC
print("ROC-AUC:")
for cfg_label, _ in configs:
    auc = roc_auc_score(y_test, test[f"prob_{cfg_label}"])
    print(f"  {cfg_label}: {auc:.4f}")


# ──────────────────────────────────────────────
# 6. 최종 시점 바 차트 3종 비교
# ──────────────────────────────────────────────
final = test.sort_values("date").groupby("team").last().reset_index()
final = final.sort_values("prob_BASE_norm", ascending=True)
colors_bar = ["steelblue" if t in actual_top5 else "lightgray" for t in final["team"]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (cfg_label, _) in zip(axes, configs):
    col = f"prob_{cfg_label}_norm"
    bars = ax.barh(final["team"], final[col], color=colors_bar)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("포스트시즌 예측 확률")
    ax.set_title(f"2025 최종 시점 — {cfg_label}\n(파란색: 실제 포스트시즌 진출팀)")
    for bar, val in zip(bars, final[col]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
out = os.path.join(BASE, "data/modeling/multiyear_compare_bar.png")
plt.savefig(out, dpi=120)
plt.show()
print(f"\n차트 저장: {out}")


# ──────────────────────────────────────────────
# 7. 시점별 적중률 바 차트
# ──────────────────────────────────────────────
cp_labels = list(checkpoints.keys())
hits_by_cfg = {cfg_label: [] for cfg_label, _ in configs}

for ratio in checkpoints.values():
    snap   = test[test["games_played_ratio"] <= ratio]
    latest = snap.sort_values("date").groupby("team").last().reset_index()
    for cfg_label, _ in configs:
        top5 = set(latest.nlargest(5, f"prob_{cfg_label}_norm")["team"])
        hits_by_cfg[cfg_label].append(len(top5 & actual_top5))

x = np.arange(len(cp_labels))
w = 0.25
colors_cfg = ["steelblue", "coral", "seagreen"]

fig, ax = plt.subplots(figsize=(10, 5))
for i, (cfg_label, _) in enumerate(configs):
    bars = ax.bar(x + (i - 1) * w, hits_by_cfg[cfg_label], w,
                  label=cfg_label, color=colors_cfg[i], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{int(bar.get_height())}/5", ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(cp_labels, fontsize=11)
ax.set_ylabel("적중 팀 수 (/ 5)")
ax.set_ylim(0, 6)
ax.set_yticks(range(6))
ax.axhline(5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_title("시점별 포스트시즌 예측 적중률\nBASE vs 3yr+trend vs 3yr_only (2025 검증)")
ax.legend(fontsize=11)
plt.tight_layout()
out2 = os.path.join(BASE, "data/modeling/multiyear_compare_checkpoint.png")
plt.savefig(out2, dpi=120)
plt.show()
print(f"차트 저장: {out2}")


# ──────────────────────────────────────────────
# 8. 피처 중요도 분석 (BASE vs 3yr)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

for ax, cfg_label in zip(axes, ["BASE", "3yr"]):
    (xgb, lgbm, rf), cols = trained_models[cfg_label]

    # XGB + LGBM + RF 중요도 평균 (정규화 후)
    imp_xgb  = pd.Series(xgb.feature_importances_,  index=cols)
    imp_lgbm = pd.Series(lgbm.feature_importances_, index=cols)
    imp_rf   = pd.Series(rf.feature_importances_,   index=cols)

    imp = (
        imp_xgb  / imp_xgb.sum() +
        imp_lgbm / imp_lgbm.sum() +
        imp_rf   / imp_rf.sum()
    ) / 3

    top20 = imp.sort_values(ascending=False).head(20)

    colors = []
    for feat in top20.index:
        if feat.startswith("trend_"):
            colors.append("coral")
        elif feat.startswith("avg3yr_") or feat.startswith("avg5yr_"):
            colors.append("seagreen")
        elif feat.startswith("prev_"):
            colors.append("steelblue")
        else:
            colors.append("slategray")

    ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.set_title(f"피처 중요도 Top 20 — {cfg_label}\n"
                 f"(회색: 현재시즌, 파랑: prev_, 초록: avg_nyr, 빨강: trend_)")
    ax.set_xlabel("중요도 (3개 모델 평균 정규화)")

plt.tight_layout()
out3 = os.path.join(BASE, "data/modeling/multiyear_feature_importance.png")
plt.savefig(out3, dpi=120)
plt.show()
print(f"차트 저장: {out3}")

# trend_ 변수 순위 출력
print("\n[3yr 모델] trend_ 변수 중요도 순위:")
(_, _, _), cols_3yr = trained_models["3yr"]
(xgb3, lgbm3, rf3), _ = trained_models["3yr"]
imp3 = (
    pd.Series(xgb3.feature_importances_,  index=cols_3yr) / pd.Series(xgb3.feature_importances_,  index=cols_3yr).sum() +
    pd.Series(lgbm3.feature_importances_, index=cols_3yr) / pd.Series(lgbm3.feature_importances_, index=cols_3yr).sum() +
    pd.Series(rf3.feature_importances_,   index=cols_3yr) / pd.Series(rf3.feature_importances_,   index=cols_3yr).sum()
) / 3
imp3_sorted = imp3.sort_values(ascending=False)
trend_ranks = [(feat, imp3_sorted.index.get_loc(feat) + 1, val)
               for feat, val in imp3.items() if feat.startswith("trend_")]
trend_ranks.sort(key=lambda x: x[1])
for feat, rank, val in trend_ranks:
    print(f"  {rank:>3}위  {feat:<40} {val:.5f}")
