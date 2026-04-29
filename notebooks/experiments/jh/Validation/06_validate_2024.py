"""
06_validate_2024.py

2024년 데이터가 없다는 가정 하에 2017~2023 학습 → 2024 예측 검증.
BASE / 3yr+trend / 3yr_only 세 가지 피처셋을 비교한다.

실행: uv run python "notebooks/experiments/jh /06_validate_2024.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.config import FEATURE_COLS as BASE_COLS
from src.evaluation.metrics import (
    evaluate_binary_model,
    checkpoint_hits,
    print_checkpoint_report,
    print_metrics,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE   = os.path.join(os.path.dirname(__file__), "../../../..")
ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

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

TREND_COLS    = [f"trend_{k}"  for k in MULTI_YEAR_KEYS]
AVG3_COLS     = [f"avg3yr_{k}" for k in MULTI_YEAR_KEYS]

COLS_3YR      = BASE_COLS + AVG3_COLS + TREND_COLS
COLS_3YR_ONLY = BASE_COLS + AVG3_COLS

configs = [
    ("BASE",     BASE_COLS),
    ("3yr",      COLS_3YR),
    ("3yr_only", COLS_3YR_ONLY),
]

print(f"BASE:      {len(BASE_COLS)}개")
print(f"3yr:       {len(COLS_3YR)}개")
print(f"3yr_only:  {len(COLS_3YR_ONLY)}개\n")

# ──────────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

train = df[df["season"] < 2024].copy()    # 2017~2023 학습
test  = df[df["season"] == 2024].copy()   # 2024 검증

s_min, s_max = train["season"].min(), train["season"].max()
sw = (0.3 + 0.7 * (train["season"] - s_min) / (s_max - s_min)).values

print(f"학습 시즌: {sorted(train['season'].unique())}")
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
# 4. 학습 및 예측
# ──────────────────────────────────────────────
y_train = train["postseason"]
y_test  = test["postseason"]
test    = test.copy()

missing = {label: [c for c in cols if c not in df.columns] for label, cols in configs}
for label, miss in missing.items():
    if miss:
        print(f"[경고] {label} 누락 피처: {miss}")

trained_models = {}
for label, cols in configs:
    print(f"[{label}] 학습 중...")
    models = build_ensemble(train[cols], y_train, sw)
    trained_models[label] = (models, cols)
    prob_raw = predict_prob(models, test[cols])
    test[f"prob_{label}"]      = prob_raw
    test[f"prob_{label}_norm"] = normalize(test.groupby("date")[f"prob_{label}"])

# ──────────────────────────────────────────────
# 5. 시점별 적중률
# ──────────────────────────────────────────────
checkpoints = {
    "50%  (72경기)":  0.50,
    "75% (108경기)":  0.75,
    "90% (130경기)":  0.90,
    "최종 (144경기)":  1.00,
}

actual_top5 = set(test[test["postseason"] == 1]["team"].unique())
print(f"\n[실제 2024 포스트시즌] {sorted(actual_top5)}\n")

header = f"{'시점':<16} {'BASE':>6} {'3yr':>8} {'3yr_only':>10}"
print(header)
print("─" * len(header))

for label, ratio in checkpoints.items():
    snap   = test[test["games_played_ratio"] <= ratio]
    latest = snap.sort_values("date").groupby("team").last().reset_index()

    hits, teams = {}, {}
    for cfg, _ in configs:
        top5       = set(latest.nlargest(5, f"prob_{cfg}_norm")["team"])
        hits[cfg]  = len(top5 & actual_top5)
        teams[cfg] = sorted(top5)

    print(f"{label:<16} {hits['BASE']}/5  {hits['3yr']}/5  {hits['3yr_only']}/5")
    print(f"  BASE:     {teams['BASE']}")
    print(f"  3yr:      {teams['3yr']}")
    print(f"  3yr_only: {teams['3yr_only']}")
    print()

print("ROC-AUC / F1 / Accuracy:")
for cfg, _ in configs:
    m = evaluate_binary_model(y_test, test[f"prob_{cfg}"])
    print_metrics(m, label=cfg)

# ──────────────────────────────────────────────
# 6. 최종 시점 바 차트
# ──────────────────────────────────────────────
final      = test.sort_values("date").groupby("team").last().reset_index()
final      = final.sort_values("prob_BASE_norm", ascending=True)
bar_colors = ["steelblue" if t in actual_top5 else "lightgray" for t in final["team"]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (cfg, _) in zip(axes, configs):
    col  = f"prob_{cfg}_norm"
    bars = ax.barh(final["team"], final[col], color=bar_colors)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("포스트시즌 예측 확률")
    ax.set_title(f"2024 최종 시점 — {cfg}\n(파란색: 실제 포스트시즌 진출팀)")
    for bar, val in zip(bars, final[col]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
out = os.path.join(ASSETS, "validate_2024_bar.png")
plt.savefig(out, dpi=120)
plt.show()
print(f"\n차트 저장: {out}")

# ──────────────────────────────────────────────
# 7. 시점별 적중률 바 차트
# ──────────────────────────────────────────────
hits_by_cfg = {cfg: [] for cfg, _ in configs}
for ratio in checkpoints.values():
    snap   = test[test["games_played_ratio"] <= ratio]
    latest = snap.sort_values("date").groupby("team").last().reset_index()
    for cfg, _ in configs:
        top5 = set(latest.nlargest(5, f"prob_{cfg}_norm")["team"])
        hits_by_cfg[cfg].append(len(top5 & actual_top5))

x = np.arange(len(checkpoints))
w = 0.25
colors_cfg = ["steelblue", "coral", "seagreen"]

fig, ax = plt.subplots(figsize=(10, 5))
for i, (cfg, _) in enumerate(configs):
    bars = ax.bar(x + (i - 1) * w, hits_by_cfg[cfg], w,
                  label=cfg, color=colors_cfg[i], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{int(bar.get_height())}/5", ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(list(checkpoints.keys()), fontsize=11)
ax.set_ylabel("적중 팀 수 (/ 5)")
ax.set_ylim(0, 6)
ax.set_yticks(range(6))
ax.axhline(5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_title("시점별 포스트시즌 예측 적중률\nBASE vs 3yr+trend vs 3yr_only (2024 검증)")
ax.legend(fontsize=11)
plt.tight_layout()
out2 = os.path.join(ASSETS, "validate_2024_checkpoint.png")
plt.savefig(out2, dpi=120)
plt.show()
print(f"차트 저장: {out2}")

# ──────────────────────────────────────────────
# 8. 확률 추이 차트 (BASE)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
teams  = sorted(test["team"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(teams)))

for team, color in zip(teams, colors):
    t  = test[test["team"] == team].sort_values("games_played_ratio")
    ls = "-"  if team in actual_top5 else "--"
    lw = 2.2  if team in actual_top5 else 1.0
    ax.plot(t["games_played_ratio"] * 144, t["prob_BASE_norm"],
            label=team, color=color, linestyle=ls, linewidth=lw)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("경기 수 (진행)")
ax.set_ylabel("포스트시즌 예측 확률")
ax.set_title(
    f"2024 시즌 팀별 포스트시즌 확률 추이 — BASE\n"
    f"(학습: 2017~2023 / 피처: {len(BASE_COLS)}개 / 실선: 실제 진출팀)"
)
ax.legend(loc="upper left", ncol=2, fontsize=9)
ax.xaxis.set_major_locator(mticker.MultipleLocator(18))
plt.tight_layout()
out3 = os.path.join(ASSETS, "validate_2024_trend.png")
plt.savefig(out3, dpi=120)
plt.show()
print(f"차트 저장: {out3}")
