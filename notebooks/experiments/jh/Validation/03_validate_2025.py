"""
03_validate_2025.py

2025년 데이터가 없다는 가정 하에 2016~2024 학습 → 2025 예측 검증.
실제 2026 예측 파이프라인과 동일한 구조로 작동한다.

실행: uv run python "notebooks/experiments/jh /03_validate_2025.py"
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
from src.utils.config import FEATURE_COLS
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
# 1. 데이터 로드
# ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

train = df[df["season"] != 2025].copy()   # 2017~2024 학습
pred  = df[df["season"] == 2025].copy()   # 2025: 예측 대상 (정답 모름 가정)

print(f"피처 수: {len(FEATURE_COLS)}개")
print(f"학습: {train.shape}  시즌 {sorted(train['season'].unique())}")
print(f"예측: {pred.shape}   시즌 2025\n")

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"[경고] 없는 피처: {missing}")

# ──────────────────────────────────────────────
# 2. 학습
# ──────────────────────────────────────────────
X_train = train[FEATURE_COLS]
y_train = train["postseason"]
X_pred  = pred[FEATURE_COLS]

# 최근 시즌일수록 높은 가중치 (2017=0.3 → 2024=1.0)
s_min, s_max = train["season"].min(), train["season"].max()
season_w = 0.3 + 0.7 * (train["season"] - s_min) / (s_max - s_min)
sw = season_w.values

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

print("모델 학습 중...")
xgb.fit(X_train, y_train, sample_weight=sw)
lgbm.fit(X_train, y_train, sample_weight=sw)
rf.fit(X_train, y_train, sample_weight=sw)
print("학습 완료\n")

# ──────────────────────────────────────────────
# 3. 예측 확률 계산
# ──────────────────────────────────────────────
pred = pred.copy()
pred["prob_raw"] = (
    xgb.predict_proba(X_pred)[:, 1] +
    lgbm.predict_proba(X_pred)[:, 1] +
    rf.predict_proba(X_pred)[:, 1]
) / 3

# 일별 스냅샷 내 상대 확률 정규화 (10팀 합 → 5)
pred["prob"] = pred.groupby("date")["prob_raw"].transform(
    lambda x: (x / x.sum() * 5).clip(upper=1.0)
)

# ──────────────────────────────────────────────
# 4. 시점별 상위 5팀 예측
# ──────────────────────────────────────────────
checkpoints = {
    
    "50%  (72경기)":  0.50,
    "75% (108경기)":  0.75,
    "90% (130경기)":  0.90,
    "최종 (144경기)":  1.00,
}

# 실제 정답 (검증용 — 실제 예측 시엔 없는 정보)
actual_top5 = set(pred[pred["postseason"] == 1]["team"].unique())
print(f"[실제 2025 포스트시즌] {sorted(actual_top5)}\n")

results = checkpoint_hits(pred, "prob", actual_top5, checkpoints)
print_checkpoint_report(results)

metrics = evaluate_binary_model(pred["postseason"], pred["prob_raw"])
print()
print_metrics(metrics, label="2025 검증")

# ──────────────────────────────────────────────
# 5. 확률 추이 차트
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
teams  = sorted(pred["team"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(teams)))

for team, color in zip(teams, colors):
    t  = pred[pred["team"] == team].sort_values("games_played_ratio")
    ls = "-" if team in actual_top5 else "--"
    lw = 2.2 if team in actual_top5 else 1.0
    ax.plot(t["games_played_ratio"] * 144, t["prob"],
            label=team, color=color, linestyle=ls, linewidth=lw)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("경기 수 (진행)")
ax.set_ylabel("포스트시즌 예측 확률")
ax.set_title(
    f"2025 시즌 팀별 포스트시즌 확률 추이\n"
    f"(학습: 2017~2024 / 피처: {len(FEATURE_COLS)}개 / 실선: 실제 진출팀)"
)
ax.legend(loc="upper left", ncol=2, fontsize=9)
ax.xaxis.set_major_locator(mticker.MultipleLocator(18))
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "validate_2025_trend.png"), dpi=120)
plt.show()

# ──────────────────────────────────────────────
# 6. 최종 시점 바 차트
# ──────────────────────────────────────────────
final      = pred.sort_values("date").groupby("team").last().reset_index()
final      = final.sort_values("prob", ascending=True)
colors_bar = ["steelblue" if t in actual_top5 else "lightgray" for t in final["team"]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(final["team"], final["prob"], color=colors_bar)
ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 기준선")
ax.set_xlabel("포스트시즌 예측 확률")
ax.set_title(
    f"2025 최종 시점 팀별 예측 확률\n"
    f"(학습: 2017~2024, 파란색: 실제 포스트시즌 진출팀)"
)
ax.legend()
for bar, val in zip(bars, final["prob"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "validate_2025_bar.png"), dpi=120)
plt.show()

print("\n차트 저장 완료: notebooks/experiments/jh/Validation/assets/validate_2025_*.png")
