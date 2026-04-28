"""
02_slim_features_experiment.py

핵심 피처만 사용했을 때 vs 전체 피처(01번) 결과 비교.

실행: uv run python "notebooks/experiments/jh /02_slim_features_experiment.py"
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

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# 1. 피처 정의
# ──────────────────────────────────────────────
from src.utils.config import FEATURE_COLS as FULL_COLS

SLIM_COLS = [
    # 현재 시즌 순위 지표
    "rank",
    "win_rate",
    "games_behind",
    "games_behind_5th",
    "wins_to_5th",
    "games",
    "remaining_games",
    "games_played_ratio",

    # 최근 N경기 승률
    "recent10_win_rate",
    "recent20_win_rate",
    "recent30_win_rate",

    # 홈/원정
    "home_win_rate",
    "away_win_rate",
    "home_away_win_diff",

    # 연속 기록
    "streak_type",
    "streak_count",

    # 추세
    "win_rate_delta_30d",
    "rank_delta_30d",

    # 전년도 핵심 지표만 (9개)
    "prev_pythagorean_win_rate",   # 종합 전력 (운 제거)
    "prev_run_differential",       # 득실차 (공+수 종합)
    "prev_team_era",               # 투수력
    "prev_k_bb_ratio",             # 투수 제구 (ERA보다 안정적)
    "prev_top5_hitter_ops_avg",    # 타격력
    "prev_ace_era",                # 에이스 ERA
    "prev_iso",                    # 장타력
    "prev_ops_concentration",      # 타선 균형도
    "prev_bb_rate",                # 선구안
]

print(f"전체 피처: {len(FULL_COLS)}개  →  핵심 피처: {len(SLIM_COLS)}개")
print(f"제거된 prev_ 피처: {len(FULL_COLS) - len(SLIM_COLS)}개\n")


# ──────────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "../../..")
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

train = df[df["season"] != 2025].copy()
test  = df[df["season"] == 2025].copy()

print(f"학습: {train.shape}  ({sorted(train['season'].unique())})")
print(f"테스트: {test.shape}  (2025)\n")

# 가중치 (최근 시즌 우선, 시즌 가중치는 제거 — 144경기 모두 동등)
s_min, s_max = train["season"].min(), train["season"].max()
season_w = 0.3 + 0.7 * (train["season"] - s_min) / (s_max - s_min)
sample_weights = season_w.values


# ──────────────────────────────────────────────
# 3. 모델 학습 함수
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


def predict_ensemble(models, X):
    xgb, lgbm, rf = models
    return (
        xgb.predict_proba(X)[:, 1] +
        lgbm.predict_proba(X)[:, 1] +
        rf.predict_proba(X)[:, 1]
    ) / 3


def normalize_prob(series_by_date):
    """일별 스냅샷 내 상대 확률 정규화 (합 → 5, clip 1.0)"""
    return series_by_date.transform(lambda x: (x / x.sum() * 5).clip(upper=1.0))


# ──────────────────────────────────────────────
# 4. 전체 피처 학습
# ──────────────────────────────────────────────
print("=" * 50)
print("[FULL] 전체 피처 학습 중...")
y_train = train["postseason"]
y_test  = test["postseason"]

models_full = build_ensemble(train[FULL_COLS], y_train, sample_weights)
test = test.copy()
test["prob_full"] = predict_ensemble(models_full, test[FULL_COLS])
test["prob_full_norm"] = normalize_prob(test.groupby("date")["prob_full"])


# ──────────────────────────────────────────────
# 5. 핵심 피처 학습
# ──────────────────────────────────────────────
print("[SLIM] 핵심 피처 학습 중...")
models_slim = build_ensemble(train[SLIM_COLS], y_train, sample_weights)
test["prob_slim"] = predict_ensemble(models_slim, test[SLIM_COLS])
test["prob_slim_norm"] = normalize_prob(test.groupby("date")["prob_slim"])


# ──────────────────────────────────────────────
# 6. 시점별 적중률 비교
# ──────────────────────────────────────────────
checkpoints = {
    "50%  (72경기)": 0.50,
    "75% (108경기)": 0.75,
    "90% (130경기)": 0.90,
    "최종 (144경기)": 1.00,
}

actual_top5 = set(test[test["postseason"] == 1]["team"].unique())
print(f"\n2025 실제 포스트시즌: {sorted(actual_top5)}\n")
print(f"{'시점':<16} {'FULL(56개)':>12} {'SLIM(26개)':>12}")
print("-" * 42)

for label, ratio in checkpoints.items():
    snap   = test[test["games_played_ratio"] <= ratio]
    latest = snap.sort_values("date").groupby("team").last().reset_index()

    top5_full = set(latest.nlargest(5, "prob_full_norm")["team"])
    top5_slim = set(latest.nlargest(5, "prob_slim_norm")["team"])

    hit_full = len(top5_full & actual_top5)
    hit_slim = len(top5_slim & actual_top5)

    diff = "▲" if hit_slim > hit_full else ("▼" if hit_slim < hit_full else "  ")
    print(f"{label:<16} {hit_full}/5 {sorted(top5_full)!s:<30}  {hit_slim}/5 {diff} {sorted(top5_slim)}")

# ROC-AUC 비교
auc_full = roc_auc_score(y_test, test["prob_full"])
auc_slim = roc_auc_score(y_test, test["prob_slim"])
print(f"\nROC-AUC — FULL: {auc_full:.4f}  SLIM: {auc_slim:.4f}")


# ──────────────────────────────────────────────
# 7. 확률 추이 비교 차트
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
teams  = sorted(test["team"].unique())

for ax, (prob_col, title) in zip(axes, [
    ("prob_full_norm", f"전체 피처 ({len(FULL_COLS)}개)"),
    ("prob_slim_norm", f"핵심 피처 ({len(SLIM_COLS)}개)"),
]):
    for team, color in zip(teams, colors):
        t  = test[test["team"] == team].sort_values("games_played_ratio")
        ls = "-" if team in actual_top5 else "--"
        lw = 2.2 if team in actual_top5 else 1.0
        ax.plot(t["games_played_ratio"] * 144, t[prob_col],
                label=team, color=color, linestyle=ls, linewidth=lw)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("포스트시즌 예측 확률")
    ax.set_title(f"2025 팀별 확률 추이 — {title}\n(실선: 실제 포스트시즌 진출팀)")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(18))

axes[-1].set_xlabel("경기 수 (진행)")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "data/modeling/compare_full_vs_slim.png"), dpi=120)
plt.show()
print("\n차트 저장: data/modeling/compare_full_vs_slim.png")


# ──────────────────────────────────────────────
# 8. 최종 시점 바 차트 비교
# ──────────────────────────────────────────────
final = test.sort_values("date").groupby("team").last().reset_index()
final = final.sort_values("prob_slim_norm", ascending=True)

colors_bar = ["steelblue" if t in actual_top5 else "lightgray" for t in final["team"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (col, title) in zip(axes, [
    ("prob_full_norm", f"FULL ({len(FULL_COLS)}개)"),
    ("prob_slim_norm", f"SLIM ({len(SLIM_COLS)}개)"),
]):
    bars = ax.barh(final["team"], final[col], color=colors_bar)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("포스트시즌 예측 확률")
    ax.set_title(f"2025 최종 시점\n(파란색: 실제 포스트시즌 진출팀)\n{title}")
    for bar, val in zip(bars, final[col]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "data/modeling/compare_full_vs_slim_bar.png"), dpi=120)
plt.show()
print("차트 저장: data/modeling/compare_full_vs_slim_bar.png")


# ──────────────────────────────────────────────
# 9. 시점별 적중률 바 차트
# ──────────────────────────────────────────────
cp_labels = list(checkpoints.keys())
hits_full, hits_slim = [], []

for ratio in checkpoints.values():
    snap   = test[test["games_played_ratio"] <= ratio]
    latest = snap.sort_values("date").groupby("team").last().reset_index()
    hits_full.append(len(set(latest.nlargest(5, "prob_full_norm")["team"]) & actual_top5))
    hits_slim.append(len(set(latest.nlargest(5, "prob_slim_norm")["team"]) & actual_top5))

x = np.arange(len(cp_labels))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars_f = ax.bar(x - w/2, hits_full, w, label=f"FULL ({len(FULL_COLS)}개)", color="steelblue", alpha=0.8)
bars_s = ax.bar(x + w/2, hits_slim, w, label=f"SLIM ({len(SLIM_COLS)}개)", color="coral", alpha=0.8)

for bar in bars_f:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{int(bar.get_height())}/5", ha="center", va="bottom", fontsize=11)
for bar in bars_s:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{int(bar.get_height())}/5", ha="center", va="bottom", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(cp_labels, fontsize=11)
ax.set_ylabel("적중 팀 수 (/ 5)")
ax.set_ylim(0, 6)
ax.set_yticks(range(6))
ax.axhline(5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_title("시점별 포스트시즌 예측 적중률\nFULL vs SLIM 피처 비교 (2025 검증)")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "data/modeling/compare_checkpoint_bar.png"), dpi=120)
plt.show()
print("차트 저장: data/modeling/compare_checkpoint_bar.png")
