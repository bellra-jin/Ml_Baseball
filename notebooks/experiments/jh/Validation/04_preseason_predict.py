"""
04_preseason_predict.py

시즌 시작 전 예측 — 경기 데이터 없이 전년도 성적만으로 포스트시즌 예측.
각 팀-시즌당 하나의 행(prev_ 피처만 사용).
PREDICT_SEASON 직전까지의 모든 시즌을 학습에 사용.

실행: uv run python "notebooks/experiments/jh /04_preseason_predict.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(os.path.dirname(__file__), "../../..")

# 예측 대상 시즌 (이 시즌 직전까지의 모든 데이터를 학습에 사용)
PREDICT_SEASON = 2025

# ──────────────────────────────────────────────
# 1. 데이터 로드 → 팀-시즌 단위로 축약
# ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

PREV_COLS = [
    "prev_pythagorean_win_rate",
    "prev_run_differential",
    "prev_team_era",
    "prev_k_bb_ratio",
    "prev_top5_hitter_ops_avg",
    "prev_ace_era",
    "prev_iso",
    "prev_ops_concentration",
    "prev_bb_rate",
]

# 팀-시즌당 하나의 행만 (prev_ 피처는 시즌 내 동일값)
season_df = (
    df.sort_values("date")
    .groupby(["season", "team"])
    .first()
    .reset_index()
)[["season", "team", "postseason"] + PREV_COLS]

missing = season_df[PREV_COLS].isna().sum()
if missing.any():
    print(f"[경고] 결측치:\n{missing[missing > 0]}\n")

print(f"전체 팀-시즌 행 수: {len(season_df)}")
print(f"prev_ 피처 수: {len(PREV_COLS)}개\n")

# ──────────────────────────────────────────────
# 2. 학습 / 예측 분리
#    PREDICT_SEASON 이전의 모든 시즌을 학습에 사용
#    (prev_ 피처가 없는 첫 시즌은 dropna로 자동 제외)
# ──────────────────────────────────────────────
train = season_df[season_df["season"] < PREDICT_SEASON].dropna(subset=PREV_COLS)
pred  = season_df[season_df["season"] == PREDICT_SEASON].dropna(subset=PREV_COLS)

X_train = train[PREV_COLS]
y_train = train["postseason"]
X_pred  = pred[PREV_COLS]

print(f"학습: {train.shape}  시즌 {sorted(train['season'].unique())}")
print(f"예측: {pred.shape}   시즌 {PREDICT_SEASON}\n")

# 실제 정답 (검증용)
actual_top5 = set(pred[pred["postseason"] == 1]["team"].unique())
print(f"[실제 {PREDICT_SEASON} 포스트시즌] {sorted(actual_top5)}\n")

# ──────────────────────────────────────────────
# 3. 모델 학습
#    (행 수가 ~70개로 적으므로 강하게 정규화)
# ──────────────────────────────────────────────
pos_w = (y_train == 0).sum() / (y_train == 1).sum()
s_min, s_max = train["season"].min(), train["season"].max()
season_w = (0.3 + 0.7 * (train["season"] - s_min) / (s_max - s_min)).values

xgb = XGBClassifier(
    n_estimators=100, max_depth=2, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=2.0, reg_lambda=10.0, min_child_weight=5,
    scale_pos_weight=pos_w, eval_metric="logloss", random_state=42,
)
lgbm = LGBMClassifier(
    n_estimators=100, max_depth=2, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=2.0, reg_lambda=10.0, min_child_samples=5,
    scale_pos_weight=pos_w, random_state=42, verbose=-1,
)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=3, min_samples_leaf=5,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1,
)
lr = LogisticRegression(
    C=0.5, class_weight="balanced", max_iter=1000, random_state=42,
)

print("모델 학습 중...")
xgb.fit(X_train, y_train, sample_weight=season_w)
lgbm.fit(X_train, y_train, sample_weight=season_w)
rf.fit(X_train, y_train, sample_weight=season_w)
lr.fit(X_train, y_train)
print("학습 완료\n")

# ──────────────────────────────────────────────
# 4. 예측 (각 모델 + 앙상블)
# ──────────────────────────────────────────────
pred = pred.copy()
pred["prob_xgb"]  = xgb.predict_proba(X_pred)[:, 1]
pred["prob_lgbm"] = lgbm.predict_proba(X_pred)[:, 1]
pred["prob_rf"]   = rf.predict_proba(X_pred)[:, 1]
pred["prob_lr"]   = lr.predict_proba(X_pred)[:, 1]
pred["prob_ens"]  = (
    pred["prob_xgb"] + pred["prob_lgbm"] +
    pred["prob_rf"]  + pred["prob_lr"]
) / 4

# ──────────────────────────────────────────────
# 5. 결과 출력
# ──────────────────────────────────────────────
print(f"{'모델':<12} {'예측 상위 5팀':<40} {'적중':>6}")
print("─" * 62)

model_cols = {
    "XGBoost":   "prob_xgb",
    "LightGBM":  "prob_lgbm",
    "RF":        "prob_rf",
    "LogReg":    "prob_lr",
    "Ensemble":  "prob_ens",
}

for name, col in model_cols.items():
    top5 = set(pred.nlargest(5, col)["team"])
    hit  = len(top5 & actual_top5)
    print(f"{name:<12} {str(sorted(top5)):<40} {hit}/5")

print()
# 앙상블 확률 상세
result = pred[["team", "postseason"] + list(model_cols.values())].copy()
result = result.sort_values("prob_ens", ascending=False).reset_index(drop=True)
result["rank_pred"] = result.index + 1
result["correct"] = result["team"].isin(actual_top5)

print("─" * 70)
print(f"{'순위':>4}  {'팀':>6}  {'실제':>6}  {'XGB':>6}  {'LGBM':>6}  {'RF':>6}  {'LR':>6}  {'앙상블':>6}")
print("─" * 70)
for _, row in result.iterrows():
    mark = "★" if row["correct"] else "  "
    print(
        f"{row['rank_pred']:>4}  {row['team']:>6}  {mark}      "
        f"{row['prob_xgb']:.3f}  {row['prob_lgbm']:.3f}  "
        f"{row['prob_rf']:.3f}  {row['prob_lr']:.3f}  {row['prob_ens']:.3f}"
    )

# ROC-AUC (10팀 단일 예측이라 참고용)
try:
    auc = roc_auc_score(pred["postseason"], pred["prob_ens"])
    print(f"\nROC-AUC (앙상블): {auc:.4f}")
except Exception as e:
    print(f"\nROC-AUC 계산 불가: {e}")

# ──────────────────────────────────────────────
# 6. 리브-원-아웃 교차검증 (시즌 단위)
#    — 학습 풀 전체(PREDICT_SEASON 이전)에서 한 시즌씩 제외해 검증
# ──────────────────────────────────────────────
train_seasons = sorted(train["season"].unique())
print("\n" + "=" * 50)
print(f"시즌별 Leave-One-Out 검증 ({min(train_seasons)}~{max(train_seasons)})")
print("=" * 50)
print(f"{'시즌':<6} {'예측 상위 5팀':<40} {'적중':>6}")
print("─" * 55)

loo_hits = []
for test_season in train_seasons:
    loo_train = season_df[
        (season_df["season"] != test_season) &
        (season_df["season"] < PREDICT_SEASON)
    ].dropna(subset=PREV_COLS)
    loo_test = season_df[
        season_df["season"] == test_season
    ].dropna(subset=PREV_COLS)

    if loo_test.empty or loo_train.empty:
        continue

    Xtr = loo_train[PREV_COLS]
    ytr = loo_train["postseason"]
    Xte = loo_test[PREV_COLS]

    sw = (0.3 + 0.7 * (loo_train["season"] - loo_train["season"].min()) /
          max(1, loo_train["season"].max() - loo_train["season"].min())).values

    pw = max(1, (ytr == 0).sum() / max(1, (ytr == 1).sum()))

    m_xgb = XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=10.0, min_child_weight=5,
        scale_pos_weight=pw, eval_metric="logloss", random_state=42,
    )
    m_lgbm = LGBMClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=10.0, min_child_samples=3,
        scale_pos_weight=pw, random_state=42, verbose=-1,
    )
    m_rf = RandomForestClassifier(
        n_estimators=300, max_depth=3, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    m_lr = LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1000, random_state=42,
    )

    m_xgb.fit(Xtr, ytr, sample_weight=sw)
    m_lgbm.fit(Xtr, ytr, sample_weight=sw)
    m_rf.fit(Xtr, ytr, sample_weight=sw)
    m_lr.fit(Xtr, ytr)

    prob_ens = (
        m_xgb.predict_proba(Xte)[:, 1] +
        m_lgbm.predict_proba(Xte)[:, 1] +
        m_rf.predict_proba(Xte)[:, 1] +
        m_lr.predict_proba(Xte)[:, 1]
    ) / 4

    loo_test = loo_test.copy()
    loo_test["prob"] = prob_ens
    actual = set(loo_test[loo_test["postseason"] == 1]["team"])
    top5   = set(loo_test.nlargest(5, "prob")["team"])
    hit    = len(top5 & actual)
    loo_hits.append(hit)
    print(f"{test_season:<6} {str(sorted(top5)):<40} {hit}/5")

print("─" * 55)
print(f"평균 적중: {np.mean(loo_hits):.2f}/5  (최저 {min(loo_hits)}/5, 최고 {max(loo_hits)}/5)")
print(f"\n→ {PREDICT_SEASON} 프리시즌 예측(앙상블): {sorted(set(pred.nlargest(5, 'prob_ens')['team']))}")
print(f"→ 실제 {PREDICT_SEASON} 포스트시즌:      {sorted(actual_top5)}")

# ──────────────────────────────────────────────
# 7. 차트 1: 프리시즌 예측 확률 바 차트
# ──────────────────────────────────────────────
bar_data = pred[["team", "prob_ens", "postseason"]].sort_values("prob_ens", ascending=True)
colors_bar = ["steelblue" if t in actual_top5 else "lightgray" for t in bar_data["team"]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(bar_data["team"], bar_data["prob_ens"], color=colors_bar)
ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 기준선")
ax.set_xlabel("포스트시즌 예측 확률")
ax.set_title(
    f"{PREDICT_SEASON} 프리시즌 팀별 예측 확률\n"
    f"(학습: {min(train_seasons)}~{max(train_seasons)} / prev_ 피처 {len(PREV_COLS)}개 / 파란색: 실제 포스트시즌 진출팀)"
)
ax.legend()
for bar, val in zip(bars, bar_data["prob_ens"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
out_bar = os.path.join(BASE, f"data/modeling/preseason_{PREDICT_SEASON}_bar.png")
plt.savefig(out_bar, dpi=120)
plt.show()

# ──────────────────────────────────────────────
# 8. 차트 2: LOO 시즌별 적중률
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ["steelblue" if h >= 4 else "coral" if h <= 2 else "gold" for h in loo_hits]
bars = ax.bar(train_seasons, loo_hits, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.axhline(np.mean(loo_hits), color="navy", linestyle="--", linewidth=1.5,
           label=f"평균 {np.mean(loo_hits):.2f}/5")
ax.axhline(2.5, color="gray", linestyle=":", linewidth=1, label="랜덤 기대값 2.5/5")
ax.set_xticks(train_seasons)
ax.set_xlabel("검증 시즌")
ax.set_ylabel("적중 팀 수 (/ 5)")
ax.set_ylim(0, 5.8)
ax.set_yticks(range(6))
ax.set_title(
    f"Leave-One-Out 검증: 시즌별 포스트시즌 예측 적중률\n"
    f"(prev_ 피처 {len(PREV_COLS)}개만 사용 — 경기 데이터 없음)"
)
ax.legend(fontsize=10)
for bar, h in zip(bars, loo_hits):
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h}/5",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
out_loo = os.path.join(BASE, f"data/modeling/preseason_loo_bar.png")
plt.savefig(out_loo, dpi=120)
plt.show()

print(f"\n차트 저장: {out_bar}")
print(f"차트 저장: {out_loo}")
