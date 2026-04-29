"""
predict_2026_postseason.py

2016~2025 전체 데이터로 Strategy C 앙상블 모델을 학습하고
2026 시즌 포스트시즌 진출 확률을 예측한다.

피처셋  : TOP_FEATURES (상위 20개, 중요도 기반 선정)
앙상블  : Strategy C — LR 25% + RF 25% + lightXGB 25% + lightLGBM 25%
           과적합 갭 ~0.12 (기존 XGB+LGBM+RF 대비 0.07 개선)

검증 (LOSO-CV 2018~2025):
  V1. 성능 스코어카드   — 시즌별 지표 히트맵
  V2. 과적합 갭         — Train vs Test AUC 폴드별
  V3. 로스 커브         — XGB · LGBM 부스팅 라운드별 logloss
  V4. 캘리브레이션      — 확률 신뢰도 + 분포
  V5. 체크포인트 적중   — 시즌 50%·75%·90%·최종 시점 상위 5팀 예측

결과물: notebooks/experiments/jh/kbo_prediction_2026/
  validation/          — LOSO-CV 검증 차트 5종
  predict_2026_*.png   — 2026 예측 차트 7종
  predict_2026_result.csv

실행: uv run python "notebooks/experiments/jh/predict_2026_postseason.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as _lgb

from src.evaluation.metrics import (
    evaluate_binary_model, print_metrics, checkpoint_hits,
)
from src.utils.config import TOP_FEATURES

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
ROOT   = os.path.join(os.path.dirname(__file__), "../../..")
OUTDIR = os.path.join(os.path.dirname(__file__), "kbo_prediction_2026")
VALDIR = os.path.join(OUTDIR, "validation")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(VALDIR, exist_ok=True)

TEST_SEASONS = list(range(2017, 2026))


# ─────────────────────────────────────────────
# 스타일 설정
# ─────────────────────────────────────────────
BG        = "#F8F9FA"
GRAY_AXIS = "#AAAAAA"
ACCENT    = "#1B3F7A"
WARM_RED  = "#CC4444"
GREEN     = "#2E8B57"
ORANGE    = "#E07B20"
TOP5_COLORS  = ["#1B3F7A", "#2563A8", "#3E84C8", "#6BADD6", "#9ECAE1"]
BOTTOM_COLOR = "#D5D5D5"

TEAM_COLORS = {
    "KIA":  "#ea0029", "삼성": "#074CA1", "LG":  "#a50034",
    "두산":  "#1a1748", "KT":  "#000000", "SSG": "#ce0e2d",
    "롯데":  "#041E42", "한화": "#FC4E00", "NC":  "#315288", "키움": "#570514",
}
ALPHA_TOP    = 1.00
ALPHA_BOTTOM = 0.30


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
print("데이터 로드 중...")
train_df = pd.read_csv(os.path.join(ROOT, "data/modeling/train_dataset.csv"))
pred_df  = pd.read_csv(os.path.join(ROOT, "data/modeling/predict_dataset_2026.csv"))

train_df["date"] = pd.to_datetime(train_df["date"])
pred_df["date"]  = pd.to_datetime(pred_df["date"])

missing = [c for c in TOP_FEATURES if c not in train_df.columns]
if missing:
    print(f"[경고] 학습 데이터 누락 피처: {missing}")
    TOP_FEATURES = [c for c in TOP_FEATURES if c in train_df.columns]

missing_pred = [c for c in TOP_FEATURES if c not in pred_df.columns]
if missing_pred:
    print(f"[경고] 예측 데이터 누락 피처: {missing_pred}")
    TOP_FEATURES = [c for c in TOP_FEATURES if c in pred_df.columns]

print(f"학습 데이터: {train_df.shape}  ({sorted(train_df['season'].unique())})")
print(f"예측 데이터: {pred_df.shape}")
print(f"예측 기간:   {pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
print(f"시즌 진행도: {pred_df['games_played_ratio'].max():.1%}")
print(f"사용 피처:   {len(TOP_FEATURES)}개  (Strategy C)\n")


# ─────────────────────────────────────────────
# 모델 빌더 (Strategy C)
# ─────────────────────────────────────────────
def _build_models(pos_w):
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C=0.1, max_iter=2000, random_state=42,
            class_weight="balanced", solver="lbfgs",
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


# ─────────────────────────────────────────────
# LOSO-CV 검증 (2018~2025)
# ─────────────────────────────────────────────
print("=" * 55)
print("LOSO-CV 검증  (Strategy C | 20 피처 | 2017~2025)")
print("=" * 55)

cv_records  = []
cv_rows     = []
xgb_losses  = {}
lgbm_losses = {}

for test_season in TEST_SEASONS:
    tr = train_df[train_df["season"] != test_season].copy()
    te = train_df[train_df["season"] == test_season].copy()

    X_tr, y_tr = tr[TOP_FEATURES], tr["postseason"]
    X_te, y_te = te[TOP_FEATURES], te["postseason"]

    s_min, s_max = tr["season"].min(), tr["season"].max()
    sw    = (0.3 + 0.7 * (tr["season"] - s_min) / max(s_max - s_min, 1)).values
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    X_es  = tr.loc[tr["season"] == s_max, TOP_FEATURES]
    y_es  = tr.loc[tr["season"] == s_max, "postseason"]

    lr, rf, xgb, lgbm = _build_models(pos_w)
    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr, sample_weight=sw)
    xgb.fit(X_tr, y_tr, sample_weight=sw,
            eval_set=[(X_tr, y_tr), (X_es, y_es)], verbose=False)
    lgbm.fit(X_tr, y_tr, sample_weight=sw,
             eval_set=[(X_tr, y_tr), (X_es, y_es)],
             eval_names=["train", "valid"],
             eval_metric="binary_logloss",
             callbacks=[_lgb.log_evaluation(period=0)])

    prob_te = _ensemble_proba(lr, rf, xgb, lgbm, X_te)
    prob_tr = _ensemble_proba(lr, rf, xgb, lgbm, X_tr)

    xgb_losses[test_season]  = xgb.evals_result()
    lgbm_losses[test_season] = lgbm.evals_result_

    m_te = evaluate_binary_model(y_te, prob_te)
    m_tr = evaluate_binary_model(y_tr, prob_tr)
    cv_records.append({
        "season":    test_season,
        "test_auc":  m_te["roc_auc"],
        "train_auc": m_tr["roc_auc"],
        "gap":       m_tr["roc_auc"] - m_te["roc_auc"],
        "brier":     m_te["brier"],
        "f1":        m_te["f1"],
        "precision": m_te["precision"],
        "recall":    m_te["recall"],
        "accuracy":  m_te["accuracy"],
    })

    row_pred = te[["season", "date", "team", "postseason", "games_played_ratio"]].copy()
    row_pred["prob"] = prob_te
    cv_rows.append(row_pred)
    print_metrics(m_te, label=str(test_season))

cv_df    = pd.DataFrame(cv_records)
all_rows = pd.concat(cv_rows, ignore_index=True)
gap_mean = cv_df["gap"].mean()

print(f"\n  Test AUC  = {cv_df['test_auc'].mean():.4f}  |  "
      f"Train AUC = {cv_df['train_auc'].mean():.4f}  |  "
      f"갭 = {gap_mean:.4f}  |  Brier = {cv_df['brier'].mean():.4f}\n")

# 시즌별 실제 포스트시즌 진출팀 (train_df에서 추출)
ACTUAL_TOP5 = {}
for season in TEST_SEASONS:
    labels = train_df[train_df["season"] == season].groupby("team")["postseason"].first()
    ACTUAL_TOP5[season] = set(labels[labels == 1].index)


# ─────────────────────────────────────────────
# 검증 V1: 성능 스코어카드 (히트맵)
# ─────────────────────────────────────────────
metric_cols  = ["test_auc", "train_auc", "gap", "f1", "precision", "recall", "brier"]
metric_names = ["Test AUC", "Train AUC", "Gap (과적합)", "F1", "Precision", "Recall", "Brier"]
lower_better = {"brier", "gap"}

scorecard = cv_df.set_index("season")[metric_cols]
n_m = len(metric_cols)
n_s = len(TEST_SEASONS)

norm_mat = np.zeros((n_m, n_s))
for j, col in enumerate(metric_cols):
    vals = scorecard[col].values.astype(float)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        n = (vals - vmin) / (vmax - vmin)
        norm_mat[j] = (1 - n) if col in lower_better else n
    else:
        norm_mat[j] = 0.5

fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(norm_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
               interpolation="nearest")

for j in range(n_m):
    for i in range(n_s):
        raw = scorecard[metric_cols[j]].iloc[i]
        tc  = "white" if norm_mat[j, i] < 0.2 or norm_mat[j, i] > 0.8 else "#333333"
        ax.text(i, j, f"{raw:.3f}", ha="center", va="center",
                fontsize=8.5, color=tc, fontweight="bold")

ax.set_xticks(range(n_s))
ax.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=10)
ax.set_yticks(range(n_m))
ax.set_yticklabels(metric_names, fontsize=10)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="상대적 성능 (초록=우수)")
ax.set_title("Strategy C LOSO-CV 성능 스코어카드 (2018~2025)",
             fontsize=14, fontweight="bold", color="#1B1B1B", pad=20)
ax.text(0.5, 1.025, "LR 25% + RF 25% + lightXGB 25% + lightLGBM 25%  |  피처 20개",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

plt.tight_layout()
out_v1 = os.path.join(VALDIR, "val_scorecard.png")
plt.savefig(out_v1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"[V1] 스코어카드 저장: {out_v1}")


# ─────────────────────────────────────────────
# 검증 V2: 과적합 갭 (Train vs Test AUC)
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

x = np.arange(n_s)
w = 0.35

ax1.bar(x - w/2, cv_df["train_auc"], w, color=WARM_RED, alpha=0.75,
        label="Train AUC", edgecolor="white")
ax1.bar(x + w/2, cv_df["test_auc"],  w, color=ACCENT,   alpha=0.85,
        label="Test AUC",  edgecolor="white")
ax1.axhline(cv_df["test_auc"].mean(), color=ACCENT, linewidth=1.5,
            linestyle="--", alpha=0.7, label=f"평균 Test {cv_df['test_auc'].mean():.3f}")
ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=10)
ax1.set_ylim(0.5, 1.12)
ax1.set_ylabel("ROC-AUC", fontsize=11, color="#444444")
ax1.set_title("Train vs Test AUC", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
ax1.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
for sp in ["top", "right"]:
    ax1.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax1.spines[sp].set_color(GRAY_AXIS)
ax1.tick_params(colors=GRAY_AXIS, labelsize=10)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

gap_colors = [GREEN if g < 0.10 else (ORANGE if g < 0.15 else WARM_RED) for g in cv_df["gap"]]
bars_gap = ax2.bar(x, cv_df["gap"], color=gap_colors, alpha=0.85, width=0.55, edgecolor="white")
ax2.axhline(gap_mean, color="#555555", linewidth=1.5, linestyle="--", alpha=0.8,
            label=f"평균 갭 {gap_mean:.3f}")
ax2.axhline(0.10, color=WARM_RED, linewidth=1.2, linestyle=":", alpha=0.6, label="갭 0.10")

for bar, val in zip(bars_gap, cv_df["gap"]):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#333333", fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=10)
ax2.set_ylim(0, cv_df["gap"].max() * 1.4)
ax2.set_ylabel("Train AUC − Test AUC", fontsize=11, color="#444444")
ax2.set_title("과적합 갭 (폴드별)", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
for sp in ["top", "right"]:
    ax2.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax2.spines[sp].set_color(GRAY_AXIS)
ax2.tick_params(colors=GRAY_AXIS, labelsize=10)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle("Strategy C 과적합 분석  (LR+RF+lightXGB+lightLGBM | 20 피처)",
             fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
plt.tight_layout()
out_v2 = os.path.join(VALDIR, "val_overfit_gap.png")
plt.savefig(out_v2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"[V2] 과적합 갭 저장: {out_v2}")


# ─────────────────────────────────────────────
# 검증 V3: 로스 커브 (XGB + LGBM)
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

fold_cmap = mcm.get_cmap("tab10")

for i, season in enumerate(TEST_SEASONS):
    color = fold_cmap(i)
    lw    = 2.2 if season == TEST_SEASONS[-1] else 1.0
    alpha = 1.0 if season == TEST_SEASONS[-1] else 0.40

    # XGB
    xr      = xgb_losses[season]
    tr_key  = list(xr.keys())[0]
    va_key  = list(xr.keys())[1]
    rds     = range(1, len(xr[tr_key]["logloss"]) + 1)
    ax1.plot(rds, xr[tr_key]["logloss"], color=color, lw=lw, alpha=alpha, ls="-")
    ax1.plot(rds, xr[va_key]["logloss"], color=color, lw=lw, alpha=alpha, ls="--")

    # LGBM
    lr_     = lgbm_losses[season]
    tr_l    = lr_["train"]["binary_logloss"]
    va_l    = lr_["valid"]["binary_logloss"]
    rds_l   = range(1, len(tr_l) + 1)
    ax2.plot(rds_l, tr_l, color=color, lw=lw, alpha=alpha, ls="-")
    ax2.plot(rds_l, va_l, color=color, lw=lw, alpha=alpha, ls="--")

legend_els = (
    [Line2D([0], [0], color=fold_cmap(i), lw=1.5, label=str(s)) for i, s in enumerate(TEST_SEASONS)] +
    [Line2D([0], [0], color="#555", lw=1.5, ls="-",  label="Train"),
     Line2D([0], [0], color="#555", lw=1.5, ls="--", label="Valid")]
)

for ax, title in [(ax1, "XGBoost 로스 커브"), (ax2, "LightGBM 로스 커브")]:
    ax.set_xlabel("부스팅 라운드", fontsize=11, color="#444444")
    ax.set_ylabel("Log Loss", fontsize=11, color="#444444")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(GRAY_AXIS)
    ax.tick_params(colors=GRAY_AXIS, labelsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

ax2.legend(handles=legend_els, fontsize=8.5, loc="upper right",
           framealpha=0.9, edgecolor="#DDDDDD", ncol=2)

fig.suptitle(f"LOSO-CV 로스 커브  (각 선 = 1 폴드 | 굵은 선 = {TEST_SEASONS[-1]} 폴드 | 실선=Train / 점선=Valid)",
             fontsize=13, fontweight="bold", color="#1B1B1B", y=1.02)
plt.tight_layout()
out_v3 = os.path.join(VALDIR, "val_loss_curves.png")
plt.savefig(out_v3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"[V3] 로스 커브 저장: {out_v3}")


# ─────────────────────────────────────────────
# 검증 V4: 캘리브레이션 + 분포
# ─────────────────────────────────────────────
prob_all   = all_rows["prob"].values
actual_all = all_rows["postseason"].values

frac_pos, mean_pred = calibration_curve(actual_all, prob_all, n_bins=8, strategy="uniform")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="완벽 캘리브레이션")
ax1.plot(mean_pred, frac_pos, "o-", color=ACCENT, lw=2.0, ms=8, label="Strategy C")

for xv, yv in zip(mean_pred, frac_pos):
    diff  = yv - xv
    color = GREEN if abs(diff) < 0.05 else WARM_RED
    ax1.annotate(f"{diff:+.2f}", (xv, yv),
                 xytext=(xv + 0.01, yv + 0.025), fontsize=8,
                 color=color, fontweight="bold")

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("예측 확률", fontsize=11, color="#444444")
ax1.set_ylabel("실제 양성 비율", fontsize=11, color="#444444")
ax1.set_title("Reliability Diagram", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
ax1.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
for sp in ["top", "right"]:
    ax1.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax1.spines[sp].set_color(GRAY_AXIS)
ax1.tick_params(colors=GRAY_AXIS, labelsize=10)
ax1.set_axisbelow(True)
ax1.xaxis.grid(True, color="#E8E8E8", linewidth=0.8)
ax1.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

bins = np.linspace(0, 1, 25)
ax2.hist(prob_all[actual_all == 0], bins=bins, color=WARM_RED, alpha=0.65,
         label="비진출 (0)", density=True)
ax2.hist(prob_all[actual_all == 1], bins=bins, color=GREEN, alpha=0.65,
         label="진출 (1)", density=True)
ax2.axvline(0.5, color="#555555", lw=1.2, ls="--", alpha=0.7)
ax2.set_xlabel("예측 확률", fontsize=11, color="#444444")
ax2.set_ylabel("밀도", fontsize=11, color="#444444")
ax2.set_title("예측 확률 분포 (클래스별)", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
ax2.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
for sp in ["top", "right"]:
    ax2.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax2.spines[sp].set_color(GRAY_AXIS)
ax2.tick_params(colors=GRAY_AXIS, labelsize=10)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle("예측 확률 캘리브레이션 및 분포 (LOSO-CV 전체)",
             fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
plt.tight_layout()
out_v4 = os.path.join(VALDIR, "val_calibration.png")
plt.savefig(out_v4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"[V4] 캘리브레이션 저장: {out_v4}")


# ─────────────────────────────────────────────
# 검증 V5: 체크포인트 적중률
# ─────────────────────────────────────────────
CHECKPOINTS = {"50% 시점": 0.50, "75% 시점": 0.75, "90% 시점": 0.90, "최종 시점": 1.01}

cp_results = {}
for season in TEST_SEASONS:
    if season not in ACTUAL_TOP5:
        continue
    season_rows = all_rows[all_rows["season"] == season]
    cp_results[season] = checkpoint_hits(season_rows, "prob", ACTUAL_TOP5[season], CHECKPOINTS)

cp_labels_list = list(CHECKPOINTS.keys())
cp_hit_avg = {lbl: [cp_results[s][lbl]["hit"] for s in TEST_SEASONS if s in cp_results]
              for lbl in cp_labels_list}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

cp_means = [np.mean(cp_hit_avg[lbl]) if cp_hit_avg[lbl] else 0 for lbl in cp_labels_list]
cp_stds  = [np.std(cp_hit_avg[lbl])  if cp_hit_avg[lbl] else 0 for lbl in cp_labels_list]
cp_colors = [GREEN if m >= 4 else (ORANGE if m >= 3 else WARM_RED) for m in cp_means]

bars_cp = ax1.bar(cp_labels_list, cp_means, color=cp_colors, alpha=0.85,
                  width=0.5, edgecolor="white")
ax1.errorbar(cp_labels_list, cp_means, yerr=cp_stds, fmt="none",
             capsize=5, ecolor="#555555", elinewidth=1.5)
ax1.axhline(3, color=ORANGE, lw=1.2, ls="--", alpha=0.7, label="3/5 기준")
ax1.axhline(4, color=GREEN,  lw=1.2, ls="--", alpha=0.7, label="4/5 기준")

for bar, mean in zip(bars_cp, cp_means):
    ax1.text(bar.get_x() + bar.get_width()/2, mean + 0.06,
             f"{mean:.2f}/5", ha="center", va="bottom", fontsize=10,
             color="#333333", fontweight="bold")

ax1.set_ylim(0, 5.8)
ax1.set_ylabel("평균 적중 팀 수 (/5)", fontsize=11, color="#444444")
ax1.set_title("시점별 평균 적중 수", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
ax1.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
for sp in ["top", "right"]:
    ax1.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax1.spines[sp].set_color(GRAY_AXIS)
ax1.tick_params(colors=GRAY_AXIS, labelsize=10)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

valid_seasons = [s for s in TEST_SEASONS if s in cp_results]
final_hits    = [cp_results[s]["최종 시점"]["hit"] for s in valid_seasons]
hit_colors    = [GREEN if h >= 4 else (ORANGE if h == 3 else WARM_RED) for h in final_hits]

bars_fin = ax2.bar(range(len(valid_seasons)), final_hits,
                   color=hit_colors, alpha=0.85, width=0.55, edgecolor="white")
for bar, val in zip(bars_fin, final_hits):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.06,
             f"{val}/5", ha="center", va="bottom", fontsize=10,
             color="#333333", fontweight="bold")

ax2.set_xticks(range(len(valid_seasons)))
ax2.set_xticklabels([str(s) for s in valid_seasons], fontsize=10)
ax2.set_ylim(0, 5.8)
ax2.set_ylabel("적중 팀 수 (/5)", fontsize=11, color="#444444")
ax2.set_title("시즌별 최종 시점 적중 수", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
for sp in ["top", "right"]:
    ax2.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax2.spines[sp].set_color(GRAY_AXIS)
ax2.tick_params(colors=GRAY_AXIS, labelsize=10)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle("체크포인트 적중률 — 포스트시즌 상위 5팀 예측 (LOSO-CV)",
             fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
plt.tight_layout()
out_v5 = os.path.join(VALDIR, "val_checkpoint.png")
plt.savefig(out_v5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"[V5] 체크포인트 저장: {out_v5}")

print("\n검증 완료.")
print(f"  V1 스코어카드    : {out_v1}")
print(f"  V2 과적합 갭     : {out_v2}")
print(f"  V3 로스 커브     : {out_v3}")
print(f"  V4 캘리브레이션  : {out_v4}")
print(f"  V5 체크포인트    : {out_v5}\n")


# ─────────────────────────────────────────────
# 최종 모델 학습 (전체 2017~2025)
# ─────────────────────────────────────────────
print("=" * 55)
print("최종 모델 학습 (2017~2025 전체)")
print("=" * 55)

X_train = train_df[TOP_FEATURES]
y_train = train_df["postseason"]

s_min, s_max = train_df["season"].min(), train_df["season"].max()
sample_weight = (0.3 + 0.7 * (train_df["season"] - s_min) / (s_max - s_min)).values
pos_w         = (y_train == 0).sum() / (y_train == 1).sum()

lr_f, rf_f, xgb_f, lgbm_f = _build_models(pos_w)

lr_f.fit(X_train, y_train)
print("  LogisticRegression 완료")

rf_f.fit(X_train, y_train, sample_weight=sample_weight)
print("  RandomForest 완료")

xgb_f.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
print("  XGBoost 완료")

lgbm_f.fit(X_train, y_train, sample_weight=sample_weight,
           callbacks=[_lgb.log_evaluation(period=0)])
print("  LightGBM 완료\n")


# ─────────────────────────────────────────────
# 2026 예측
# ─────────────────────────────────────────────
X_pred   = pred_df[TOP_FEATURES]
prob_raw = _ensemble_proba(lr_f, rf_f, xgb_f, lgbm_f, X_pred)

pred_df = pred_df.copy()
pred_df["prob_raw"] = prob_raw

pred_df["prob_norm"] = pred_df.groupby("date")["prob_raw"].transform(
    lambda x: (x / x.sum() * 5).clip(upper=1.0)
)

latest     = pred_df.sort_values("date").groupby("team").last().reset_index()
latest     = latest.sort_values("prob_norm", ascending=False)
top5_teams = set(latest.head(5)["team"])

print("=" * 50)
print(f"[2026 포스트시즌 예측] 기준: {latest['date'].iloc[0].date()}")
print("=" * 50)
print(f"{'순위':<4} {'팀':<8} {'확률':>8}  {'진행도':>6}")
print("─" * 35)
for i, row in enumerate(latest.itertuples(), 1):
    marker = "★" if i <= 5 else "  "
    print(f"{marker}{i:>2}위  {row.team:<8} {row.prob_norm:>7.1%}  ({row.games_played_ratio:.1%})")
print()


# ─────────────────────────────────────────────
# 결과 CSV 저장
# ─────────────────────────────────────────────
result_csv = latest[["team", "date", "games", "games_played_ratio", "prob_raw", "prob_norm"]].copy()
result_csv.columns = ["팀", "기준일", "경기수", "시즌진행도", "원시확률", "정규화확률"]
result_csv.to_csv(os.path.join(OUTDIR, "predict_2026_result.csv"),
                  index=False, encoding="utf-8-sig")
print(f"결과 CSV 저장: {OUTDIR}/predict_2026_result.csv\n")


# ─────────────────────────────────────────────
# 공통 변수 (예측 차트용)
# ─────────────────────────────────────────────
ref_date  = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y.%m.%d")
ref_ratio = latest["games_played_ratio"].mean()
bar_order = latest.sort_values("prob_norm", ascending=True).reset_index(drop=True)
top5_ranks = {team: i for i, team in enumerate(latest.head(5)["team"])}

bar_colors = []
for team in bar_order["team"]:
    if team in top5_ranks:
        bar_colors.append(TOP5_COLORS[top5_ranks[team]])
    else:
        bar_colors.append(BOTTOM_COLOR)

SUBTITLE = "LR 25% + RF 25% + lightXGB 25% + lightLGBM 25%  |  피처 20개"


# ─────────────────────────────────────────────
# 차트 1: 포스트시즌 확률 바 차트
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
ax.set_facecolor(BG)

bars = ax.barh(bar_order["team"], bar_order["prob_norm"],
               color=bar_colors, height=0.6, edgecolor="white", linewidth=0.5)

ax.axhline(4.5, color="#888888", linewidth=1.2, linestyle="--", alpha=0.6)
ax.text(0.01, 4.55, "── 포스트시즌 컷라인", color="#888888", fontsize=9, va="bottom")
ax.axvline(0.5, color="#CC4444", linewidth=1.0, linestyle=":", alpha=0.8)

for bar, val, team in zip(bars, bar_order["prob_norm"], bar_order["team"]):
    is_top5 = team in top5_teams
    ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left", fontsize=11,
            color="#222222" if is_top5 else "#666666",
            fontweight="bold" if is_top5 else "normal")

for i, (team, val) in enumerate(zip(bar_order["team"], bar_order["prob_norm"])):
    rank   = len(bar_order) - i
    marker = "★" if team in top5_teams else f"{rank}위"
    color  = TOP5_COLORS[top5_ranks[team]] if team in top5_teams else "#AAAAAA"
    ax.text(-0.03, i, marker, va="center", ha="right",
            fontsize=10, color=color, fontweight="bold")

ax.set_xlim(-0.04, 1.18)
ax.set_xlabel("포스트시즌 진출 확률", fontsize=11, color="#444444", labelpad=8)
ax.set_title("2026 KBO 포스트시즌 진출 확률 예측",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02, f"기준: {ref_date}  |  시즌 {ref_ratio:.1%} 경과  |  {SUBTITLE}",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRAY_AXIS)
ax.tick_params(axis="x", colors=GRAY_AXIS, labelsize=10)
ax.tick_params(axis="y", left=False, labelsize=11)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.8)

plt.tight_layout()
out1 = os.path.join(OUTDIR, "predict_2026_bar.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"바 차트 저장: {out1}")


# ─────────────────────────────────────────────
# 차트 2: 팀별 확률 추이
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
ax.set_facecolor(BG)

ax.axhspan(0.5, 1.05, alpha=0.04, color="#1B3F7A", zorder=0)
ax.axhline(0.5, color="#CC4444", lw=1.0, ls="--", alpha=0.6, zorder=1)
ax.text(0.5, 0.505, "포스트시즌 기준선 (50%)", color="#CC4444", fontsize=8.5, va="bottom", ha="left")

x_end = pred_df["games"].max()

for team in sorted(pred_df["team"].unique()):
    t     = pred_df[pred_df["team"] == team].sort_values("games")
    x     = t["games"].values
    y     = t["prob_norm"].values
    color = TEAM_COLORS[team]

    if team in top5_teams:
        ax.plot(x, y, color=color, lw=2.4, alpha=ALPHA_TOP, zorder=3)
        ax.text(x[-1] + 0.4, y[-1], team, color=color, fontsize=10,
                fontweight="bold", va="center", ha="left")
    else:
        ax.plot(x, y, color=color, lw=1.1, ls="--", alpha=ALPHA_BOTTOM, zorder=2)
        ax.text(x[-1] + 0.4, y[-1], team,
                color=mcolors.to_rgba(color, 0.5), fontsize=9, va="center", ha="left")

ax.set_xlim(0, x_end + 8)
ax.set_ylim(0, 1.05)
ax.set_xlabel("누적 경기 수", fontsize=11, color="#444444", labelpad=8)
ax.set_ylabel("포스트시즌 진출 확률", fontsize=11, color="#444444", labelpad=8)
ax.set_title("2026 KBO 포스트시즌 확률 추이",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02, f"기준: {ref_date}  |  진한 실선: 현재 상위 5팀",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=10)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out2 = os.path.join(OUTDIR, "predict_2026_trend.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"추이 차트 저장: {out2}")


# ─────────────────────────────────────────────
# 차트 3: 피처 중요도 Top 20
# ─────────────────────────────────────────────
imp_xgb  = pd.Series(xgb_f.feature_importances_,  index=TOP_FEATURES)
imp_lgbm = pd.Series(lgbm_f.feature_importances_, index=TOP_FEATURES)
imp_rf   = pd.Series(rf_f.feature_importances_,   index=TOP_FEATURES)

imp = (imp_xgb / imp_xgb.sum() + imp_lgbm / imp_lgbm.sum() + imp_rf / imp_rf.sum()) / 3
top_imp = imp.sort_values(ascending=False)

GROUP_COLOR = {"dyn_": "#0ea5e9", "prev_": "#2563A8", "other": "#888888"}
GROUP_LABEL = {"dyn_": "3년 평균 역가중 (dyn_)", "prev_": "전년도 기록 (prev_)", "other": "현재 시즌"}

def _feat_group(name):
    if name.startswith("dyn_"):  return "dyn_"
    if name.startswith("prev_"): return "prev_"
    return "other"

palette_imp = [GROUP_COLOR[_feat_group(f)] for f in top_imp.index]

fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG)
ax.set_facecolor(BG)

bars = ax.barh(range(len(top_imp)), top_imp.values[::-1],
               color=palette_imp[::-1], height=0.65, edgecolor="white", linewidth=0.4)
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels(top_imp.index[::-1], fontsize=10)

for bar, val in zip(bars, top_imp.values[::-1]):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="#444444")

legend_handles = [Patch(color=v, label=GROUP_LABEL[k]) for k, v in GROUP_COLOR.items()]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")

ax.set_xlabel("중요도 (XGB + LGBM + RF 평균 정규화)", fontsize=11, color="#444444", labelpad=8)
ax.set_title("피처 중요도 Top 20",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02, "XGBoost · LightGBM · RandomForest 중요도 평균 (LR 제외)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=10)
ax.tick_params(axis="y", left=False)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out3 = os.path.join(OUTDIR, "predict_2026_importance.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"피처 중요도 저장: {out3}")


# ─────────────────────────────────────────────
# 차트 4: 범프 차트 (날짜별 예측 순위 변화)
# ─────────────────────────────────────────────
rank_df = (
    pred_df
    .sort_values(["date", "prob_norm"], ascending=[True, False])
    .assign(pred_rank=lambda d: d.groupby("date").cumcount() + 1)
    [["date", "team", "pred_rank", "prob_norm"]]
)

fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
ax.set_facecolor(BG)

dates_ordered = sorted(rank_df["date"].unique())
date_to_x     = {d: i for i, d in enumerate(dates_ordered)}
n             = len(dates_ordered)

for team in sorted(rank_df["team"].unique()):
    t     = rank_df[rank_df["team"] == team].sort_values("date")
    xs    = [date_to_x[d] for d in t["date"]]
    ys    = t["pred_rank"].values
    color = TEAM_COLORS[team]

    if team in top5_teams:
        ax.plot(xs, ys, color=color, lw=2.6, alpha=ALPHA_TOP, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]],
                   color=color, s=60, zorder=4, edgecolors="white", linewidths=1.5)
        ax.text(xs[-1] + 0.3, ys[-1], team, color=color, fontsize=10,
                fontweight="bold", va="center", ha="left")
    else:
        ax.plot(xs, ys, color=color, lw=1.2, ls="--", alpha=ALPHA_BOTTOM, zorder=2)
        ax.text(xs[-1] + 0.3, ys[-1], team,
                color=mcolors.to_rgba(color, 0.5), fontsize=9, va="center", ha="left")

ax.axhline(5.5, color="#CC4444", lw=1.0, ls=":", alpha=0.7, zorder=1)
ax.text(0, 5.62, "포스트시즌 컷라인", color="#CC4444", fontsize=8.5, va="bottom")

tick_step   = max(1, n // 6)
tick_xs     = list(range(0, n, tick_step))
tick_labels = [pd.to_datetime(dates_ordered[i]).strftime("%m/%d") for i in tick_xs]
ax.set_xticks(tick_xs)
ax.set_xticklabels(tick_labels, fontsize=10)
ax.set_xlim(-0.5, n + 1.5)
ax.set_ylim(10.5, 0.5)
ax.set_yticks(range(1, 11))
ax.set_yticklabels([f"{i}위" for i in range(1, 11)], fontsize=10)
ax.set_xlabel("날짜", fontsize=11, color="#444444", labelpad=8)
ax.set_ylabel("예측 순위", fontsize=11, color="#444444", labelpad=8)
ax.set_title("2026 KBO 포스트시즌 예측 순위 변화",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02, f"기준: {ref_date}  |  진한 실선: 현재 상위 5팀  |  점: 시작·최신 시점",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=10)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out4 = os.path.join(OUTDIR, "predict_2026_bump.png")
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"범프 차트 저장: {out4}")


# ─────────────────────────────────────────────
# 차트 5: 현재 승률 vs 예측 확률 산점도
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG)
ax.set_facecolor(BG)

x_win        = latest["win_rate"].values
y_prob       = latest["prob_norm"].values
teams_latest = latest["team"].values

ax.axvline(0.5, color="#DDDDDD", lw=1.0, zorder=0)
ax.axhline(0.5, color="#DDDDDD", lw=1.0, zorder=0)

quad_kw = dict(fontsize=8.5, color="#BBBBBB", ha="center", va="center")
ax.text(0.35, 0.78, "현재 부진\n모델 낙관", **quad_kw)
ax.text(0.68, 0.78, "현재 강세\n모델 낙관", **quad_kw)
ax.text(0.35, 0.22, "현재 부진\n모델 비관", **quad_kw)
ax.text(0.68, 0.22, "현재 강세\n모델 비관", **quad_kw)

diag = np.linspace(0.2, 0.85, 100)
ax.plot(diag, diag, color="#CCCCCC", lw=1.0, ls="--", alpha=0.8, zorder=1, label="모델확률 = 현재승률")

for team, xv, yv in zip(teams_latest, x_win, y_prob):
    is_top5    = team in top5_teams
    base_color = TEAM_COLORS.get(team, "#888888")
    alpha_val  = ALPHA_TOP if is_top5 else ALPHA_BOTTOM
    color      = mcolors.to_rgba(base_color, alpha_val)
    ax.scatter(xv, yv, color=color, s=160 if is_top5 else 100, zorder=3,
               edgecolors="white", linewidths=1.2)
    ax.annotate(team, (xv, yv), xytext=(xv + 0.012, yv + 0.012),
                fontsize=10, fontweight="bold" if is_top5 else "normal",
                color=color, ha="left", va="bottom")

ax.set_xlim(0.25, 0.80)
ax.set_ylim(0.0, 1.08)
ax.set_xlabel("현재 승률 (실제 성적)", fontsize=12, color="#444444", labelpad=8)
ax.set_ylabel("모델 예측 확률 (정규화)", fontsize=12, color="#444444", labelpad=8)
ax.set_title("현재 승률 vs 모델 예측 확률",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02,
        f"기준: {ref_date}  |  대각선 위 = 모델 낙관  |  대각선 아래 = 모델 비관",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=10)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)
ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out5 = os.path.join(OUTDIR, "predict_2026_scatter.png")
plt.savefig(out5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"산점도 저장: {out5}")


# ─────────────────────────────────────────────
# 차트 6: 히트맵 (팀 × 날짜 확률 매트릭스)
# ─────────────────────────────────────────────
team_order  = list(latest.sort_values("prob_norm", ascending=False)["team"])
dates_list  = sorted(pred_df["date"].unique())
date_labels = [pd.to_datetime(d).strftime("%m/%d") for d in dates_list]

heatmap_data = np.zeros((len(team_order), len(dates_list)))
for i, team in enumerate(team_order):
    for j, date in enumerate(dates_list):
        val = pred_df[(pred_df["team"] == team) & (pred_df["date"] == date)]["prob_norm"]
        heatmap_data[i, j] = val.values[0] if len(val) else np.nan

fig, ax = plt.subplots(figsize=(15, 6), facecolor=BG)
ax.set_facecolor(BG)

im = ax.imshow(heatmap_data, aspect="auto", cmap="Blues",
               vmin=0.0, vmax=1.0, interpolation="nearest")

ax.axhline(4.5, color="#CC4444", lw=1.5, ls="--", alpha=0.8)
ax.text(len(dates_list) - 0.4, 4.35, "컷라인", color="#CC4444", fontsize=8.5, ha="right", va="bottom")

for i in range(len(team_order)):
    for j in range(len(dates_list)):
        val = heatmap_data[i, j]
        txt_color = "white" if val > 0.6 else "#333333"
        ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7.5, color=txt_color)

ax.set_yticks(range(len(team_order)))
ax.set_yticklabels([f"{'★' if t in top5_teams else '  '} {t}" for t in team_order], fontsize=10)
tick_step = max(1, len(dates_list) // 8)
ax.set_xticks(range(0, len(dates_list), tick_step))
ax.set_xticklabels(date_labels[::tick_step], fontsize=9, rotation=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("포스트시즌 진출 확률", fontsize=10, color="#444444")
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
cbar.ax.tick_params(labelsize=9, colors=GRAY_AXIS)

ax.set_title("2026 KBO 포스트시즌 확률 히트맵",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02, f"기준: {ref_date}  |  ★ 현재 예측 상위 5팀  |  진할수록 진출 확률 높음",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
out6 = os.path.join(OUTDIR, "predict_2026_heatmap.png")
plt.savefig(out6, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"히트맵 저장: {out6}")


# ─────────────────────────────────────────────
# 차트 7: 레이더 차트 (예측 상위 5팀 핵심 지표)
# ─────────────────────────────────────────────
RADAR_METRICS = {
    "종합전력\n(피타고라스)": "prev_pythagorean_win_rate",
    "득실차":                 "prev_run_differential",
    "투수력\n(ERA↓)":        "prev_team_era",
    "에이스\n(ERA↓)":        "prev_ace_era",
    "타격력\n(OPS)":         "prev_top5_hitter_ops_avg",
    "장타력\n(ISO)":         "prev_iso",
}
LOWER_IS_BETTER = {"prev_team_era", "prev_ace_era"}

prev_2026 = pd.read_csv(
    os.path.join(ROOT, "data/processed/2026/prev_features_from_2025.csv")
)
radar_df = prev_2026[["team"] + list(RADAR_METRICS.values())].copy()

for col in RADAR_METRICS.values():
    mn, mx = radar_df[col].min(), radar_df[col].max()
    radar_df[col] = (radar_df[col] - mn) / (mx - mn)
    if col in LOWER_IS_BETTER:
        radar_df[col] = 1 - radar_df[col]

labels   = list(RADAR_METRICS.keys())
n_labels = len(labels)
angles   = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
angles  += angles[:1]

radar_teams = list(latest.head(5)["team"])

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True}, facecolor=BG)
ax.set_facecolor(BG)

for team in radar_teams:
    row    = radar_df[radar_df["team"] == team]
    values = [row[col].values[0] for col in RADAR_METRICS.values()]
    values += values[:1]
    color  = TEAM_COLORS.get(team, "#888888")

    ax.plot(angles, values, color=color, lw=2.2, zorder=3)
    ax.fill(angles, values, color=color, alpha=0.10)
    peak_idx = int(np.argmax(values[:-1]))
    ax.scatter(angles[peak_idx], values[peak_idx],
               color=color, s=60, zorder=4, edgecolors="white", linewidths=1.2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10.5, color="#333333")
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color=GRAY_AXIS)
ax.set_ylim(0, 1)
ax.spines["polar"].set_color("#DDDDDD")
ax.grid(color="#E0E0E0", linewidth=0.8)

legend_handles = [
    Line2D([0], [0], color=TEAM_COLORS.get(team, "#888888"), lw=2.5, label=team)
    for team in radar_teams
]
ax.legend(handles=legend_handles, loc="upper right",
          bbox_to_anchor=(1.28, 1.12), fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")

ax.set_title("예측 상위 5팀 핵심 지표 프로파일\n(전년도 기록 기준, 높을수록 유리)",
             fontsize=13, fontweight="bold", color="#1B1B1B", pad=32, y=1.06)
ax.text(0.5, -0.06, "기준: 2025 시즌 기록  |  ERA 계열은 낮을수록 좋아 반전 정규화 적용",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")

plt.tight_layout()
out7 = os.path.join(OUTDIR, "predict_2026_radar.png")
plt.savefig(out7, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"레이더 차트 저장: {out7}")


print("\n완료.")
print(f"  검증 차트 (5종): {VALDIR}/")
print(f"  예측 차트 (7종): {OUTDIR}/")
