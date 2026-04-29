"""
09_feature_reduced_eval.py

피처를 36개 → 상위 20개로 줄였을 때 과적합 갭이 개선되는지 검증.
  1) 전체 데이터로 피처 중요도 계산 → 상위 20개 선정
  2) LOSO-CV (2018~2025) 재실행
  3) 36개 기준선과 Test AUC / 과적합 갭 비교

기준선 (36개, 08_ensemble_model_eval.py 결과):
  Test AUC 평균 0.7958 / Train AUC 1.0000 / 과적합 갭 0.1933 / Brier 0.1965

시각화 저장: notebooks/experiments/jh/Validation/assets/feat20_*.png
실행: uv run python "notebooks/experiments/jh/Validation/09_feature_reduced_eval.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import brier_score_loss
import lightgbm as _lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import FEATURE_COLS
from src.evaluation.metrics import evaluate_binary_model, print_metrics

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE   = os.path.join(os.path.dirname(__file__), "../../../..")
ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

BG        = "#F8F9FA"
GRAY_AXIS = "#AAAAAA"
ACCENT    = "#1B3F7A"
WARM_RED  = "#CC4444"
GREEN     = "#2E8B57"

# 36개 기준선 (08_ensemble_model_eval.py 결과)
BASELINE = {
    "label":          "36개 피처 (기준)",
    "test_auc_mean":  0.7958,
    "train_auc_mean": 1.0000,
    "gap":            0.1933,
    "brier":          0.1965,
    "season_auc": {
        2018: 0.9399, 2019: 0.6678, 2020: 0.9619,
        2021: 0.6016, 2022: 0.8762, 2023: 0.8019,
        2024: 0.9168, 2025: 0.6876,
    },
}


def _build_models(pos_w):
    xgb = XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
        scale_pos_weight=pos_w, eval_metric="logloss",
        early_stopping_rounds=20, random_state=42,
    )
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, min_child_samples=20,
        scale_pos_weight=pos_w, random_state=42, verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=20,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    return xgb, lgbm, rf


def _fit(xgb, lgbm, rf, X_tr, y_tr, sw, X_es, y_es):
    xgb.fit(X_tr, y_tr, sample_weight=sw,
            eval_set=[(X_tr, y_tr), (X_es, y_es)],
            verbose=False)
    lgbm.fit(X_tr, y_tr, sample_weight=sw,
             eval_set=[(X_tr, y_tr), (X_es, y_es)],
             eval_names=["train", "valid"],
             eval_metric="binary_logloss",
             callbacks=[_lgb.early_stopping(20, verbose=False)])
    rf.fit(X_tr, y_tr, sample_weight=sw)


def _proba(xgb, lgbm, rf, X):
    return (
        xgb.predict_proba(X)[:, 1] +
        lgbm.predict_proba(X)[:, 1] +
        rf.predict_proba(X)[:, 1]
    ) / 3


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

print(f"전체 데이터: {df.shape}  |  피처: {len(FEATURE_COLS)}개\n")


# ──────────────────────────────────────────────
# STEP 1: 전체 데이터로 피처 중요도 계산 → 상위 20개 선정
# ──────────────────────────────────────────────
print("=" * 55)
print("STEP 1 — 피처 중요도 계산 (전체 데이터)")
print("=" * 55)

X_all = df[FEATURE_COLS]
y_all = df["postseason"]

s_min, s_max = df["season"].min(), df["season"].max()
sw_all  = (0.3 + 0.7 * (df["season"] - s_min) / (s_max - s_min)).values
pos_w_all = (y_all == 0).sum() / max((y_all == 1).sum(), 1)

xgb_imp, lgbm_imp, rf_imp = _build_models(pos_w_all)

# 중요도 계산용은 Early Stopping 불필요 → 그냥 fit
xgb_imp_m = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
    scale_pos_weight=pos_w_all, eval_metric="logloss", random_state=42,
)
lgbm_imp_m = LGBMClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=5.0, min_child_samples=20,
    scale_pos_weight=pos_w_all, random_state=42, verbose=-1,
)
rf_imp_m = RandomForestClassifier(
    n_estimators=300, max_depth=4, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1,
)
xgb_imp_m.fit(X_all, y_all, sample_weight=sw_all)
lgbm_imp_m.fit(X_all, y_all, sample_weight=sw_all)
rf_imp_m.fit(X_all, y_all, sample_weight=sw_all)

imp_xgb  = pd.Series(xgb_imp_m.feature_importances_,  index=FEATURE_COLS)
imp_lgbm = pd.Series(lgbm_imp_m.feature_importances_, index=FEATURE_COLS)
imp_rf   = pd.Series(rf_imp_m.feature_importances_,   index=FEATURE_COLS)

imp = (imp_xgb / imp_xgb.sum() + imp_lgbm / imp_lgbm.sum() + imp_rf / imp_rf.sum()) / 3
imp_sorted = imp.sort_values(ascending=False)

TOP_N = 20
TOP_FEATURES = imp_sorted.head(TOP_N).index.tolist()

print(f"\n상위 {TOP_N}개 피처 (중요도 내림차순):")
for i, (feat, val) in enumerate(imp_sorted.head(TOP_N).items(), 1):
    print(f"  {i:>2}. {feat:<38} {val:.4f}")

print(f"\n제거된 피처 ({len(FEATURE_COLS) - TOP_N}개):")
dropped = imp_sorted.tail(len(FEATURE_COLS) - TOP_N).index.tolist()
for feat in dropped:
    print(f"  - {feat:<38} {imp[feat]:.4f}")


# ──────────────────────────────────────────────
# STEP 2: 상위 20개 피처로 LOSO-CV
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"STEP 2 — LOSO-CV ({TOP_N}개 피처, 2018~2025)")
print("=" * 55)

TEST_SEASONS = list(range(2018, 2026))
fold_metrics = []

for test_season in TEST_SEASONS:
    train = df[df["season"] != test_season].copy()
    test  = df[df["season"] == test_season].copy()

    X_tr, y_tr = train[TOP_FEATURES], train["postseason"]
    X_te, y_te = test[TOP_FEATURES],  test["postseason"]

    s_min_f = train["season"].min()
    s_max_f = train["season"].max()
    sw = (0.3 + 0.7 * (train["season"] - s_min_f) / max(s_max_f - s_min_f, 1)).values
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    _last = train["season"].max()
    X_es  = train.loc[train["season"] == _last, TOP_FEATURES]
    y_es  = train.loc[train["season"] == _last, "postseason"]

    xgb, lgbm, rf = _build_models(pos_w)
    _fit(xgb, lgbm, rf, X_tr, y_tr, sw, X_es, y_es)

    prob_te = _proba(xgb, lgbm, rf, X_te)
    prob_tr = _proba(xgb, lgbm, rf, X_tr)

    m_te = evaluate_binary_model(y_te, prob_te)
    m_tr = evaluate_binary_model(y_tr, prob_tr)

    fold_metrics.append({
        "season":    test_season,
        "test_auc":  m_te["roc_auc"],
        "train_auc": m_tr["roc_auc"],
        "brier":     m_te["brier"],
    })
    print_metrics(m_te, label=str(test_season))

fold_df = pd.DataFrame(fold_metrics)
gap_new = (fold_df["train_auc"] - fold_df["test_auc"]).mean()

print(f"\n{'─'*55}")
print(f"  Test  AUC 평균 : {fold_df['test_auc'].mean():.4f}  "
      f"(기준 {BASELINE['test_auc_mean']:.4f}  "
      f"{'↑' if fold_df['test_auc'].mean() > BASELINE['test_auc_mean'] else '↓'}"
      f"{abs(fold_df['test_auc'].mean() - BASELINE['test_auc_mean']):.4f})")
print(f"  Train AUC 평균 : {fold_df['train_auc'].mean():.4f}  "
      f"(기준 {BASELINE['train_auc_mean']:.4f})")
print(f"  과적합 갭      : {gap_new:.4f}  "
      f"(기준 {BASELINE['gap']:.4f}  "
      f"{'↓ 개선' if gap_new < BASELINE['gap'] else '↑ 악화'} "
      f"{abs(gap_new - BASELINE['gap']):.4f})")
print(f"  Brier Score    : {fold_df['brier'].mean():.4f}  "
      f"(기준 {BASELINE['brier']:.4f})")


# ──────────────────────────────────────────────
# 차트 1: 피처 중요도 (36개 전체)
# ──────────────────────────────────────────────
GROUP_COLOR = {"dyn_": "#2E8B57", "prev_": "#2563A8", "other": "#888888"}

def _group(name):
    if name.startswith("dyn_"):  return "dyn_"
    if name.startswith("prev_"): return "prev_"
    return "other"

colors_all = [GROUP_COLOR[_group(f)] for f in imp_sorted.index]
is_selected = [f in TOP_FEATURES for f in imp_sorted.index]

fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
ax.set_facecolor(BG)

bars = ax.barh(range(len(imp_sorted)), imp_sorted.values[::-1],
               color=[GROUP_COLOR[_group(f)] for f in imp_sorted.index[::-1]],
               height=0.65, edgecolor="white", linewidth=0.4)

feats_rev = list(imp_sorted.index[::-1])
for bar, feat in zip(bars, feats_rev):
    bar.set_alpha(1.0 if feat in TOP_FEATURES else 0.30)

ax.set_yticks(range(len(imp_sorted)))
ax.set_yticklabels(imp_sorted.index[::-1], fontsize=9)

# 컷라인 (20위/21위 경계)
ax.axhline(len(imp_sorted) - TOP_N - 0.5,
           color=WARM_RED, linewidth=1.3, linestyle="--", alpha=0.8)
ax.text(imp_sorted.values[TOP_N - 1] * 0.5,
        len(imp_sorted) - TOP_N - 0.35,
        f"상위 {TOP_N}개 선택 기준선",
        color=WARM_RED, fontsize=8.5, va="bottom")

for bar, val, feat in zip(bars, imp_sorted.values[::-1], imp_sorted.index[::-1]):
    alpha = 1.0 if feat in TOP_FEATURES else 0.5
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8, color="#333333", alpha=alpha)

from matplotlib.patches import Patch
legend_handles = [
    Patch(color=GROUP_COLOR["dyn_"],   label="dyn_  : 3년 평균 역가중"),
    Patch(color=GROUP_COLOR["prev_"],  label="prev_ : 전년도 기록"),
    Patch(color=GROUP_COLOR["other"],  label="현재 시즌"),
    Patch(color="#BBBBBB", alpha=0.35, label="제거 피처 (하위 16개)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
          framealpha=0.9, edgecolor="#DDDDDD")

ax.set_xlabel("중요도 (XGB + LGBM + RF 평균 정규화)", fontsize=11, color="#444444")
ax.set_title(f"피처 중요도 전체 — 상위 {TOP_N}개 선택",
             fontsize=14, fontweight="bold", color="#1B1B1B", pad=20)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=9)
ax.tick_params(axis="y", left=False)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out1 = os.path.join(ASSETS, "feat20_importance.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"\n피처 중요도 저장: {out1}")


# ──────────────────────────────────────────────
# 차트 2: 36개 vs 20개 Test AUC 시즌별 비교
# ──────────────────────────────────────────────
x     = np.arange(len(TEST_SEASONS))
width = 0.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

# ── 왼쪽: Test AUC 시즌별 비교 ──
auc_36 = [BASELINE["season_auc"][s] for s in TEST_SEASONS]
auc_20 = fold_df["test_auc"].tolist()

bars_36 = ax1.bar(x - width / 2, auc_36, width,
                  color=GRAY_AXIS, alpha=0.7, label="36개 피처 (기준)", edgecolor="white")
bars_20 = ax1.bar(x + width / 2, auc_20, width,
                  color=ACCENT, alpha=0.85, label=f"상위 {TOP_N}개 피처", edgecolor="white")

for bar, v36, v20 in zip(bars_20, auc_36, auc_20):
    diff = v20 - v36
    color = GREEN if diff >= 0 else WARM_RED
    ax1.text(bar.get_x() + bar.get_width() / 2,
             max(v36, v20) + 0.015,
             f"{'+' if diff >= 0 else ''}{diff:.3f}",
             ha="center", fontsize=8, color=color, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=10)
ax1.set_ylim(0.4, 1.12)
ax1.set_xlabel("테스트 시즌", fontsize=11, color="#444444")
ax1.set_ylabel("Test ROC-AUC", fontsize=11, color="#444444")
ax1.set_title("Test AUC 시즌별 비교", fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax1.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax1.spines[spine].set_color(GRAY_AXIS)
ax1.tick_params(colors=GRAY_AXIS, labelsize=10)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

# ── 오른쪽: 핵심 지표 요약 비교 ──
metrics_labels = ["Test AUC 평균", "Train AUC 평균", "과적합 갭", "Brier Score"]
vals_36 = [
    BASELINE["test_auc_mean"],
    BASELINE["train_auc_mean"],
    BASELINE["gap"],
    BASELINE["brier"],
]
vals_20 = [
    fold_df["test_auc"].mean(),
    fold_df["train_auc"].mean(),
    gap_new,
    fold_df["brier"].mean(),
]
# 과적합 갭 / Brier는 낮을수록 좋음 (색상 반전)
lower_is_better = [False, False, True, True]

x2 = np.arange(len(metrics_labels))
w2 = 0.38
b36 = ax2.bar(x2 - w2 / 2, vals_36, w2, color=GRAY_AXIS, alpha=0.7,
              label="36개 피처", edgecolor="white")
b20 = ax2.bar(x2 + w2 / 2, vals_20, w2, color=ACCENT, alpha=0.85,
              label=f"상위 {TOP_N}개 피처", edgecolor="white")

for j, (v36, v20, lower) in enumerate(zip(vals_36, vals_20, lower_is_better)):
    diff  = v20 - v36
    better = diff < 0 if lower else diff > 0
    color = GREEN if better else WARM_RED
    ax2.text(j + w2 / 2, max(v36, v20) + 0.008,
             f"{'+' if diff >= 0 else ''}{diff:.4f}",
             ha="center", fontsize=8, color=color, fontweight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(metrics_labels, fontsize=9.5)
ax2.set_ylim(0, 1.15)
ax2.set_title("핵심 지표 요약 비교", fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax2.text(0.5, 1.01,
         "과적합 갭 · Brier: 낮을수록 좋음 (초록=개선, 빨강=악화)",
         transform=ax2.transAxes, ha="center", fontsize=8.5, color="#777777")
ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color(GRAY_AXIS)
ax2.tick_params(colors=GRAY_AXIS, labelsize=10)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle(
    f"36개 피처 vs 상위 {TOP_N}개 피처 성능 비교",
    fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02,
)
plt.tight_layout()
out2 = os.path.join(ASSETS, "feat20_comparison.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"비교 차트 저장: {out2}")

print(f"\n완료 (1~2). 저장 위치: {ASSETS}")


# ──────────────────────────────────────────────
# 차트 3: 학습 손실 곡선 (20개 피처, 2025 폴드)
# ──────────────────────────────────────────────
print("\n학습 손실 곡선 계산 중 (20개 피처 / 2025 폴드)...")

train_lc = df[df["season"] != 2025].copy()
test_lc  = df[df["season"] == 2025].copy()

X_tr_lc  = train_lc[TOP_FEATURES]
y_tr_lc  = train_lc["postseason"]
X_te_lc  = test_lc[TOP_FEATURES]
y_te_lc  = test_lc["postseason"]

s_min_lc = train_lc["season"].min()
s_max_lc = train_lc["season"].max()
sw_lc    = (0.3 + 0.7 * (train_lc["season"] - s_min_lc) / (s_max_lc - s_min_lc)).values
pos_w_lc = (y_tr_lc == 0).sum() / max((y_tr_lc == 1).sum(), 1)

# XGBoost
xgb_lc = XGBClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
    scale_pos_weight=pos_w_lc, eval_metric="logloss", random_state=42,
)
xgb_lc.fit(X_tr_lc, y_tr_lc, sample_weight=sw_lc,
           eval_set=[(X_tr_lc, y_tr_lc), (X_te_lc, y_te_lc)],
           verbose=False)
xgb_res      = xgb_lc.evals_result()
xgb_tr_loss  = xgb_res["validation_0"]["logloss"]
xgb_val_loss = xgb_res["validation_1"]["logloss"]
print(f"  XGBoost 완료  ({len(xgb_tr_loss)} rounds)")

# LightGBM
lgbm_eval_res = {}
lgbm_lc = LGBMClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=5.0, min_child_samples=20,
    scale_pos_weight=pos_w_lc, random_state=42, verbose=-1,
)
lgbm_lc.fit(X_tr_lc, y_tr_lc, sample_weight=sw_lc,
            eval_set=[(X_tr_lc, y_tr_lc), (X_te_lc, y_te_lc)],
            eval_names=["train", "valid"],
            eval_metric="binary_logloss",
            callbacks=[_lgb.record_evaluation(lgbm_eval_res)])
lgbm_tr_loss  = lgbm_eval_res["train"]["binary_logloss"]
lgbm_val_loss = lgbm_eval_res["valid"]["binary_logloss"]
print(f"  LightGBM 완료  ({len(lgbm_tr_loss)} rounds)")

# 36개 피처 기준 손실값 (08_ensemble_model_eval.py 동일 폴드 참고용)
BASELINE_LOSS = {
    "xgb_val_final":  None,   # 비교용 수치 없음 — 차트 타이틀에 표기
    "lgbm_val_final": None,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

for ax, tr_loss, val_loss, title in [
    (ax1, xgb_tr_loss,  xgb_val_loss,  "XGBoost"),
    (ax2, lgbm_tr_loss, lgbm_val_loss, "LightGBM"),
]:
    rounds = np.arange(1, len(tr_loss) + 1)

    ax.plot(rounds, tr_loss,  color=ACCENT,   linewidth=2.2, label="Train Loss",      zorder=3)
    ax.plot(rounds, val_loss, color=WARM_RED,  linewidth=2.2, label="Validation Loss", zorder=3)

    # Train/Val 갭 음영
    ax.fill_between(rounds, tr_loss, val_loss,
                    where=[v > t for t, v in zip(tr_loss, val_loss)],
                    alpha=0.08, color=WARM_RED, label="과적합 구간")

    # Val Loss 최저점
    min_idx = int(np.argmin(val_loss))
    ax.scatter(min_idx + 1, val_loss[min_idx], color=WARM_RED, s=70, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.text(min_idx + max(len(rounds) * 0.03, 3), val_loss[min_idx],
            f"Val 최저\n{val_loss[min_idx]:.4f}\n({min_idx + 1}R)",
            fontsize=8, color=WARM_RED, va="center")

    # 최종 수치 주석
    ax.annotate(f"Train {tr_loss[-1]:.4f}", xy=(rounds[-1], tr_loss[-1]),
                xytext=(rounds[-1] * 0.82, tr_loss[-1] - 0.015),
                fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.8))
    ax.annotate(f"Val {val_loss[-1]:.4f}", xy=(rounds[-1], val_loss[-1]),
                xytext=(rounds[-1] * 0.82, val_loss[-1] + 0.015),
                fontsize=8, color=WARM_RED,
                arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.8))

    final_gap = val_loss[-1] - tr_loss[-1]
    ax.set_xlabel("부스팅 라운드", fontsize=11, color="#444444")
    ax.set_ylabel("Log Loss", fontsize=11, color="#444444")
    ax.set_title(f"{title} 학습 손실 곡선  (20개 피처)",
                 fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
    ax.text(0.5, 1.01,
            f"최종 갭(Val-Train): {final_gap:.4f}  |  Val 최저점: {min_idx + 1}R",
            transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    ax.set_xlim(0, len(rounds) + len(rounds) * 0.12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(GRAY_AXIS)
    ax.tick_params(colors=GRAY_AXIS, labelsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle(
    f"학습 손실 곡선  —  상위 {TOP_N}개 피처  /  Train 2017~2024  ·  Val 2025",
    fontsize=13, fontweight="bold", color="#1B1B1B", y=1.02,
)
plt.tight_layout()
out3 = os.path.join(ASSETS, "feat20_loss_curves.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"학습 손실 곡선 저장: {out3}")

print(f"\n완료. 저장 위치: {ASSETS}")
print(f"  {os.path.basename(out1)}  — 피처 중요도 (제거 피처 표시)")
print(f"  {os.path.basename(out2)}  — 36개 vs {TOP_N}개 비교")
print(f"  {os.path.basename(out3)}  — 학습 손실 곡선 (20개 피처)")
