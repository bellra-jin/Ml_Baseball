"""
08_ensemble_model_eval.py

앙상블 모델(XGBoost + LightGBM + RandomForest)의 설명력 및 일반화 성능 평가.
Leave-One-Season-Out CV(2018~2025)로 과적합 여부·ROC-AUC·F1·Brier Score·보정 곡선을 검증.

시각화 저장: notebooks/experiments/jh/Validation/assets/
  - eval_cv_scorecard.png  — 시즌별 전 지표 성적표 (히트맵 테이블)
  - eval_roc_curves.png    — 폴드별 ROC 커브 + 평균
  - eval_overfitting.png   — Train vs Test AUC (과적합 갭)
  - eval_calibration.png   — 보정 곡선 + 예측 확률 분포

실행: uv run python "notebooks/experiments/jh/Validation/08_ensemble_model_eval.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, auc as sk_auc, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as _lgb
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


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"[경고] 없는 피처: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

print(f"전체 데이터: {df.shape}  시즌: {sorted(df['season'].unique())}")
print(f"피처 수: {len(FEATURE_COLS)}개\n")


# ──────────────────────────────────────────────
# Leave-One-Season-Out Cross-Validation
# ──────────────────────────────────────────────
TEST_SEASONS = list(range(2018, 2026))

fold_metrics   = []
fold_curves    = []
fold_team_preds = []
all_y_true     = []
all_y_proba    = []

print("=" * 62)
print("Leave-One-Season-Out Cross-Validation (2018~2025)")
print("=" * 62)

for test_season in TEST_SEASONS:
    train = df[df["season"] != test_season].copy()
    test  = df[df["season"] == test_season].copy()

    X_train, y_train = train[FEATURE_COLS], train["postseason"]
    X_test,  y_test  = test[FEATURE_COLS],  test["postseason"]

    s_min, s_max = train["season"].min(), train["season"].max()
    denom = max(s_max - s_min, 1)
    sw    = (0.3 + 0.7 * (train["season"] - s_min) / denom).values
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # Early Stopping 검증셋: 학습 데이터 중 가장 최신 시즌
    _last = train["season"].max()
    X_es  = train.loc[train["season"] == _last, FEATURE_COLS]
    y_es  = train.loc[train["season"] == _last, "postseason"]

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

    xgb.fit(X_train, y_train, sample_weight=sw,
            eval_set=[(X_train, y_train), (X_es, y_es)],
            verbose=False)
    lgbm.fit(X_train, y_train, sample_weight=sw,
             eval_set=[(X_train, y_train), (X_es, y_es)],
             eval_names=["train", "valid"],
             eval_metric="binary_logloss",
             callbacks=[_lgb.early_stopping(20, verbose=False)])
    rf.fit(X_train, y_train, sample_weight=sw)

    def _proba(X, xgb=xgb, lgbm=lgbm, rf=rf):
        return (
            xgb.predict_proba(X)[:, 1] +
            lgbm.predict_proba(X)[:, 1] +
            rf.predict_proba(X)[:, 1]
        ) / 3

    prob_test  = _proba(X_test)
    prob_train = _proba(X_train)

    m_test  = evaluate_binary_model(y_test,  prob_test)
    m_train = evaluate_binary_model(y_train, prob_train)

    fpr, tpr, _ = roc_curve(y_test, prob_test)

    fold_metrics.append({
        "season":    test_season,
        "test_auc":  m_test["roc_auc"],
        "train_auc": m_train["roc_auc"],
        "test_f1":   m_test["f1"],
        "test_acc":  m_test["accuracy"],
        "test_prec": m_test["precision"],
        "test_rec":  m_test["recall"],
        "brier":     m_test["brier"],
    })
    fold_curves.append({"season": test_season, "fpr": fpr, "tpr": tpr})

    all_y_true.extend(y_test.values)
    all_y_proba.extend(prob_test)

    # 팀별 최종 시점 예측 수집 (오분류 분석용)
    test_tagged = test.copy()
    test_tagged["prob_cv"] = prob_test
    team_final = (
        test_tagged.sort_values("date").groupby("team").last().reset_index()
    )
    top5_pred = set(team_final.nlargest(5, "prob_cv")["team"])
    team_final["pred_top5"] = team_final["team"].isin(top5_pred).astype(int)
    team_final["season"] = test_season
    fold_team_preds.append(
        team_final[["team", "season", "postseason", "prob_cv", "pred_top5"]]
    )

    print_metrics(m_test, label=str(test_season))

fold_df     = pd.DataFrame(fold_metrics)
all_y_true  = np.array(all_y_true)
all_y_proba = np.array(all_y_proba)

# 전체 집계
print("\n" + "=" * 62)
overall     = evaluate_binary_model(all_y_true, all_y_proba)
brier_all   = overall["brier"]
auc_gap_avg = (fold_df["train_auc"] - fold_df["test_auc"]).mean()
print_metrics(overall, label="전체 CV 집계")
print(f"  Brier Score    = {brier_all:.4f}  (낮을수록 좋음 | 완벽=0 | 랜덤≈0.25)")
print(f"  Train AUC 평균 = {fold_df['train_auc'].mean():.4f}")
print(f"  Test  AUC 평균 = {fold_df['test_auc'].mean():.4f}")
print(f"  과적합 갭      = {auc_gap_avg:.4f}  "
      f"{'[!] 과적합 의심' if auc_gap_avg > 0.08 else '[OK] 정상 범위'}")
print()


# ──────────────────────────────────────────────
# 차트 1: CV 성적표 (히트맵 테이블)
# ──────────────────────────────────────────────
seasons_str = [str(s) for s in fold_df["season"]]
row_labels  = ["ROC-AUC", "F1 Score", "Precision", "Recall", "Accuracy", "Brier↓"]
table_vals  = np.array([
    fold_df["test_auc"].values,
    fold_df["test_f1"].values,
    fold_df["test_prec"].values,
    fold_df["test_rec"].values,
    fold_df["test_acc"].values,
    fold_df["brier"].values,
])

# 행별 min-max 정규화 (색상용). Brier는 낮을수록 좋으므로 반전.
color_vals = np.zeros_like(table_vals)
for i in range(len(row_labels)):
    row = table_vals[i]
    mn, mx = row.min(), row.max()
    normed = (row - mn) / (mx - mn + 1e-9)
    color_vals[i] = (1 - normed) if i == 5 else normed

mean_col   = table_vals.mean(axis=1, keepdims=True)
mean_col_c = color_vals.mean(axis=1, keepdims=True)
table_full = np.hstack([table_vals, mean_col])
color_full = np.hstack([color_vals, mean_col_c])
col_labels = seasons_str + ["평균"]

fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
ax.set_facecolor(BG)

ax.imshow(color_full, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        val = table_full[i, j]
        c   = color_full[i, j]
        txt_color = "white" if c < 0.25 else "#333333"
        fw  = "bold" if j == len(col_labels) - 1 else "normal"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=10, color=txt_color, fontweight=fw)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=11)
ax.axvline(len(seasons_str) - 0.5, color="white", linewidth=2.5)
ax.set_title(
    "앙상블 모델 CV 성적표  (Leave-One-Season-Out, 2018~2025)\n"
    "초록: 우수  ·  빨강: 미흡  ·  Brier↓ 는 낮을수록 좋음",
    fontsize=13, fontweight="bold", color="#1B1B1B", pad=12,
)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
out1 = os.path.join(ASSETS, "eval_cv_scorecard.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"성적표 저장: {out1}")


# ──────────────────────────────────────────────
# 차트 2: ROC 커브 (폴드별 + 평균)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
ax.set_facecolor(BG)

mean_fpr    = np.linspace(0, 1, 100)
tprs_interp = []
n           = len(fold_curves)
cmap        = plt.cm.Blues

for i, fc in enumerate(fold_curves):
    interp_tpr      = np.interp(mean_fpr, fc["fpr"], fc["tpr"])
    interp_tpr[0]   = 0.0
    tprs_interp.append(interp_tpr)
    season_auc = fold_df.loc[fold_df["season"] == fc["season"], "test_auc"].values[0]
    ax.plot(fc["fpr"], fc["tpr"],
            color=cmap(0.35 + 0.55 * i / max(n - 1, 1)),
            alpha=0.65, linewidth=1.2,
            label=f"{fc['season']} (AUC={season_auc:.3f})")

mean_tpr     = np.mean(tprs_interp, axis=0)
mean_tpr[-1] = 1.0
mean_auc_val = sk_auc(mean_fpr, mean_tpr)
std_tpr      = np.std(tprs_interp, axis=0)

ax.plot(mean_fpr, mean_tpr, color=ACCENT, linewidth=2.8, zorder=5,
        label=f"평균 ROC (AUC={mean_auc_val:.3f})")
ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                color=ACCENT, alpha=0.12, label="±1σ")
ax.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=1, linestyle="--", label="랜덤 분류기")

ax.set_xlabel("False Positive Rate (1 - 특이도)", fontsize=11, color="#444444")
ax.set_ylabel("True Positive Rate (민감도)", fontsize=11, color="#444444")
ax.set_title("ROC 커브 (시즌별 + 평균)", fontsize=14, fontweight="bold", color="#1B1B1B", pad=20)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor="#DDDDDD")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=10)
ax.set_axisbelow(True)
ax.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out2 = os.path.join(ASSETS, "eval_roc_curves.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"ROC 커브 저장: {out2}")


# ──────────────────────────────────────────────
# 차트 3: 과적합 체크 (Train vs Test AUC)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(BG)

x     = np.arange(len(fold_df))
width = 0.38

bars_train = ax.bar(x - width / 2, fold_df["train_auc"], width,
                    color=ACCENT, alpha=0.85, label="Train AUC", edgecolor="white")
bars_test  = ax.bar(x + width / 2, fold_df["test_auc"],  width,
                    color="#6BADD6", alpha=0.85, label="Test AUC", edgecolor="white")

for bar, val in zip(bars_train, fold_df["train_auc"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#333333")
for bar, val in zip(bars_test, fold_df["test_auc"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#333333")

for i, (tr, te) in enumerate(zip(fold_df["train_auc"], fold_df["test_auc"])):
    gap   = tr - te
    color = WARM_RED if gap > 0.10 else "#888888"
    ax.annotate(
        f"Δ{gap:.3f}",
        xy=(i, max(tr, te) + 0.025),
        ha="center", fontsize=8.5, color=color,
        fontweight="bold" if gap > 0.10 else "normal",
    )

ax.set_xticks(x)
ax.set_xticklabels(fold_df["season"].astype(str), fontsize=10)
ax.set_ylim(0.5, 1.15)
ax.set_xlabel("테스트 시즌", fontsize=11, color="#444444")
ax.set_ylabel("ROC-AUC", fontsize=11, color="#444444")
ax.set_title("과적합 체크: Train AUC vs Test AUC",
             fontsize=14, fontweight="bold", color="#1B1B1B", pad=20)
ax.text(
    0.5, 1.02,
    f"Train 평균 {fold_df['train_auc'].mean():.3f}  ·  "
    f"Test 평균 {fold_df['test_auc'].mean():.3f}  ·  "
    f"갭 {auc_gap_avg:.3f}  "
    f"{'[!] 과적합 의심' if auc_gap_avg > 0.08 else '[OK] 정상 범위'}",
    transform=ax.transAxes, ha="center", fontsize=9, color="#555555",
)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=10)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out3 = os.path.join(ASSETS, "eval_overfitting.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"과적합 체크 저장: {out3}")


# ──────────────────────────────────────────────
# 차트 4: 보정 곡선 + 예측 확률 분포
# ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

fop, mpv = calibration_curve(all_y_true, all_y_proba, n_bins=10, strategy="uniform")

ax1.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=1.2, linestyle="--", label="완벽한 보정")
ax1.plot(mpv, fop, color=ACCENT, linewidth=2.5, marker="o", markersize=7, label="앙상블 모델")
ax1.fill_between(mpv, mpv, fop, alpha=0.12, color=ACCENT)

ax1.set_xlabel("예측 확률 (모델 출력)", fontsize=11, color="#444444")
ax1.set_ylabel("실제 양성 비율 (포스트시즌 진출)", fontsize=11, color="#444444")
ax1.set_title("보정 곡선 (Reliability Diagram)",
              fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax1.text(0.5, 1.01,
         f"Brier Score={brier_all:.4f}  (완벽=0 | 랜덤≈0.25 | 낮을수록 좋음)",
         transform=ax1.transAxes, ha="center", fontsize=9, color="#777777")
ax1.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.05)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax1.spines[spine].set_color(GRAY_AXIS)
ax1.tick_params(colors=GRAY_AXIS, labelsize=10)
ax1.set_axisbelow(True)
ax1.grid(True, color="#E8E8E8", linewidth=0.8)

pos_proba = all_y_proba[all_y_true == 1]
neg_proba = all_y_proba[all_y_true == 0]

ax2.hist(neg_proba, bins=20, range=(0, 1), alpha=0.65,
         color="#D5D5D5", label=f"비포스트시즌 (n={len(neg_proba)})", edgecolor="white")
ax2.hist(pos_proba, bins=20, range=(0, 1), alpha=0.80,
         color=ACCENT, label=f"포스트시즌 진출 (n={len(pos_proba)})", edgecolor="white")
ax2.axvline(0.5, color=WARM_RED, linewidth=1.2, linestyle="--", alpha=0.8, label="0.5 기준선")

ax2.set_xlabel("예측 확률", fontsize=11, color="#444444")
ax2.set_ylabel("샘플 수", fontsize=11, color="#444444")
ax2.set_title("예측 확률 분포",
              fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax2.text(0.5, 1.01,
         "두 분포가 잘 분리될수록 모델 분별력(Discrimination) 우수",
         transform=ax2.transAxes, ha="center", fontsize=9, color="#777777")
ax2.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color(GRAY_AXIS)
ax2.tick_params(colors=GRAY_AXIS, labelsize=10)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out4 = os.path.join(ASSETS, "eval_calibration.png")
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"보정 곡선 저장: {out4}")

print(f"\n완료 (1~4). 저장 위치: {ASSETS}")


# ──────────────────────────────────────────────
# 오분류 분석 데이터 준비
# ──────────────────────────────────────────────
team_df = pd.concat(fold_team_preds, ignore_index=True)

team_df["tp"] = ((team_df["pred_top5"] == 1) & (team_df["postseason"] == 1)).astype(int)
team_df["tn"] = ((team_df["pred_top5"] == 0) & (team_df["postseason"] == 0)).astype(int)
team_df["fp"] = ((team_df["pred_top5"] == 1) & (team_df["postseason"] == 0)).astype(int)
team_df["fn"] = ((team_df["pred_top5"] == 0) & (team_df["postseason"] == 1)).astype(int)

def _result_type(row):
    if row["tp"]: return "TP"
    if row["fp"]: return "FP"
    if row["fn"]: return "FN"
    return "TN"

team_df["result"] = team_df.apply(_result_type, axis=1)

# 시즌별 FP/FN 집계
season_err = (
    team_df.groupby("season")[["fp", "fn"]]
    .sum()
    .reset_index()
)
print("\n[오분류 집계]")
print(season_err.to_string(index=False))


# ──────────────────────────────────────────────
# 차트 5: 시즌별 FP / FN 발생 팀 (발산형 막대)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
ax.set_facecolor(BG)

COLOR_FP = "#CC4444"   # 과대예측 (빨강)
COLOR_FN = "#F4900C"   # 과소예측 (주황)

x_pos = np.arange(len(TEST_SEASONS))
width = 0.55

for i, season in enumerate(TEST_SEASONS):
    fps = team_df[(team_df["season"] == season) & (team_df["fp"] == 1)]["team"].tolist()
    fns = team_df[(team_df["season"] == season) & (team_df["fn"] == 1)]["team"].tolist()

    ax.bar(i, len(fps),  width, color=COLOR_FP, alpha=0.80, edgecolor="white")
    ax.bar(i, -len(fns), width, color=COLOR_FN, alpha=0.80, edgecolor="white")

    # FP 팀명 (막대 위)
    for j, team in enumerate(fps):
        ax.text(i, j + 0.08, team, ha="center", va="bottom",
                fontsize=9, color=COLOR_FP, fontweight="bold")

    # FN 팀명 (막대 아래)
    for j, team in enumerate(fns):
        ax.text(i, -(j + 0.15), team, ha="center", va="top",
                fontsize=9, color="#C05000", fontweight="bold")

ax.axhline(0, color="#AAAAAA", linewidth=1.0)

# 범례 패치
from matplotlib.patches import Patch
legend_handles = [
    Patch(color=COLOR_FP, alpha=0.8, label="FP — 포스트시즌 예측했으나 탈락 (과대예측)"),
    Patch(color=COLOR_FN, alpha=0.8, label="FN — 탈락 예측했으나 포스트시즌 진출 (과소예측)"),
]
ax.legend(handles=legend_handles, fontsize=10, loc="upper right",
          framealpha=0.9, edgecolor="#DDDDDD")

ax.set_xticks(x_pos)
ax.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=11)
ax.set_yticks(range(-5, 6))
ax.set_yticklabels(
    [f"FN {abs(v)}팀" if v < 0 else (f"FP {v}팀" if v > 0 else "0")
     for v in range(-5, 6)],
    fontsize=9,
)
ax.set_xlabel("테스트 시즌", fontsize=11, color="#444444", labelpad=8)
ax.set_title(
    "시즌별 오분류 현황: FP (과대예측) vs FN (과소예측)",
    fontsize=14, fontweight="bold", color="#1B1B1B", pad=20,
)
ax.text(0.5, 1.02,
        "FP: 포스트시즌 예측 → 실제 탈락  |  FN: 탈락 예측 → 실제 포스트시즌 진출",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=9)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out5 = os.path.join(ASSETS, "eval_fp_fn_season.png")
plt.savefig(out5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"FP/FN 시즌 차트 저장: {out5}")


# ──────────────────────────────────────────────
# 차트 6: 팀 × 시즌 오분류 히트맵
# ──────────────────────────────────────────────
RESULT_COLOR = {"TP": "#2E8B57", "TN": "#E8E8E8", "FP": "#CC4444", "FN": "#F4900C"}
RESULT_NUM   = {"TP": 3, "TN": 0, "FP": 2, "FN": 1}

# 오류 빈도 순으로 팀 정렬 (FP+FN 많은 팀이 위)
team_err_total = (
    team_df.groupby("team")[["fp", "fn"]].sum()
    .assign(total=lambda d: d["fp"] + d["fn"])
    .sort_values("total", ascending=True)
)
team_order = team_err_total.index.tolist()

pivot = team_df.pivot(index="team", columns="season", values="result").reindex(team_order)
color_matrix = pivot.map(lambda v: RESULT_NUM.get(v, 0))

# 배경 이미지용 숫자 행렬 → matplotlib imshow로 색 지정
from matplotlib.colors import ListedColormap
cmap_custom = ListedColormap(
    [RESULT_COLOR["TN"], RESULT_COLOR["FN"], RESULT_COLOR["FP"], RESULT_COLOR["TP"]]
)

fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
ax.set_facecolor(BG)

ax.imshow(color_matrix.values, cmap=cmap_custom, vmin=0, vmax=3,
          aspect="auto", interpolation="nearest")

# 셀 텍스트 (result type + prob)
for i, team in enumerate(team_order):
    for j, season in enumerate(TEST_SEASONS):
        result = pivot.loc[team, season] if season in pivot.columns else "TN"
        prob   = team_df[(team_df["team"] == team) & (team_df["season"] == season)]["prob_cv"].values
        prob_v = prob[0] if len(prob) else 0.0
        bright = result in ("TN",)
        txt_color = "#666666" if bright else "white"
        ax.text(j, i,
                f"{result}\n{prob_v:.2f}",
                ha="center", va="center", fontsize=8.5, color=txt_color,
                fontweight="bold" if result in ("FP", "FN") else "normal")

# 오른쪽에 FP+FN 합계 레이블
for i, team in enumerate(team_order):
    total = team_err_total.loc[team, "total"]
    fp_n  = team_err_total.loc[team, "fp"]
    fn_n  = team_err_total.loc[team, "fn"]
    ax.text(len(TEST_SEASONS) + 0.05, i,
            f"  FP{fp_n} FN{fn_n}",
            va="center", fontsize=9, color="#555555")

ax.set_xticks(range(len(TEST_SEASONS)))
ax.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=10)
ax.set_yticks(range(len(team_order)))
ax.set_yticklabels(team_order, fontsize=10)
ax.set_xlim(-0.5, len(TEST_SEASONS) + 1.2)
ax.set_title(
    "팀별 · 시즌별 예측 결과 히트맵",
    fontsize=14, fontweight="bold", color="#1B1B1B", pad=16,
)
ax.text(0.5, 1.02,
        "TP: 맞음(진출)  TN: 맞음(탈락)  FP: 과대예측(빨강)  FN: 과소예측(주황)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

# 범례
from matplotlib.patches import Patch as _Patch
ax.legend(
    handles=[_Patch(color=v, label=k) for k, v in RESULT_COLOR.items()],
    loc="lower left", bbox_to_anchor=(0, -0.12), ncol=4,
    fontsize=9, framealpha=0.9, edgecolor="#DDDDDD",
)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
out6 = os.path.join(ASSETS, "eval_team_error_heatmap.png")
plt.savefig(out6, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"팀별 오분류 히트맵 저장: {out6}")

print(f"\n완료 (1~6). 저장 위치: {ASSETS}")


# ──────────────────────────────────────────────
# 차트 7: 학습 손실 곡선 (Log Loss per Round)
# 대표 폴드: Train 2017~2024 / Test 2025
# ──────────────────────────────────────────────
import lightgbm as _lgb

print("\n학습 손실 곡선 계산 중 (2025 폴드 기준)...")

train_lc = df[df["season"] != 2025].copy()
test_lc  = df[df["season"] == 2025].copy()

X_tr_lc  = train_lc[FEATURE_COLS]
y_tr_lc  = train_lc["postseason"]
X_te_lc  = test_lc[FEATURE_COLS]
y_te_lc  = test_lc["postseason"]

s_min_lc = train_lc["season"].min()
s_max_lc = train_lc["season"].max()
sw_lc    = (0.3 + 0.7 * (train_lc["season"] - s_min_lc) / (s_max_lc - s_min_lc)).values
pos_w_lc = (y_tr_lc == 0).sum() / max((y_tr_lc == 1).sum(), 1)

# XGBoost — eval_set으로 라운드별 logloss 수집
xgb_lc = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
    scale_pos_weight=pos_w_lc, eval_metric="logloss", random_state=42,
)
xgb_lc.fit(
    X_tr_lc, y_tr_lc, sample_weight=sw_lc,
    eval_set=[(X_tr_lc, y_tr_lc), (X_te_lc, y_te_lc)],
    verbose=False,
)
xgb_res       = xgb_lc.evals_result()
xgb_tr_loss   = xgb_res["validation_0"]["logloss"]
xgb_val_loss  = xgb_res["validation_1"]["logloss"]
print("  XGBoost 완료")

# LightGBM — record_evaluation 콜백으로 수집
lgbm_eval_res = {}
lgbm_lc = LGBMClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=5.0, min_child_samples=20,
    scale_pos_weight=pos_w_lc, random_state=42, verbose=-1,
)
lgbm_lc.fit(
    X_tr_lc, y_tr_lc, sample_weight=sw_lc,
    eval_set=[(X_tr_lc, y_tr_lc), (X_te_lc, y_te_lc)],
    eval_names=["train", "valid"],
    eval_metric="binary_logloss",
    callbacks=[_lgb.record_evaluation(lgbm_eval_res)],
)
lgbm_tr_loss  = lgbm_eval_res["train"]["binary_logloss"]
lgbm_val_loss = lgbm_eval_res["valid"]["binary_logloss"]
print("  LightGBM 완료")

rounds = np.arange(1, 201)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

for ax, tr_loss, val_loss, title in [
    (ax1, xgb_tr_loss,  xgb_val_loss,  "XGBoost"),
    (ax2, lgbm_tr_loss, lgbm_val_loss, "LightGBM"),
]:
    ax.plot(rounds, tr_loss,  color=ACCENT,   linewidth=2.2, label="Train Loss",      zorder=3)
    ax.plot(rounds, val_loss, color=WARM_RED,  linewidth=2.2, label="Validation Loss", zorder=3)

    # 두 곡선 사이 갭 음영
    ax.fill_between(rounds, tr_loss, val_loss,
                    where=[v > t for t, v in zip(tr_loss, val_loss)],
                    alpha=0.08, color=WARM_RED, label="과적합 구간")

    # Val Loss 최저점 마커
    min_idx = int(np.argmin(val_loss))
    ax.scatter(min_idx + 1, val_loss[min_idx], color=WARM_RED, s=70, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.text(min_idx + 4, val_loss[min_idx],
            f"Val 최저\n{val_loss[min_idx]:.4f}\n({min_idx + 1}R)",
            fontsize=8, color=WARM_RED, va="center")

    # 최종 시점 Train/Val 수치 주석
    ax.annotate(f"Train {tr_loss[-1]:.4f}", xy=(200, tr_loss[-1]),
                xytext=(175, tr_loss[-1] - 0.015),
                fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.8))
    ax.annotate(f"Val {val_loss[-1]:.4f}", xy=(200, val_loss[-1]),
                xytext=(175, val_loss[-1] + 0.015),
                fontsize=8, color=WARM_RED,
                arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.8))

    ax.set_xlabel("부스팅 라운드 (n_estimators)", fontsize=11, color="#444444")
    ax.set_ylabel("Log Loss", fontsize=11, color="#444444")
    ax.set_title(f"{title} 학습 손실 곡선",
                 fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
    ax.text(0.5, 1.01,
            "Train/Val 격차가 클수록 과적합  |  Val Loss가 올라가면 조기 종료 필요",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    ax.set_xlim(0, 210)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(GRAY_AXIS)
    ax.tick_params(colors=GRAY_AXIS, labelsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle(
    "학습 손실 곡선 (Train 2017~2024 / Validation 2025)",
    fontsize=13, fontweight="bold", color="#1B1B1B", y=1.02,
)
plt.tight_layout()
out7 = os.path.join(ASSETS, "eval_loss_curves.png")
plt.savefig(out7, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"학습 손실 곡선 저장: {out7}")

print(f"\n완료. 저장 위치: {ASSETS}")
print(f"  {os.path.basename(out1)}  — CV 성적표")
print(f"  {os.path.basename(out2)}  — ROC 커브")
print(f"  {os.path.basename(out3)}  — 과적합 체크")
print(f"  {os.path.basename(out4)}  — 보정 곡선")
print(f"  {os.path.basename(out5)}  — FP/FN 시즌별")
print(f"  {os.path.basename(out6)}  — 팀별 오분류 히트맵")
print(f"  {os.path.basename(out7)}  — 학습 손실 곡선")
