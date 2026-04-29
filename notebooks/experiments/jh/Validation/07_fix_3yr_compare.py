"""
07_fix_3yr_compare.py

3yr 모델의 SSG 과대평가 문제를 두 가지 방향으로 수정해서 비교한다.

- baseline : BASE + avg3yr + trend  (현재 3yr 방식)
- opt1     : BASE + avg3yr + dev_   (trend → deviation으로 교체)
             dev_ = prev_ - avg3yr_  → 최근 성적이 3년 평균 대비 위/아래 위치
- opt2     : BASE + dyn_            (avg3yr에 진행도 역가중)
             dyn_ = (1 - games_played_ratio) × avg3yr_
             → 시즌 후반으로 갈수록 과거 데이터 비중이 0에 수렴

2024, 2025 두 시즌 모두 검증.

실행: uv run python "notebooks/experiments/jh /07_fix_3yr_compare.py"
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
from sklearn.metrics import roc_auc_score

from src.utils.config import FEATURE_COLS as BASE_COLS

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.join(os.path.dirname(__file__), "../../..")

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

CHECKPOINTS = {
    "50%  (72경기)": 0.50,
    "75% (108경기)": 0.75,
    "90% (130경기)": 0.90,
    "최종 (144경기)": 1.00,
}


# ──────────────────────────────────────────────
# 모델 학습 / 예측
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
# 파생 피처 생성 (전처리 없이 데이터셋에서 직접 계산)
# ──────────────────────────────────────────────
def add_derived_features(df):
    """dev_ 와 dyn_ 컬럼을 데이터프레임에 추가한다."""
    df = df.copy()
    for k in MULTI_YEAR_KEYS:
        prev_col  = f"prev_{k}"
        avg3_col  = f"avg3yr_{k}"
        dev_col   = f"dev_{k}"
        dyn_col   = f"dyn_{k}"

        if prev_col in df.columns and avg3_col in df.columns:
            # opt1: 최근 성적이 3년 평균 대비 위/아래
            df[dev_col] = df[prev_col] - df[avg3_col]
            # opt2: 진행도에 따라 역가중된 3년 평균
            df[dyn_col] = (1 - df["games_played_ratio"]) * df[avg3_col]

    return df


# ──────────────────────────────────────────────
# 단일 시즌 검증
# ──────────────────────────────────────────────
def validate_season(df, val_year, configs):
    train = df[df["season"] < val_year].copy()
    test  = df[df["season"] == val_year].copy()

    s_min, s_max = train["season"].min(), train["season"].max()
    sw = (0.3 + 0.7 * (train["season"] - s_min) / (s_max - s_min)).values

    y_train      = train["postseason"]
    y_test       = test["postseason"]
    actual_top5  = set(test[test["postseason"] == 1]["team"].unique())

    print(f"\n{'='*60}")
    print(f"[{val_year} 검증]  학습: {sorted(train['season'].unique())}")
    print(f"실제 포스트시즌: {sorted(actual_top5)}")
    print(f"{'='*60}")

    results = {}
    for label, cols in configs:
        models   = build_ensemble(train[cols], y_train, sw)
        prob_raw = predict_prob(models, test[cols])
        test     = test.copy()
        test[f"prob_{label}"]      = prob_raw
        test[f"prob_{label}_norm"] = normalize(test.groupby("date")[f"prob_{label}"])
        results[label] = roc_auc_score(y_test, prob_raw)

    # 시점별 적중률
    labels = [l for l, _ in configs]
    header = f"{'시점':<16} " + "  ".join(f"{l:>10}" for l in labels)
    print(f"\n{header}")
    print("─" * len(header))

    for cp_label, ratio in CHECKPOINTS.items():
        snap   = test[test["games_played_ratio"] <= ratio]
        latest = snap.sort_values("date").groupby("team").last().reset_index()
        row    = f"{cp_label:<16}"
        for label, _ in configs:
            top5 = set(latest.nlargest(5, f"prob_{label}_norm")["team"])
            hit  = len(top5 & actual_top5)
            row += f"  {hit}/5 {str(sorted(top5)):<32}"
        print(row)

    print("\nROC-AUC: " + "  ".join(f"{l}={v:.4f}" for l, v in results.items()))

    return test, actual_top5, results


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
df_raw = pd.read_csv(os.path.join(ROOT, "data/modeling/train_dataset.csv"))
df_raw["date"] = pd.to_datetime(df_raw["date"])
df = add_derived_features(df_raw)

AVG3_COLS = [f"avg3yr_{k}" for k in MULTI_YEAR_KEYS]
TREND_COLS = [f"trend_{k}"  for k in MULTI_YEAR_KEYS]
DEV_COLS   = [f"dev_{k}"    for k in MULTI_YEAR_KEYS]
DYN_COLS   = [f"dyn_{k}"    for k in MULTI_YEAR_KEYS]

configs = [
    ("baseline",  BASE_COLS + AVG3_COLS + TREND_COLS),  # 현재 3yr 방식
    ("opt1_dev",  BASE_COLS + AVG3_COLS + DEV_COLS),    # trend → deviation
    ("opt2_dyn",  BASE_COLS + DYN_COLS),                 # 진행도 역가중
]

print("피처 수:")
for label, cols in configs:
    print(f"  {label:<12}: {len(cols)}개")

# 2024, 2025 두 시즌 검증
all_results = {}
all_tests   = {}
all_top5    = {}

for val_year in [2024, 2025]:
    test_df, actual, auc_dict = validate_season(df, val_year, configs)
    all_results[val_year] = auc_dict
    all_tests[val_year]   = test_df
    all_top5[val_year]    = actual

# ──────────────────────────────────────────────
# 최종 시점 바 차트 (2024 / 2025 각 3종)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cfg_labels = [l for l, _ in configs]
colors_cfg = ["steelblue", "coral", "seagreen"]

for row_idx, val_year in enumerate([2024, 2025]):
    test_df    = all_tests[val_year]
    actual     = all_top5[val_year]
    final      = test_df.sort_values("date").groupby("team").last().reset_index()
    final      = final.sort_values("prob_baseline_norm", ascending=True)
    bar_colors = ["steelblue" if t in actual else "lightgray" for t in final["team"]]

    for col_idx, label in enumerate(cfg_labels):
        ax  = axes[row_idx][col_idx]
        col = f"prob_{label}_norm"
        bars = ax.barh(final["team"], final[col], color=bar_colors)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("포스트시즌 예측 확률")
        ax.set_title(
            f"{val_year} 최종 시점 — {label}\n"
            f"(파란색: 실제 포스트시즌 진출팀  ROC={all_results[val_year][label]:.3f})"
        )
        for bar, val in zip(bars, final[col]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
out = os.path.join(ROOT, "data/modeling/fix_3yr_compare_bar.png")
plt.savefig(out, dpi=110)
plt.show()
print(f"\n차트 저장: {out}")

# ──────────────────────────────────────────────
# 시점별 적중률 합산 바 차트 (2024 + 2025 합산)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, val_year in zip(axes, [2024, 2025]):
    test_df   = all_tests[val_year]
    actual    = all_top5[val_year]
    x = np.arange(len(CHECKPOINTS))
    w = 0.25

    for i, (label, _) in enumerate(configs):
        hits = []
        for ratio in CHECKPOINTS.values():
            snap   = test_df[test_df["games_played_ratio"] <= ratio]
            latest = snap.sort_values("date").groupby("team").last().reset_index()
            top5   = set(latest.nlargest(5, f"prob_{label}_norm")["team"])
            hits.append(len(top5 & actual))

        bars = ax.bar(x + (i - 1) * w, hits, w,
                      label=label, color=colors_cfg[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{int(bar.get_height())}/5", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(list(CHECKPOINTS.keys()), fontsize=10)
    ax.set_ylabel("적중 팀 수 (/ 5)")
    ax.set_ylim(0, 6)
    ax.set_yticks(range(6))
    ax.axhline(5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(f"{val_year} 시점별 적중률\nbaseline vs opt1(dev) vs opt2(dyn)")
    ax.legend(fontsize=10)

plt.tight_layout()
out2 = os.path.join(ROOT, "data/modeling/fix_3yr_compare_checkpoint.png")
plt.savefig(out2, dpi=110)
plt.show()
print(f"차트 저장: {out2}")
