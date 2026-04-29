"""
10_low_overfit_eval.py

과적합 갭 0.10 이하를 목표로 3가지 전략 비교.
피처는 09_feature_reduced_eval.py에서 선정된 상위 20개를 그대로 사용.

전략 A — Logistic Regression 단독  (선형 모델 → 학습 데이터 외우기 불가)
전략 B — LR 40% + RF 60%           (부스팅 제거)
전략 C — LR 25% + RF 25% + 극소형 XGB 25% + LGBM 25%
          (XGB/LGBM: depth=2, 40라운드 — 부스팅을 최소한만 사용)

기준선 (09_feature_reduced_eval.py, 20피처 앙상블):
  Test AUC 0.8072 / Train AUC 1.0000 / 갭 0.1928

시각화 저장: notebooks/experiments/jh/Validation/assets/gap10_*.png
실행: uv run python "notebooks/experiments/jh/Validation/10_low_overfit_eval.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config import FEATURE_COLS as ALL_FEATURE_COLS
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
ORANGE    = "#E07B20"

# 09_feature_reduced_eval.py 에서 선정된 상위 20개 피처 (고정)
TOP_FEATURES = [
    "rank", "win_rate", "games_behind_5th",
    "prev_pythagorean_win_rate", "prev_team_era", "prev_ops_concentration",
    "prev_bb_rate", "prev_top5_hitter_ops_avg", "prev_ace_era",
    "prev_run_differential", "prev_k_bb_ratio", "wins_to_5th",
    "games_behind", "home_win_rate", "dyn_run_differential",
    "prev_iso", "away_win_rate", "dyn_bb_rate",
    "dyn_pythagorean_win_rate", "dyn_k_bb_ratio",
]

# 기준선 (09 결과)
BASELINE = {
    "label":     "기존 앙상블 (XGB+LGBM+RF, 20피처)",
    "test_auc":  0.8072,
    "train_auc": 1.0000,
    "gap":       0.1928,
    "brier":     0.1941,
    "season_auc": {
        2018: 0.9409, 2019: 0.6626, 2020: 0.9717,
        2021: 0.6100, 2022: 0.8646, 2023: 0.7956,
        2024: 0.9188, 2025: 0.6932,
    },
}

TEST_SEASONS = list(range(2018, 2026))

# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data/modeling/train_dataset.csv"))
df["date"] = pd.to_datetime(df["date"])
print(f"데이터: {df.shape}  |  피처: {len(TOP_FEATURES)}개 (고정)\n")


# ──────────────────────────────────────────────
# 전략별 LOSO-CV
# ──────────────────────────────────────────────
def run_cv(strategy_name, predict_fn, build_fn):
    """
    LOSO-CV를 실행하고 fold별 metrics를 반환한다.
    build_fn(pos_w) → model(s)
    predict_fn(models, X_tr, y_tr, sw, X_es, y_es, X_te) → prob_te, prob_tr
    """
    print(f"\n{'='*55}")
    print(f"전략: {strategy_name}")
    print(f"{'='*55}")
    results = []
    for test_season in TEST_SEASONS:
        train = df[df["season"] != test_season].copy()
        test  = df[df["season"] == test_season].copy()

        X_tr, y_tr = train[TOP_FEATURES], train["postseason"]
        X_te, y_te = test[TOP_FEATURES],  test["postseason"]

        s_min = train["season"].min()
        s_max = train["season"].max()
        sw = (0.3 + 0.7 * (train["season"] - s_min) / max(s_max - s_min, 1)).values
        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        _last = train["season"].max()
        X_es  = train.loc[train["season"] == _last, TOP_FEATURES]
        y_es  = train.loc[train["season"] == _last, "postseason"]

        models = build_fn(pos_w)
        prob_te, prob_tr = predict_fn(models, X_tr, y_tr, sw, X_es, y_es, X_te)

        m_te = evaluate_binary_model(y_te, prob_te)
        m_tr = evaluate_binary_model(y_tr, prob_tr)
        results.append({
            "season":    test_season,
            "test_auc":  m_te["roc_auc"],
            "train_auc": m_tr["roc_auc"],
            "brier":     m_te["brier"],
        })
        print_metrics(m_te, label=str(test_season))

    df_res = pd.DataFrame(results)
    gap = (df_res["train_auc"] - df_res["test_auc"]).mean()
    print(f"\n  Test AUC  = {df_res['test_auc'].mean():.4f}  |  "
          f"Train AUC = {df_res['train_auc'].mean():.4f}  |  "
          f"갭 = {gap:.4f}  |  Brier = {df_res['brier'].mean():.4f}")
    return df_res


# ── 전략 A: Logistic Regression 단독 ──────────
def build_A(pos_w):
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C=0.1, max_iter=2000, random_state=42,
            class_weight="balanced", solver="lbfgs",
        )),
    ])
    return lr

def predict_A(lr, X_tr, y_tr, sw, X_es, y_es, X_te):
    lr.fit(X_tr, y_tr)
    return lr.predict_proba(X_te)[:, 1], lr.predict_proba(X_tr)[:, 1]


# ── 전략 B: LR 40% + RF 60% ──────────────────
def build_B(pos_w):
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
    return lr, rf

def predict_B(models, X_tr, y_tr, sw, X_es, y_es, X_te):
    lr, rf = models
    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr, sample_weight=sw)
    prob_te = 0.4 * lr.predict_proba(X_te)[:, 1] + 0.6 * rf.predict_proba(X_te)[:, 1]
    prob_tr = 0.4 * lr.predict_proba(X_tr)[:, 1] + 0.6 * rf.predict_proba(X_tr)[:, 1]
    return prob_te, prob_tr


# ── 전략 C: LR 25% + RF 25% + 극소형 XGB 25% + LGBM 25% ──
def build_C(pos_w):
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

def predict_C(models, X_tr, y_tr, sw, X_es, y_es, X_te):
    lr, rf, xgb, lgbm = models
    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr, sample_weight=sw)
    xgb.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    lgbm.fit(X_tr, y_tr, sample_weight=sw)
    prob_te = (
        lr.predict_proba(X_te)[:, 1] +
        rf.predict_proba(X_te)[:, 1] +
        xgb.predict_proba(X_te)[:, 1] +
        lgbm.predict_proba(X_te)[:, 1]
    ) / 4
    prob_tr = (
        lr.predict_proba(X_tr)[:, 1] +
        rf.predict_proba(X_tr)[:, 1] +
        xgb.predict_proba(X_tr)[:, 1] +
        lgbm.predict_proba(X_tr)[:, 1]
    ) / 4
    return prob_te, prob_tr


# ── 실행 ──────────────────────────────────────
df_A = run_cv("A — Logistic Regression 단독",       predict_A, build_A)
df_B = run_cv("B — LR 40% + RF 60%",                predict_B, build_B)
df_C = run_cv("C — LR + RF + 극소형 XGB + LGBM",   predict_C, build_C)


# ──────────────────────────────────────────────
# 결과 요약 테이블
# ──────────────────────────────────────────────
strategies = [
    ("기존 앙상블 (기준)",   BASELINE["test_auc"],  BASELINE["train_auc"],
     BASELINE["gap"],       BASELINE["brier"]),
    ("A — LR 단독",         df_A["test_auc"].mean(), df_A["train_auc"].mean(),
     (df_A["train_auc"] - df_A["test_auc"]).mean(), df_A["brier"].mean()),
    ("B — LR+RF",           df_B["test_auc"].mean(), df_B["train_auc"].mean(),
     (df_B["train_auc"] - df_B["test_auc"]).mean(), df_B["brier"].mean()),
    ("C — LR+RF+XGB+LGBM",  df_C["test_auc"].mean(), df_C["train_auc"].mean(),
     (df_C["train_auc"] - df_C["test_auc"]).mean(), df_C["brier"].mean()),
]

print("\n" + "=" * 70)
print(f"{'전략':<24} {'Test AUC':>9} {'Train AUC':>10} {'갭':>7} {'Brier':>8}")
print("─" * 70)
for name, te, tr, gap, brier in strategies:
    gap_mark = " [목표달성]" if gap < 0.10 else ""
    print(f"{name:<24} {te:>9.4f} {tr:>10.4f} {gap:>7.4f} {brier:>8.4f}{gap_mark}")
print("=" * 70)


# ──────────────────────────────────────────────
# 차트 1: 핵심 지표 비교 (갭 + Test AUC)
# ──────────────────────────────────────────────
labels   = ["기존 앙상블\n(기준)", "A\nLR 단독", "B\nLR+RF", "C\nLR+RF\n+XGB+LGBM"]
gaps     = [s[3] for s in strategies]
test_aucs = [s[1] for s in strategies]
brierss  = [s[4] for s in strategies]
colors_bar = [GRAY_AXIS, GREEN, "#6BADD6", ACCENT]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

# ── 왼쪽: 과적합 갭 ──
bars = ax1.bar(labels, gaps, color=colors_bar, alpha=0.85,
               width=0.55, edgecolor="white", linewidth=0.5)
ax1.axhline(0.10, color=WARM_RED, linewidth=1.5, linestyle="--", alpha=0.8,
            label="목표 갭 0.10")
ax1.axhline(0.05, color=WARM_RED, linewidth=0.8, linestyle=":", alpha=0.5,
            label="이상적 갭 0.05")

for bar, val in zip(bars, gaps):
    color = GREEN if val < 0.10 else (ORANGE if val < 0.15 else WARM_RED)
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.003,
             f"{val:.4f}", ha="center", va="bottom",
             fontsize=10, color=color, fontweight="bold")

ax1.set_ylim(0, max(gaps) * 1.3)
ax1.set_ylabel("과적합 갭 (Train AUC − Test AUC)", fontsize=11, color="#444444")
ax1.set_title("과적합 갭 비교", fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax1.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax1.spines[spine].set_color(GRAY_AXIS)
ax1.tick_params(colors=GRAY_AXIS, labelsize=10)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

# ── 오른쪽: Test AUC ──
bars2 = ax2.bar(labels, test_aucs, color=colors_bar, alpha=0.85,
                width=0.55, edgecolor="white", linewidth=0.5)
ax2.axhline(BASELINE["test_auc"], color=GRAY_AXIS, linewidth=1.2,
            linestyle="--", alpha=0.7, label=f"기존 기준선 {BASELINE['test_auc']:.4f}")

for bar, val in zip(bars2, test_aucs):
    diff = val - BASELINE["test_auc"]
    color = GREEN if diff >= 0 else WARM_RED
    label = f"{val:.4f}\n({'+' if diff >= 0 else ''}{diff:.4f})"
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             label, ha="center", va="bottom",
             fontsize=9, color=color, fontweight="bold")

ax2.set_ylim(0.5, max(test_aucs) * 1.15)
ax2.set_ylabel("Test ROC-AUC", fontsize=11, color="#444444")
ax2.set_title("Test AUC 비교 (높을수록 좋음)", fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color(GRAY_AXIS)
ax2.tick_params(colors=GRAY_AXIS, labelsize=10)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle("과적합 갭 0.10 이하 전략 비교  (20개 피처 / LOSO-CV 2018~2025)",
             fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
plt.tight_layout()
out1 = os.path.join(ASSETS, "gap10_comparison.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"\n핵심 지표 비교 저장: {out1}")


# ──────────────────────────────────────────────
# 차트 2: 시즌별 Test AUC (4개 전략 + 기준)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.set_facecolor(BG)

x = np.arange(len(TEST_SEASONS))
w = 0.18

auc_base = [BASELINE["season_auc"][s] for s in TEST_SEASONS]
auc_A    = df_A["test_auc"].tolist()
auc_B    = df_B["test_auc"].tolist()
auc_C    = df_C["test_auc"].tolist()

ax.bar(x - w * 1.5, auc_base, w, color=GRAY_AXIS,  alpha=0.7, label="기존 앙상블 (기준)", edgecolor="white")
ax.bar(x - w * 0.5, auc_A,    w, color=GREEN,       alpha=0.85, label="A — LR 단독",      edgecolor="white")
ax.bar(x + w * 0.5, auc_B,    w, color="#6BADD6",   alpha=0.85, label="B — LR+RF",        edgecolor="white")
ax.bar(x + w * 1.5, auc_C,    w, color=ACCENT,      alpha=0.85, label="C — LR+RF+XGB+LGBM", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=10)
ax.set_ylim(0.4, 1.12)
ax.set_xlabel("테스트 시즌", fontsize=11, color="#444444")
ax.set_ylabel("Test ROC-AUC", fontsize=11, color="#444444")
ax.set_title("시즌별 Test AUC — 전략별 비교",
             fontsize=14, fontweight="bold", color="#1B1B1B", pad=20)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
ax.tick_params(colors=GRAY_AXIS, labelsize=10)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

plt.tight_layout()
out2 = os.path.join(ASSETS, "gap10_season_auc.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"시즌별 AUC 저장: {out2}")

print(f"\n완료 (1~2). 저장 위치: {ASSETS}")


# ──────────────────────────────────────────────
# 전략 C: 36개 피처로 재실행 (피처 수 비교용)
# ──────────────────────────────────────────────
# run_cv의 feature set을 바꿔서 돌리기 위해 TOP_FEATURES 를 임시로 교체
TOP_FEATURES_36 = [c for c in ALL_FEATURE_COLS if c in df.columns]
print(f"\n36개 피처 유효 확인: {len(TOP_FEATURES_36)}개")


def run_cv_with_features(feat_list, strategy_name, predict_fn, build_fn):
    print(f"\n{'='*55}")
    print(f"전략: {strategy_name}  ({len(feat_list)}개 피처)")
    print(f"{'='*55}")
    results = []
    for test_season in TEST_SEASONS:
        train = df[df["season"] != test_season].copy()
        test  = df[df["season"] == test_season].copy()

        X_tr, y_tr = train[feat_list], train["postseason"]
        X_te, y_te = test[feat_list],  test["postseason"]

        s_min = train["season"].min()
        s_max = train["season"].max()
        sw = (0.3 + 0.7 * (train["season"] - s_min) / max(s_max - s_min, 1)).values
        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        _last = train["season"].max()
        X_es  = train.loc[train["season"] == _last, feat_list]
        y_es  = train.loc[train["season"] == _last, "postseason"]

        models = build_fn(pos_w)
        prob_te, prob_tr = predict_fn(models, X_tr, y_tr, sw, X_es, y_es, X_te)

        m_te = evaluate_binary_model(y_te, prob_te)
        m_tr = evaluate_binary_model(y_tr, prob_tr)
        results.append({
            "season":    test_season,
            "test_auc":  m_te["roc_auc"],
            "train_auc": m_tr["roc_auc"],
            "brier":     m_te["brier"],
        })
        print_metrics(m_te, label=str(test_season))

    df_res = pd.DataFrame(results)
    gap = (df_res["train_auc"] - df_res["test_auc"]).mean()
    print(f"\n  Test AUC  = {df_res['test_auc'].mean():.4f}  |  "
          f"Train AUC = {df_res['train_auc'].mean():.4f}  |  "
          f"갭 = {gap:.4f}  |  Brier = {df_res['brier'].mean():.4f}")
    return df_res

df_C36 = run_cv_with_features(
    TOP_FEATURES_36,
    "C — LR+RF+XGB+LGBM (36개 피처)",
    predict_C, build_C,
)


# ──────────────────────────────────────────────
# 최종 요약: 전략 C  20개 vs 36개
# ──────────────────────────────────────────────
gap_C20  = (df_C["train_auc"]  - df_C["test_auc"]).mean()
gap_C36  = (df_C36["train_auc"] - df_C36["test_auc"]).mean()

print("\n" + "=" * 65)
print(f"{'전략 C 피처 수 비교':<28} {'Test AUC':>9} {'Train AUC':>10} {'갭':>7} {'Brier':>8}")
print("─" * 65)
rows = [
    ("C — 20개 피처",  df_C["test_auc"].mean(),   df_C["train_auc"].mean(),   gap_C20, df_C["brier"].mean()),
    ("C — 36개 피처",  df_C36["test_auc"].mean(),  df_C36["train_auc"].mean(), gap_C36, df_C36["brier"].mean()),
]
for name, te, tr, gap, brier in rows:
    print(f"{name:<28} {te:>9.4f} {tr:>10.4f} {gap:>7.4f} {brier:>8.4f}")
print("=" * 65)


# ──────────────────────────────────────────────
# 차트 3: 전략 C  20개 vs 36개 비교
# ──────────────────────────────────────────────
x     = np.arange(len(TEST_SEASONS))
w     = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

auc_C20 = df_C["test_auc"].tolist()
auc_C36 = df_C36["test_auc"].tolist()

# ── 왼쪽: 시즌별 Test AUC ──
b20 = ax1.bar(x - w / 2, auc_C20, w, color=ACCENT,   alpha=0.85,
              label="전략 C — 20개 피처", edgecolor="white")
b36 = ax1.bar(x + w / 2, auc_C36, w, color=ORANGE,   alpha=0.80,
              label="전략 C — 36개 피처", edgecolor="white")

for i, (v20, v36) in enumerate(zip(auc_C20, auc_C36)):
    diff = v20 - v36
    color = GREEN if diff >= 0 else WARM_RED
    ax1.text(i, max(v20, v36) + 0.015,
             f"{'+' if diff >= 0 else ''}{diff:.3f}",
             ha="center", fontsize=8.5, color=color, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in TEST_SEASONS], fontsize=10)
ax1.set_ylim(0.4, 1.15)
ax1.set_xlabel("테스트 시즌", fontsize=11, color="#444444")
ax1.set_ylabel("Test ROC-AUC", fontsize=11, color="#444444")
ax1.set_title("시즌별 Test AUC (20개 vs 36개)", fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax1.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax1.spines[spine].set_color(GRAY_AXIS)
ax1.tick_params(colors=GRAY_AXIS, labelsize=10)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

# ── 오른쪽: 핵심 지표 요약 ──
metrics_labels = ["Test AUC 평균", "Train AUC 평균", "과적합 갭", "Brier"]
vals_20 = [df_C["test_auc"].mean(),  df_C["train_auc"].mean(),  gap_C20, df_C["brier"].mean()]
vals_36 = [df_C36["test_auc"].mean(), df_C36["train_auc"].mean(), gap_C36, df_C36["brier"].mean()]
lower_better = [False, False, True, True]

x2 = np.arange(len(metrics_labels))
w2 = 0.35
ax2.bar(x2 - w2 / 2, vals_20, w2, color=ACCENT,  alpha=0.85, label="20개 피처", edgecolor="white")
ax2.bar(x2 + w2 / 2, vals_36, w2, color=ORANGE,  alpha=0.80, label="36개 피처", edgecolor="white")

for j, (v20, v36, lower) in enumerate(zip(vals_20, vals_36, lower_better)):
    diff  = v20 - v36
    better = (diff < 0) if lower else (diff > 0)
    color = GREEN if better else WARM_RED
    ax2.text(j - w2 / 2, v20 + 0.008,
             f"{v20:.4f}", ha="center", fontsize=8, color=ACCENT, fontweight="bold")
    ax2.text(j + w2 / 2, v36 + 0.008,
             f"{v36:.4f}", ha="center", fontsize=8, color=ORANGE, fontweight="bold")
    ax2.text(j, max(v20, v36) + 0.035,
             f"{'+' if diff >= 0 else ''}{diff:.4f}",
             ha="center", fontsize=8, color=color, fontweight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(metrics_labels, fontsize=9.5)
ax2.set_ylim(0, 1.2)
ax2.set_title("핵심 지표 요약 (20개 vs 36개)", fontsize=13, fontweight="bold", color="#1B1B1B", pad=16)
ax2.text(0.5, 1.01, "갭·Brier: 낮을수록 좋음  |  초록=20개 우세  빨강=36개 우세",
         transform=ax2.transAxes, ha="center", fontsize=8.5, color="#777777")
ax2.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color(GRAY_AXIS)
ax2.tick_params(colors=GRAY_AXIS, labelsize=10)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)

fig.suptitle("전략 C (LR+RF+극소형 XGB+LGBM) — 20개 vs 36개 피처",
             fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
plt.tight_layout()
out3 = os.path.join(ASSETS, "gap10_C_feat_compare.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"피처 수 비교 차트 저장: {out3}")

print(f"\n완료. 저장 위치: {ASSETS}")
print(f"  {os.path.basename(out1)}  — 전략 A/B/C 핵심 지표")
print(f"  {os.path.basename(out2)}  — 전략 A/B/C 시즌별 AUC")
print(f"  {os.path.basename(out3)}  — 전략 C 20개 vs 36개 비교")
