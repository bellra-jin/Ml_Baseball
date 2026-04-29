"""
predict_2026_postseason.py

2017~2025 전체 데이터로 앙상블 모델을 학습하고
2026 시즌 현재까지의 경기를 기반으로 포스트시즌 진출 확률을 예측한다.

피처셋: FEATURE_COLS (36개)
  - 현재 시즌 순위·성적 (18개)
  - 전년도 핵심 지표 prev_ (9개)
  - 3년 평균 역가중 dyn_ (9개) — 시즌 후반으로 갈수록 과거 전력 비중 0에 수렴

앙상블: XGBoost + LightGBM + RandomForest (동일 가중 평균)

결과물: notebooks/experiments/jh/08_predict_2026/
  - predict_2026_bar.png       — 팀별 최신 시점 포스트시즌 확률 바 차트
  - predict_2026_trend.png     — 팀별 확률 추이 (시즌 진행도별)
  - predict_2026_importance.png — 피처 중요도 Top 20
  - predict_2026_result.csv    — 팀별 최신 예측 확률 수치

실행: uv run python "notebooks/experiments/jh/predict_2026_postseason.py"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import FEATURE_COLS

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
ROOT   = os.path.join(os.path.dirname(__file__), "../../..")
OUTDIR = os.path.join(os.path.dirname(__file__), "kbo_prediction_2026")
os.makedirs(OUTDIR, exist_ok=True)


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
print("데이터 로드 중...")
train_df = pd.read_csv(os.path.join(ROOT, "data/modeling/train_dataset.csv"))
pred_df  = pd.read_csv(os.path.join(ROOT, "data/modeling/predict_dataset_2026.csv"))

train_df["date"] = pd.to_datetime(train_df["date"])
pred_df["date"]  = pd.to_datetime(pred_df["date"])

print(f"학습 데이터: {train_df.shape}  ({sorted(train_df['season'].unique())})")
print(f"예측 데이터: {pred_df.shape}")
print(f"예측 기간:   {pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
print(f"시즌 진행도: {pred_df['games_played_ratio'].max():.1%}\n")

# 누락 피처 체크
missing = [c for c in FEATURE_COLS if c not in train_df.columns]
if missing:
    print(f"[경고] 학습 데이터 누락 피처: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in train_df.columns]

missing_pred = [c for c in FEATURE_COLS if c not in pred_df.columns]
if missing_pred:
    print(f"[경고] 예측 데이터 누락 피처: {missing_pred}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in pred_df.columns]

print(f"사용 피처 수: {len(FEATURE_COLS)}개\n")


# ─────────────────────────────────────────────
# 앙상블 모델 학습
# ─────────────────────────────────────────────
X_train = train_df[FEATURE_COLS]
y_train = train_df["postseason"]

# 최근 시즌일수록 높은 가중치 (0.3 ~ 1.0)
s_min, s_max = train_df["season"].min(), train_df["season"].max()
sample_weight = (0.3 + 0.7 * (train_df["season"] - s_min) / (s_max - s_min)).values

pos_w = (y_train == 0).sum() / (y_train == 1).sum()

print("모델 학습 중...")

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

xgb.fit(X_train, y_train, sample_weight=sample_weight)
print("  XGBoost 완료")
lgbm.fit(X_train, y_train, sample_weight=sample_weight)
print("  LightGBM 완료")
rf.fit(X_train, y_train, sample_weight=sample_weight)
print("  RandomForest 완료\n")


# ─────────────────────────────────────────────
# 2026 예측
# ─────────────────────────────────────────────
X_pred = pred_df[FEATURE_COLS]

prob_raw = (
    xgb.predict_proba(X_pred)[:, 1] +
    lgbm.predict_proba(X_pred)[:, 1] +
    rf.predict_proba(X_pred)[:, 1]
) / 3

pred_df = pred_df.copy()
pred_df["prob_raw"] = prob_raw

# 날짜별 정규화: 하루 10팀 확률 합 = 5 (포스트시즌 팀 수), 최대 1.0
pred_df["prob_norm"] = pred_df.groupby("date")["prob_raw"].transform(
    lambda x: (x / x.sum() * 5).clip(upper=1.0)
)


# ─────────────────────────────────────────────
# 최신 시점 결과 출력
# ─────────────────────────────────────────────
latest = pred_df.sort_values("date").groupby("team").last().reset_index()
latest = latest.sort_values("prob_norm", ascending=False)

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
result_csv.to_csv(os.path.join(OUTDIR, "predict_2026_result.csv"), index=False, encoding="utf-8-sig")
print(f"결과 CSV 저장: {OUTDIR}/predict_2026_result.csv\n")


# ─────────────────────────────────────────────
# 공통 스타일 설정
# ─────────────────────────────────────────────
BG        = "#F8F9FA"
GRAY_AXIS = "#AAAAAA"
TOP5_COLORS  = ["#1B3F7A", "#2563A8", "#3E84C8", "#6BADD6", "#9ECAE1"]
BOTTOM_COLOR = "#D5D5D5"

# KBO 팀 공식 컬러 — trend / bump / scatter / radar 차트에 적용
# bar / heatmap / importance 는 기존 스타일 유지
TEAM_COLORS = {
    "KIA":  "#ea0029",  # KIA 타이거즈
    "삼성":  "#074CA1",  # 삼성 라이온즈
    "LG":   "#a50034",  # LG 트윈스
    "두산":  "#1a1748",  # 두산 베어스
    "KT":   "#000000",  # KT 위즈
    "SSG":  "#ce0e2d",  # SSG 랜더스
    "롯데":  "#041E42",  # 롯데 자이언츠
    "한화":  "#FC4E00",  # 한화 이글스
    "NC":   "#315288",  # NC 다이노스
    "키움":  "#570514",  # 키움 히어로즈
}
# 상위 5팀: alpha=1.0 / 하위 5팀: alpha=0.30 (팀 색상 유지, 투명도로 구분)
ALPHA_TOP    = 1.00
ALPHA_BOTTOM = 0.30

ref_date  = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y.%m.%d")
ref_ratio = latest["games_played_ratio"].mean()

bar_order = latest.sort_values("prob_norm", ascending=True).reset_index(drop=True)
top5_teams = set(latest.head(5)["team"])

# 바 색상: 상위 5팀은 진한 파랑→연한 파랑 그라데이션, 하위 5팀은 회색
bar_colors = []
top5_ranks = {team: i for i, team in enumerate(latest.head(5)["team"])}
for team in bar_order["team"]:
    if team in top5_ranks:
        bar_colors.append(TOP5_COLORS[top5_ranks[team]])
    else:
        bar_colors.append(BOTTOM_COLOR)


# ─────────────────────────────────────────────
# 차트 1: 포스트시즌 확률 바 차트
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
ax.set_facecolor(BG)

bars = ax.barh(
    bar_order["team"], bar_order["prob_norm"],
    color=bar_colors, height=0.6, edgecolor="white", linewidth=0.5,
)

# 5위/6위 경계선
ax.axhline(4.5, color="#888888", linewidth=1.2, linestyle="--", alpha=0.6)
ax.text(0.01, 4.55, "── 포스트시즌 컷라인", color="#888888", fontsize=9, va="bottom")

# 50% 기준선
ax.axvline(0.5, color="#CC4444", linewidth=1.0, linestyle=":", alpha=0.8)

# 확률 레이블
for bar, val, team in zip(bars, bar_order["prob_norm"], bar_order["team"]):
    is_top5 = team in top5_teams
    x_pos   = val + 0.012
    color   = "#222222" if is_top5 else "#666666"
    weight  = "bold"    if is_top5 else "normal"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left", fontsize=11,
            color=color, fontweight=weight)

# 순위 레이블 (왼쪽)
for i, (team, val) in enumerate(zip(bar_order["team"], bar_order["prob_norm"])):
    rank = len(bar_order) - i
    marker = "★" if team in top5_teams else f"{rank}위"
    color  = TOP5_COLORS[top5_ranks[team]] if team in top5_teams else "#AAAAAA"
    ax.text(-0.03, i, marker, va="center", ha="right",
            fontsize=10, color=color, fontweight="bold")

ax.set_xlim(-0.04, 1.18)
ax.set_xlabel("포스트시즌 진출 확률", fontsize=11, color="#444444", labelpad=8)
ax.set_title(
    f"2026 KBO 포스트시즌 진출 확률 예측",
    fontsize=15, fontweight="bold", color="#1B1B1B", pad=32,
)
ax.text(0.5, 1.02,
        f"기준: {ref_date}  |  시즌 {ref_ratio:.1%} 경과  |  앙상블 XGBoost + LightGBM + RandomForest",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
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

# 포스트시즌 영역 배경
ax.axhspan(0.5, 1.05, alpha=0.04, color="#1B3F7A", zorder=0)
ax.axhline(0.5, color="#CC4444", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
ax.text(0.5, 0.505, "포스트시즌 기준선 (50%)",
        color="#CC4444", fontsize=8.5, va="bottom", ha="left")

x_end = pred_df["games"].max()

for team in sorted(pred_df["team"].unique()):
    t     = pred_df[pred_df["team"] == team].sort_values("games")
    x     = t["games"].values
    y     = t["prob_norm"].values
    color = TEAM_COLORS[team]

    if team in top5_teams:
        ax.plot(x, y, color=color, linewidth=2.4, alpha=ALPHA_TOP, zorder=3)
        ax.text(x[-1] + 0.4, y[-1], team,
                color=color, fontsize=10, fontweight="bold",
                va="center", ha="left")
    else:
        ax.plot(x, y, color=color, linewidth=1.1,
                linestyle="--", alpha=ALPHA_BOTTOM, zorder=2)
        ax.text(x[-1] + 0.4, y[-1], team,
                color=mcolors.to_rgba(color, 0.5),
                fontsize=9, va="center", ha="left")

ax.set_xlim(0, x_end + 8)
ax.set_ylim(0, 1.05)
ax.set_xlabel("누적 경기 수", fontsize=11, color="#444444", labelpad=8)
ax.set_ylabel("포스트시즌 진출 확률", fontsize=11, color="#444444", labelpad=8)
ax.set_title(
    "2026 KBO 포스트시즌 확률 추이",
    fontsize=15, fontweight="bold", color="#1B1B1B", pad=32,
)
ax.text(0.5, 1.02,
        f"기준: {ref_date}  |  진한 실선: 현재 상위 5팀",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
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
imp_xgb  = pd.Series(xgb.feature_importances_,  index=FEATURE_COLS)
imp_lgbm = pd.Series(lgbm.feature_importances_, index=FEATURE_COLS)
imp_rf   = pd.Series(rf.feature_importances_,   index=FEATURE_COLS)

imp = (
    imp_xgb  / imp_xgb.sum() +
    imp_lgbm / imp_lgbm.sum() +
    imp_rf   / imp_rf.sum()
) / 3

top20 = imp.sort_values(ascending=False).head(20)

GROUP_COLOR = {"dyn_": "#2E8B57", "prev_": "#2563A8", "other": "#888888"}
GROUP_LABEL = {"dyn_": "3년 평균 역가중 (dyn_)", "prev_": "전년도 기록 (prev_)", "other": "현재 시즌"}

def feat_group(name):
    if name.startswith("dyn_"):  return "dyn_"
    if name.startswith("prev_"): return "prev_"
    return "other"

palette_imp = [GROUP_COLOR[feat_group(f)] for f in top20.index]

fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG)
ax.set_facecolor(BG)

bars = ax.barh(
    range(len(top20)), top20.values[::-1],
    color=palette_imp[::-1], height=0.65,
    edgecolor="white", linewidth=0.4,
)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index[::-1], fontsize=10)

for bar, val in zip(bars, top20.values[::-1]):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="#444444")

# 범례
from matplotlib.patches import Patch
legend_handles = [Patch(color=v, label=GROUP_LABEL[k]) for k, v in GROUP_COLOR.items()]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10,
          framealpha=0.9, edgecolor="#DDDDDD")

ax.set_xlabel("중요도 (XGB + LGBM + RF 평균 정규화)", fontsize=11, color="#444444", labelpad=8)
ax.set_title(
    "피처 중요도 Top 20",
    fontsize=15, fontweight="bold", color="#1B1B1B", pad=32,
)
ax.text(0.5, 1.02,
        "3개 모델(XGBoost · LightGBM · RandomForest) 중요도 평균",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
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
# date 기준 순위 계산 (같은 날 10팀 동시에 비교)
rank_df = (
    pred_df
    .sort_values(["date", "prob_norm"], ascending=[True, False])
    .assign(pred_rank=lambda d: d.groupby("date").cumcount() + 1)
    [["date", "team", "pred_rank", "prob_norm"]]
)

fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
ax.set_facecolor(BG)

dates_ordered = sorted(rank_df["date"].unique())
date_to_x = {d: i for i, d in enumerate(dates_ordered)}

for team in sorted(rank_df["team"].unique()):
    t     = rank_df[rank_df["team"] == team].sort_values("date")
    xs    = [date_to_x[d] for d in t["date"]]
    ys    = t["pred_rank"].values
    color = TEAM_COLORS[team]

    if team in top5_teams:
        ax.plot(xs, ys, color=color, linewidth=2.6, alpha=ALPHA_TOP,
                zorder=3, solid_capstyle="round", solid_joinstyle="round")
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]],
                   color=color, s=60, zorder=4, edgecolors="white", linewidths=1.5)
        ax.text(xs[-1] + 0.3, ys[-1], team,
                color=color, fontsize=10, fontweight="bold",
                va="center", ha="left")
    else:
        ax.plot(xs, ys, color=color, linewidth=1.2,
                linestyle="--", alpha=ALPHA_BOTTOM, zorder=2)
        ax.text(xs[-1] + 0.3, ys[-1], team,
                color=mcolors.to_rgba(color, 0.5),
                fontsize=9, va="center", ha="left")

# 포스트시즌 컷라인
ax.axhline(5.5, color="#CC4444", linewidth=1.0, linestyle=":",
           alpha=0.7, zorder=1)
ax.text(0, 5.62, "포스트시즌 컷라인", color="#CC4444", fontsize=8.5, va="bottom")

# x축: 5일 간격 날짜 레이블
n = len(dates_ordered)
tick_step  = max(1, n // 6)
tick_xs    = list(range(0, n, tick_step))
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
ax.text(0.5, 1.02,
        f"기준: {ref_date}  |  진한 실선: 현재 상위 5팀  |  점: 시작·최신 시점",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
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

x_win  = latest["win_rate"].values
y_prob = latest["prob_norm"].values
teams_ordered = latest["team"].values

# 배경 사분면 구분선
ax.axvline(0.5, color="#DDDDDD", linewidth=1.0, zorder=0)
ax.axhline(0.5, color="#DDDDDD", linewidth=1.0, zorder=0)

# 사분면 레이블
quad_kw = dict(fontsize=8.5, color="#BBBBBB", ha="center", va="center")
ax.text(0.35, 0.78, "현재 부진\n모델 낙관", **quad_kw)   # 좌상 (과대평가)
ax.text(0.68, 0.78, "현재 강세\n모델 낙관", **quad_kw)   # 우상 (정상)
ax.text(0.35, 0.22, "현재 부진\n모델 비관", **quad_kw)   # 좌하 (정상)
ax.text(0.68, 0.22, "현재 강세\n모델 비관", **quad_kw)   # 우하 (과소평가)

# 기준선 (y=x, 모델=현재승률)
diag = np.linspace(0.2, 0.85, 100)
ax.plot(diag, diag, color="#CCCCCC", linewidth=1.0, linestyle="--",
        alpha=0.8, zorder=1, label="모델확률 = 현재승률")

for team, xv, yv in zip(teams_ordered, x_win, y_prob):
    is_top5 = team in top5_teams
    base_color = TEAM_COLORS.get(team, "#888888")
    alpha_val  = ALPHA_TOP if is_top5 else ALPHA_BOTTOM
    color      = mcolors.to_rgba(base_color, alpha_val)
    size       = 160 if is_top5 else 100

    ax.scatter(xv, yv, color=color, s=size, zorder=3,
               edgecolors="white", linewidths=1.2)

    # 팀명 오프셋 (겹침 방지를 위해 팀별로 조정)
    dx, dy = 0.012, 0.012
    ax.annotate(
        team, (xv, yv),
        xytext=(xv + dx, yv + dy),
        fontsize=10,
        fontweight="bold" if is_top5 else "normal",
        color=color,
        ha="left", va="bottom",
    )

ax.set_xlim(0.25, 0.80)
ax.set_ylim(0.0, 1.08)
ax.set_xlabel("현재 승률 (실제 성적)", fontsize=12, color="#444444", labelpad=8)
ax.set_ylabel("모델 예측 확률 (정규화)", fontsize=12, color="#444444", labelpad=8)
ax.set_title("현재 승률 vs 모델 예측 확률",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02,
        f"기준: {ref_date}  |  대각선 위 = 모델이 현재보다 낙관  |  대각선 아래 = 모델이 현재보다 비관",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")

ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color(GRAY_AXIS)
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
# 팀을 최신 확률 내림차순으로 정렬 → 상위팀이 위에 배치
team_order  = list(latest.sort_values("prob_norm", ascending=False)["team"])
dates_list  = sorted(pred_df["date"].unique())
date_labels = [pd.to_datetime(d).strftime("%m/%d") for d in dates_list]

# 히트맵 행렬 (팀 수 × 날짜 수)
heatmap_data = np.zeros((len(team_order), len(dates_list)))
for i, team in enumerate(team_order):
    for j, date in enumerate(dates_list):
        val = pred_df[(pred_df["team"] == team) & (pred_df["date"] == date)]["prob_norm"]
        heatmap_data[i, j] = val.values[0] if len(val) else np.nan

fig, ax = plt.subplots(figsize=(15, 6), facecolor=BG)
ax.set_facecolor(BG)

im = ax.imshow(
    heatmap_data, aspect="auto", cmap="Blues",
    vmin=0.0, vmax=1.0, interpolation="nearest",
)

# 5위/6위 경계선 (y 기준 4.5)
ax.axhline(4.5, color="#CC4444", linewidth=1.5, linestyle="--", alpha=0.8)
ax.text(len(dates_list) - 0.4, 4.35, "컷라인",
        color="#CC4444", fontsize=8.5, ha="right", va="bottom")

# 셀 안에 확률 텍스트
for i in range(len(team_order)):
    for j in range(len(dates_list)):
        val = heatmap_data[i, j]
        txt_color = "white" if val > 0.6 else "#333333"
        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                fontsize=7.5, color=txt_color)

# 축 설정
ax.set_yticks(range(len(team_order)))
ax.set_yticklabels(
    [f"{'★' if t in top5_teams else '  '} {t}" for t in team_order],
    fontsize=10,
)
tick_step = max(1, len(dates_list) // 8)
ax.set_xticks(range(0, len(dates_list), tick_step))
ax.set_xticklabels(date_labels[::tick_step], fontsize=9, rotation=0)

# 컬러바
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("포스트시즌 진출 확률", fontsize=10, color="#444444")
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
cbar.ax.tick_params(labelsize=9, colors=GRAY_AXIS)

ax.set_title("2026 KBO 포스트시즌 확률 히트맵",
             fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
ax.text(0.5, 1.02,
        f"기준: {ref_date}  |  ★ 현재 예측 상위 5팀  |  진할수록 진출 확률 높음",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777777")

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
out6 = os.path.join(OUTDIR, "predict_2026_heatmap.png")
plt.savefig(out6, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"히트맵 저장: {out6}")


# ─────────────────────────────────────────────
# 차트 7: 레이더 차트 (팀별 핵심 지표 프로파일)
# ─────────────────────────────────────────────
RADAR_METRICS = {
    "종합전력\n(피타고라스)": "prev_pythagorean_win_rate",
    "득실차":               "prev_run_differential",
    "투수력\n(ERA↓)":       "prev_team_era",
    "에이스\n(ERA↓)":       "prev_ace_era",
    "타격력\n(OPS)":        "prev_top5_hitter_ops_avg",
    "장타력\n(ISO)":        "prev_iso",
}
LOWER_IS_BETTER = {"prev_team_era", "prev_ace_era"}

prev_2026 = pd.read_csv(
    os.path.join(ROOT, "data/processed/2026/prev_features_from_2025.csv")
)
radar_df = prev_2026[["team"] + list(RADAR_METRICS.values())].copy()

# 정규화: min-max, ERA 계열은 반전 (낮을수록 좋음 → 높은 값으로)
for col in RADAR_METRICS.values():
    mn, mx = radar_df[col].min(), radar_df[col].max()
    radar_df[col] = (radar_df[col] - mn) / (mx - mn)
    if col in LOWER_IS_BETTER:
        radar_df[col] = 1 - radar_df[col]

labels   = list(RADAR_METRICS.keys())
n_labels = len(labels)
angles   = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
angles  += angles[:1]  # 닫힌 다각형

# 상위 5팀만 레이더에 표시
radar_teams = list(latest.head(5)["team"])

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True}, facecolor=BG)
ax.set_facecolor(BG)

for i, team in enumerate(radar_teams):
    row    = radar_df[radar_df["team"] == team]
    values = [row[col].values[0] for col in RADAR_METRICS.values()]
    values += values[:1]
    color  = TEAM_COLORS.get(team, "#888888")

    ax.plot(angles, values, color=color, linewidth=2.2, zorder=3)
    ax.fill(angles, values, color=color, alpha=0.10)
    # 최대값 꼭짓점에 팀명 마커
    peak_idx = int(np.argmax(values[:-1]))
    ax.scatter(angles[peak_idx], values[peak_idx],
               color=color, s=60, zorder=4, edgecolors="white", linewidths=1.2)

# 축 레이블
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10.5, color="#333333")
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color=GRAY_AXIS)
ax.set_ylim(0, 1)
ax.spines["polar"].set_color("#DDDDDD")
ax.grid(color="#E0E0E0", linewidth=0.8)

# 범례
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=TEAM_COLORS.get(team, "#888888"), linewidth=2.5, label=team)
    for team in radar_teams
]
ax.legend(handles=legend_handles, loc="upper right",
          bbox_to_anchor=(1.28, 1.12), fontsize=10,
          framealpha=0.9, edgecolor="#DDDDDD")

ax.set_title("예측 상위 5팀 핵심 지표 프로파일\n(전년도 기록 기준, 높을수록 유리)",
             fontsize=13, fontweight="bold", color="#1B1B1B",
             pad=32, y=1.06)
ax.text(0.5, -0.06,
        f"기준: 2025 시즌 기록  |  ERA 계열은 낮을수록 좋아 반전 정규화 적용",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")

plt.tight_layout()
out7 = os.path.join(OUTDIR, "predict_2026_radar.png")
plt.savefig(out7, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print(f"레이더 차트 저장: {out7}")

print("\n완료. 결과물 폴더:", OUTDIR)
