# ─────────────────────────────────────────────
# Strategy C 2026 예측 결과 저장 모듈
# ─────────────────────────────────────────────
"""
Strategy C 2026 예측 결과 저장 모듈.

원본 `predict_2026_postseason copy.py`의 아래 구간을 함수 단위로 분리했다.

- 결과 CSV 저장
- 차트 1: 포스트시즌 확률 바 차트
- 차트 2: 팀별 확률 추이
- 차트 3: 피처 중요도 Top 20
- 차트 4: 날짜별 예측 순위 변화(범프 차트)
- 차트 5: 현재 승률 vs 예측 확률 산점도
- 차트 6: 팀 x 날짜 확률 히트맵
- 차트 7: 예측 상위 5팀 핵심 지표 레이더 차트

모델 학습과 예측 확률 계산은 `strategy_c_postseason.py`가 담당하고,
이 파일은 이미 계산된 `pred_df`, `latest`, `final_model`을 받아 파일로 저장하는 역할만 맡는다.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from notebooks.experiments.jh.strategy_c_style import (
    ALPHA_BOTTOM,
    ALPHA_TOP,
    BG,
    BOTTOM_COLOR,
    GRAY_AXIS,
    TEAM_COLORS,
    TOP5_COLORS,
)

SUBTITLE = "LR 25% + RF 25% + lightXGB 25% + lightLGBM 25%  |  피처 20개"


# ─────────────────────────────────────────────
# 공통 그림 저장
# ─────────────────────────────────────────────
def _save_fig(path: Path) -> str:
    """공통 그림 저장 옵션을 적용하고 저장 경로를 문자열로 반환한다."""
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return str(path)


# ─────────────────────────────────────────────
# 공통 축 스타일
# ─────────────────────────────────────────────
def _soft_axes(ax, grid_axis: str = "y") -> None:
    """차트마다 반복되는 축/그리드 스타일을 적용한다."""
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(GRAY_AXIS)
    ax.tick_params(colors=GRAY_AXIS, labelsize=10)
    ax.set_axisbelow(True)
    if grid_axis in {"x", "both"}:
        ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.8)
    if grid_axis in {"y", "both"}:
        ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)


# ─────────────────────────────────────────────
# 예측 차트 공통 컨텍스트
# ─────────────────────────────────────────────
def _prediction_context(latest):
    """
    예측 차트 여러 곳에서 공통으로 쓰는 최신 기준일, 진행도, 정렬 순서, 색상을 만든다.

    bar chart는 낮은 확률 팀이 아래에서 위로 쌓이도록 ascending 정렬을 쓰고,
    현재 top5 팀은 순위별 파란 팔레트로 강조한다.
    """
    ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y.%m.%d")
    ref_ratio = latest["games_played_ratio"].mean()
    bar_order = latest.sort_values("prob_norm", ascending=True).reset_index(drop=True)
    top5_ranks = {team: i for i, team in enumerate(latest.head(5)["team"])}
    bar_colors = [
        TOP5_COLORS[top5_ranks[team]] if team in top5_ranks else BOTTOM_COLOR
        for team in bar_order["team"]
    ]
    return ref_date, ref_ratio, bar_order, top5_ranks, bar_colors


# ─────────────────────────────────────────────
# 피처 그룹 분류
# ─────────────────────────────────────────────
def _feat_group(name: str) -> str:
    """피처명을 현재 시즌 / 전년도(prev_) / 3년 평균 역가중(dyn_) 그룹으로 분류한다."""
    if name.startswith("dyn_"):
        return "dyn_"
    if name.startswith("prev_"):
        return "prev_"
    return "other"


# ─────────────────────────────────────────────
# 결과 CSV 저장
# ─────────────────────────────────────────────
def save_result_csv(latest, outdir: str | Path) -> str:
    """최신 기준 팀별 2026 포스트시즌 예측 결과를 CSV로 저장한다."""
    out_path = Path(outdir) / "predict_2026_result.csv"
    result_csv = latest[["team", "date", "games", "games_played_ratio", "prob_raw", "prob_norm"]].copy()
    result_csv.columns = ["팀", "기준일", "경기수", "시즌진행도", "원시확률", "정규화확률"]
    result_csv.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"결과 CSV 저장: {out_path}\n")
    return str(out_path)


# ─────────────────────────────────────────────
# 차트 1: 포스트시즌 확률 바 차트
# ─────────────────────────────────────────────
def save_bar_chart(latest, top5_teams, outdir: str | Path) -> str:
    """
    차트 1: 최신 기준 팀별 포스트시즌 진출 확률 바 차트를 저장한다.

    현재 예측 top5는 파란 계열로 강조하고, 5위와 6위 사이에 포스트시즌 컷라인을 표시한다.
    """
    ref_date, ref_ratio, bar_order, top5_ranks, bar_colors = _prediction_context(latest)
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(BG)

    bars = ax.barh(bar_order["team"], bar_order["prob_norm"], color=bar_colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.axhline(4.5, color="#888888", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.text(0.01, 4.55, "── 포스트시즌 컷라인", color="#888888", fontsize=9, va="bottom")
    ax.axvline(0.5, color="#CC4444", linewidth=1.0, linestyle=":", alpha=0.8)

    for bar, val, team in zip(bars, bar_order["prob_norm"], bar_order["team"]):
        is_top5 = team in top5_teams
        ax.text(
            val + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            ha="left",
            fontsize=11,
            color="#222222" if is_top5 else "#666666",
            fontweight="bold" if is_top5 else "normal",
        )

    for i, (team, _) in enumerate(zip(bar_order["team"], bar_order["prob_norm"])):
        rank = len(bar_order) - i
        marker = "★" if team in top5_teams else f"{rank}위"
        color = TOP5_COLORS[top5_ranks[team]] if team in top5_teams else "#AAAAAA"
        ax.text(-0.03, i, marker, va="center", ha="right", fontsize=10, color=color, fontweight="bold")

    ax.set_xlim(-0.04, 1.18)
    ax.set_xlabel("포스트시즌 진출 확률", fontsize=11, color="#444444", labelpad=8)
    ax.set_title("2026 KBO 포스트시즌 진출 확률 예측", fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
    ax.text(0.5, 1.02, f"기준: {ref_date}  |  시즌 {ref_ratio:.1%} 경과  |  {SUBTITLE}", transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRAY_AXIS)
    ax.tick_params(axis="x", colors=GRAY_AXIS, labelsize=10)
    ax.tick_params(axis="y", left=False, labelsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.8)

    plt.tight_layout()
    out_path = _save_fig(Path(outdir) / "predict_2026_bar.png")
    print(f"바 차트 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 차트 2: 팀별 확률 추이
# ─────────────────────────────────────────────
def save_trend_chart(pred_df, latest, top5_teams, outdir: str | Path) -> str:
    """
    차트 2: 시즌 진행에 따른 팀별 정규화 확률 추이를 저장한다.

    현재 top5 팀은 진한 실선, 나머지 팀은 흐린 점선으로 표시해 최신 경쟁 구도를 빠르게 읽게 한다.
    """
    ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y.%m.%d")
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axhspan(0.5, 1.05, alpha=0.04, color="#1B3F7A", zorder=0)
    ax.axhline(0.5, color="#CC4444", lw=1.0, ls="--", alpha=0.6, zorder=1)
    ax.text(0.5, 0.505, "포스트시즌 기준선 (50%)", color="#CC4444", fontsize=8.5, va="bottom", ha="left")

    x_end = pred_df["games"].max()
    for team in sorted(pred_df["team"].unique()):
        t = pred_df[pred_df["team"] == team].sort_values("games")
        x = t["games"].values
        y = t["prob_norm"].values
        color = TEAM_COLORS[team]
        if team in top5_teams:
            ax.plot(x, y, color=color, lw=2.4, alpha=ALPHA_TOP, zorder=3)
            ax.text(x[-1] + 0.4, y[-1], team, color=color, fontsize=10, fontweight="bold", va="center", ha="left")
        else:
            ax.plot(x, y, color=color, lw=1.1, ls="--", alpha=ALPHA_BOTTOM, zorder=2)
            ax.text(x[-1] + 0.4, y[-1], team, color=mcolors.to_rgba(color, 0.5), fontsize=9, va="center", ha="left")

    ax.set_xlim(0, x_end + 8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("누적 경기 수", fontsize=11, color="#444444", labelpad=8)
    ax.set_ylabel("포스트시즌 진출 확률", fontsize=11, color="#444444", labelpad=8)
    ax.set_title("2026 KBO 포스트시즌 확률 추이", fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
    ax.text(0.5, 1.02, f"기준: {ref_date}  |  진한 실선: 현재 상위 5팀", transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _soft_axes(ax)

    plt.tight_layout()
    out_path = _save_fig(Path(outdir) / "predict_2026_trend.png")
    print(f"추이 차트 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 차트 3: 피처 중요도
# ─────────────────────────────────────────────
def save_importance_chart(final_model, feature_cols, outdir: str | Path) -> str:
    """
    차트 3: 최종 모델의 피처 중요도 Top 20을 저장한다.

    LR은 계수 스케일이 다른 모델이라 제외하고, XGB/LGBM/RF의 중요도를 각각 정규화한 뒤 평균낸다.
    """
    imp_xgb = pd.Series(final_model.xgb.feature_importances_, index=feature_cols)
    imp_lgbm = pd.Series(final_model.lgbm.feature_importances_, index=feature_cols)
    imp_rf = pd.Series(final_model.rf.feature_importances_, index=feature_cols)
    imp = (imp_xgb / imp_xgb.sum() + imp_lgbm / imp_lgbm.sum() + imp_rf / imp_rf.sum()) / 3
    top_imp = imp.sort_values(ascending=False)

    group_color = {"dyn_": "#0ea5e9", "prev_": "#2563A8", "other": "#888888"}
    group_label = {"dyn_": "3년 평균 역가중 (dyn_)", "prev_": "전년도 기록 (prev_)", "other": "현재 시즌"}
    palette_imp = [group_color[_feat_group(f)] for f in top_imp.index]

    fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(range(len(top_imp)), top_imp.values[::-1], color=palette_imp[::-1], height=0.65, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp.index[::-1], fontsize=10)
    for bar, val in zip(bars, top_imp.values[::-1]):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9, color="#444444")

    legend_handles = [Patch(color=v, label=group_label[k]) for k, v in group_color.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    ax.set_xlabel("중요도 (XGB + LGBM + RF 평균 정규화)", fontsize=11, color="#444444", labelpad=8)
    ax.set_title("피처 중요도 Top 20", fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
    ax.text(0.5, 1.02, "XGBoost · LightGBM · RandomForest 중요도 평균 (LR 제외)", transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
    _soft_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", left=False)

    plt.tight_layout()
    out_path = _save_fig(Path(outdir) / "predict_2026_importance.png")
    print(f"피처 중요도 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 차트 4: 예측 순위 변화
# ─────────────────────────────────────────────
def save_bump_chart(pred_df, latest, top5_teams, outdir: str | Path) -> str:
    """
    차트 4: 날짜별 예측 순위 변화(범프 차트)를 저장한다.

    날짜마다 `prob_norm` 기준 순위를 다시 매겨 각 팀의 예측 순위가 어떻게 이동했는지 보여준다.
    """
    ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y.%m.%d")
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
    n = len(dates_ordered)

    for team in sorted(rank_df["team"].unique()):
        t = rank_df[rank_df["team"] == team].sort_values("date")
        xs = [date_to_x[d] for d in t["date"]]
        ys = t["pred_rank"].values
        color = TEAM_COLORS[team]
        if team in top5_teams:
            ax.plot(xs, ys, color=color, lw=2.6, alpha=ALPHA_TOP, zorder=3, solid_capstyle="round", solid_joinstyle="round")
            ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], color=color, s=60, zorder=4, edgecolors="white", linewidths=1.5)
            ax.text(xs[-1] + 0.3, ys[-1], team, color=color, fontsize=10, fontweight="bold", va="center", ha="left")
        else:
            ax.plot(xs, ys, color=color, lw=1.2, ls="--", alpha=ALPHA_BOTTOM, zorder=2)
            ax.text(xs[-1] + 0.3, ys[-1], team, color=mcolors.to_rgba(color, 0.5), fontsize=9, va="center", ha="left")

    ax.axhline(5.5, color="#CC4444", lw=1.0, ls=":", alpha=0.7, zorder=1)
    ax.text(0, 5.62, "포스트시즌 컷라인", color="#CC4444", fontsize=8.5, va="bottom")
    tick_step = max(1, n // 6)
    tick_xs = list(range(0, n, tick_step))
    tick_labels = [pd.to_datetime(dates_ordered[i]).strftime("%m/%d") for i in tick_xs]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_xlim(-0.5, n + 1.5)
    ax.set_ylim(10.5, 0.5)
    ax.set_yticks(range(1, 11))
    ax.set_yticklabels([f"{i}위" for i in range(1, 11)], fontsize=10)
    ax.set_xlabel("날짜", fontsize=11, color="#444444", labelpad=8)
    ax.set_ylabel("예측 순위", fontsize=11, color="#444444", labelpad=8)
    ax.set_title("2026 KBO 포스트시즌 예측 순위 변화", fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
    ax.text(0.5, 1.02, f"기준: {ref_date}  |  진한 실선: 현재 상위 5팀  |  점: 시작·최신 시점", transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
    _soft_axes(ax, grid_axis="x")

    plt.tight_layout()
    out_path = _save_fig(Path(outdir) / "predict_2026_bump.png")
    print(f"범프 차트 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 차트 5: 현재 승률 vs 예측 확률
# ─────────────────────────────────────────────
def save_scatter_chart(latest, top5_teams, outdir: str | Path) -> str:
    """
    차트 5: 현재 승률과 모델 예측 확률의 관계를 산점도로 저장한다.

    대각선 위는 현재 승률보다 모델이 더 낙관적으로 보는 팀, 아래는 더 보수적으로 보는 팀이다.
    """
    ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y.%m.%d")
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG)
    ax.set_facecolor(BG)
    x_win = latest["win_rate"].values
    y_prob = latest["prob_norm"].values
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
        is_top5 = team in top5_teams
        base_color = TEAM_COLORS.get(team, "#888888")
        color = mcolors.to_rgba(base_color, ALPHA_TOP if is_top5 else ALPHA_BOTTOM)
        ax.scatter(xv, yv, color=color, s=160 if is_top5 else 100, zorder=3, edgecolors="white", linewidths=1.2)
        ax.annotate(team, (xv, yv), xytext=(xv + 0.012, yv + 0.012), fontsize=10, fontweight="bold" if is_top5 else "normal", color=color, ha="left", va="bottom")

    ax.set_xlim(0.25, 0.80)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("현재 승률 (실제 성적)", fontsize=12, color="#444444", labelpad=8)
    ax.set_ylabel("모델 예측 확률 (정규화)", fontsize=12, color="#444444", labelpad=8)
    ax.set_title("현재 승률 vs 모델 예측 확률", fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
    ax.text(0.5, 1.02, f"기준: {ref_date}  |  대각선 위 = 모델 낙관  |  대각선 아래 = 모델 비관", transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _soft_axes(ax, grid_axis="both")

    plt.tight_layout()
    out_path = _save_fig(Path(outdir) / "predict_2026_scatter.png")
    print(f"산점도 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 차트 6: 확률 히트맵
# ─────────────────────────────────────────────
def save_heatmap_chart(pred_df, latest, top5_teams, outdir: str | Path) -> str:
    """
    차트 6: 팀 x 날짜 형태의 포스트시즌 확률 히트맵을 저장한다.

    팀은 최신 예측 확률 순으로 정렬하고, 날짜별 확률 변화를 한 화면에서 비교한다.
    """
    ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y.%m.%d")
    team_order = list(latest.sort_values("prob_norm", ascending=False)["team"])
    dates_list = sorted(pred_df["date"].unique())
    date_labels = [pd.to_datetime(d).strftime("%m/%d") for d in dates_list]

    heatmap_data = np.zeros((len(team_order), len(dates_list)))
    for i, team in enumerate(team_order):
        for j, date in enumerate(dates_list):
            val = pred_df[(pred_df["team"] == team) & (pred_df["date"] == date)]["prob_norm"]
            heatmap_data[i, j] = val.values[0] if len(val) else np.nan

    fig, ax = plt.subplots(figsize=(15, 6), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(heatmap_data, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
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
    ax.set_title("2026 KBO 포스트시즌 확률 히트맵", fontsize=15, fontweight="bold", color="#1B1B1B", pad=32)
    ax.text(0.5, 1.02, f"기준: {ref_date}  |  ★ 현재 예측 상위 5팀  |  진할수록 진출 확률 높음", transform=ax.transAxes, ha="center", fontsize=9, color="#777777")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    out_path = _save_fig(Path(outdir) / "predict_2026_heatmap.png")
    print(f"히트맵 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 차트 7: 상위 5팀 레이더 차트
# ─────────────────────────────────────────────
def save_radar_chart(latest, root: str | Path, outdir: str | Path) -> str:
    """
    차트 7: 예측 상위 5팀의 전년도 핵심 지표 레이더 차트를 저장한다.

    2026 예측의 전년도 기반 전력 신호를 설명하기 위해 2025년 prev feature를 사용한다.
    ERA 계열은 낮을수록 좋으므로 정규화 후 값을 반전한다.
    """
    radar_metrics = {
        "종합전력\n(피타고라스)": "prev_pythagorean_win_rate",
        "득실차": "prev_run_differential",
        "투수력\n(ERA↓)": "prev_team_era",
        "에이스\n(ERA↓)": "prev_ace_era",
        "타격력\n(OPS)": "prev_top5_hitter_ops_avg",
        "장타력\n(ISO)": "prev_iso",
    }
    lower_is_better = {"prev_team_era", "prev_ace_era"}

    prev_2026 = pd.read_csv(Path(root) / "data/processed/2026/prev_features_from_2025.csv")
    radar_df = prev_2026[["team"] + list(radar_metrics.values())].copy()

    # 팀 간 비교를 위해 각 지표를 0~1로 min-max 정규화한다.
    for col in radar_metrics.values():
        mn, mx = radar_df[col].min(), radar_df[col].max()
        radar_df[col] = (radar_df[col] - mn) / (mx - mn)
        if col in lower_is_better:
            radar_df[col] = 1 - radar_df[col]

    labels = list(radar_metrics.keys())
    n_labels = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
    angles += angles[:1]
    radar_teams = list(latest.head(5)["team"])

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True}, facecolor=BG)
    ax.set_facecolor(BG)
    for team in radar_teams:
        row = radar_df[radar_df["team"] == team]
        values = [row[col].values[0] for col in radar_metrics.values()]
        values += values[:1]
        color = TEAM_COLORS.get(team, "#888888")
        ax.plot(angles, values, color=color, lw=2.2, zorder=3)
        ax.fill(angles, values, color=color, alpha=0.10)
        peak_idx = int(np.argmax(values[:-1]))
        ax.scatter(angles[peak_idx], values[peak_idx], color=color, s=60, zorder=4, edgecolors="white", linewidths=1.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5, color="#333333")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color=GRAY_AXIS)
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color("#DDDDDD")
    ax.grid(color="#E0E0E0", linewidth=0.8)
    legend_handles = [Line2D([0], [0], color=TEAM_COLORS.get(team, "#888888"), lw=2.5, label=team) for team in radar_teams]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    ax.set_title("예측 상위 5팀 핵심 지표 프로파일\n(전년도 기록 기준, 높을수록 유리)", fontsize=13, fontweight="bold", color="#1B1B1B", pad=32, y=1.06)
    ax.text(0.5, -0.06, "기준: 2025 시즌 기록  |  ERA 계열은 낮을수록 좋아 반전 정규화 적용", transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")

    plt.tight_layout()
    out_path = _save_fig(Path(outdir) / "predict_2026_radar.png")
    print(f"레이더 차트 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 예측 산출물 일괄 저장
# ─────────────────────────────────────────────
def save_prediction_outputs(
    train_df,
    pred_df,
    latest,
    top5_teams,
    final_model,
    feature_cols,
    root: str | Path,
    outdir: str | Path,
) -> dict[str, str]:
    """
    예측 결과 CSV와 7개 차트를 모두 저장한다.

    `strategy_c_report.py`에서는 이 함수 하나만 호출해 예측 산출물 전체를 생성한다.
    반환값은 산출물 이름과 저장 경로의 매핑이다.
    """
    _ = train_df
    return {
        "result_csv": save_result_csv(latest, outdir),
        "bar": save_bar_chart(latest, top5_teams, outdir),
        "trend": save_trend_chart(pred_df, latest, top5_teams, outdir),
        "importance": save_importance_chart(final_model, feature_cols, outdir),
        "bump": save_bump_chart(pred_df, latest, top5_teams, outdir),
        "scatter": save_scatter_chart(latest, top5_teams, outdir),
        "heatmap": save_heatmap_chart(pred_df, latest, top5_teams, outdir),
        "radar": save_radar_chart(latest, root, outdir),
    }
