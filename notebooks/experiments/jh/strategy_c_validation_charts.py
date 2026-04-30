"""
Strategy C LOSO-CV 검증 차트 저장 모듈.

원본 `predict_2026_postseason copy.py`의 "검증 V1~V5" 시각화 구간을
차트별 함수로 분리했다.

- V1. 성능 스코어카드: 시즌별 metric heatmap
- V2. 과적합 갭: Train AUC vs Test AUC와 gap bar
- V3. 로스 커브: XGBoost/LightGBM boosting round별 logloss
- V4. 캘리브레이션: reliability diagram과 확률 분포
- V5. 체크포인트: 시즌 진행률별 top5 적중 팀 수
"""

from pathlib import Path

import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve

from src.evaluation.metrics import checkpoint_hits
from notebooks.experiments.jh.strategy_c_postseason import TEST_SEASONS
from notebooks.experiments.jh.strategy_c_style import (
    ACCENT,
    BG,
    GRAY_AXIS,
    GREEN,
    ORANGE,
    WARM_RED,
)


# ─────────────────────────────────────────────
# 공통 저장/축 스타일 헬퍼
# ─────────────────────────────────────────────
def _save_fig(path: Path) -> str:
    """공통 저장 옵션을 적용하고 저장 경로를 문자열로 반환한다."""
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return str(path)


def _soft_axes(ax, grid_axis: str = "y") -> None:
    """반복되는 축 테두리/그리드 스타일을 적용한다."""
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
# 검증 V1: 성능 스코어카드
# ─────────────────────────────────────────────
def save_scorecard_chart(cv_df, valdir: str | Path, test_seasons=TEST_SEASONS) -> str:
    """
    시즌별 검증 metric을 heatmap 형태로 저장한다.

    metric마다 값의 방향이 다르므로 brier/gap은 낮을수록 좋게, 나머지는 높을수록 좋게 정규화한다.
    """
    metric_cols = ["test_auc", "train_auc", "gap", "f1", "precision", "recall", "brier"]
    metric_names = ["Test AUC", "Train AUC", "Gap (과적합)", "F1", "Precision", "Recall", "Brier"]
    lower_better = {"brier", "gap"}

    scorecard = cv_df.set_index("season")[metric_cols]
    n_m = len(metric_cols)
    n_s = len(test_seasons)
    norm_mat = np.zeros((n_m, n_s))

    for j, col in enumerate(metric_cols):
        # heatmap 색상은 metric별 상대 순위를 보여주기 위한 min-max 정규화 값이다.
        vals = scorecard[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            n = (vals - vmin) / (vmax - vmin)
            norm_mat[j] = (1 - n) if col in lower_better else n
        else:
            norm_mat[j] = 0.5

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(norm_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")

    for j in range(n_m):
        for i in range(n_s):
            raw = scorecard[metric_cols[j]].iloc[i]
            tc = "white" if norm_mat[j, i] < 0.2 or norm_mat[j, i] > 0.8 else "#333333"
            ax.text(i, j, f"{raw:.3f}", ha="center", va="center", fontsize=8.5, color=tc, fontweight="bold")

    ax.set_xticks(range(n_s))
    ax.set_xticklabels([str(s) for s in test_seasons], fontsize=10)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(metric_names, fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="상대적 성능 (초록=우수)")
    ax.set_title("Strategy C LOSO-CV 성능 스코어카드 (2018~2025)", fontsize=14, fontweight="bold", color="#1B1B1B", pad=20)
    ax.text(
        0.5,
        1.025,
        "LR 25% + RF 25% + lightXGB 25% + lightLGBM 25%  |  피처 20개",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#777777",
    )

    plt.tight_layout()
    out_path = _save_fig(Path(valdir) / "val_scorecard.png")
    print(f"[V1] 스코어카드 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 검증 V2: 과적합 갭
# ─────────────────────────────────────────────
def save_overfit_gap_chart(cv_df, valdir: str | Path, test_seasons=TEST_SEASONS) -> str:
    """
    Train/Test AUC와 과적합 gap을 나란히 보여주는 차트를 저장한다.

    gap이 작을수록 일반화 성능이 안정적이므로 0.10, 평균 gap 기준선을 함께 표시한다.
    """
    gap_mean = cv_df["gap"].mean()
    n_s = len(test_seasons)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    x = np.arange(n_s)
    w = 0.35
    ax1.bar(x - w / 2, cv_df["train_auc"], w, color=WARM_RED, alpha=0.75, label="Train AUC", edgecolor="white")
    ax1.bar(x + w / 2, cv_df["test_auc"], w, color=ACCENT, alpha=0.85, label="Test AUC", edgecolor="white")
    ax1.axhline(
        cv_df["test_auc"].mean(),
        color=ACCENT,
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=f"평균 Test {cv_df['test_auc'].mean():.3f}",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in test_seasons], fontsize=10)
    ax1.set_ylim(0.5, 1.12)
    ax1.set_ylabel("ROC-AUC", fontsize=11, color="#444444")
    ax1.set_title("Train vs Test AUC", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
    ax1.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
    _soft_axes(ax1)

    gap_colors = [GREEN if g < 0.10 else (ORANGE if g < 0.15 else WARM_RED) for g in cv_df["gap"]]
    bars_gap = ax2.bar(x, cv_df["gap"], color=gap_colors, alpha=0.85, width=0.55, edgecolor="white")
    ax2.axhline(gap_mean, color="#555555", linewidth=1.5, linestyle="--", alpha=0.8, label=f"평균 갭 {gap_mean:.3f}")
    ax2.axhline(0.10, color=WARM_RED, linewidth=1.2, linestyle=":", alpha=0.6, label="갭 0.10")

    for bar, val in zip(bars_gap, cv_df["gap"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#333333", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in test_seasons], fontsize=10)
    ax2.set_ylim(0, cv_df["gap"].max() * 1.4)
    ax2.set_ylabel("Train AUC − Test AUC", fontsize=11, color="#444444")
    ax2.set_title("과적합 갭 (폴드별)", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
    ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
    _soft_axes(ax2)

    fig.suptitle("Strategy C 과적합 분석  (LR+RF+lightXGB+lightLGBM | 20 피처)", fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
    plt.tight_layout()
    out_path = _save_fig(Path(valdir) / "val_overfit_gap.png")
    print(f"[V2] 과적합 갭 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 검증 V3: 로스 커브
# ─────────────────────────────────────────────
def save_loss_curves_chart(xgb_losses, lgbm_losses, valdir: str | Path, test_seasons=TEST_SEASONS) -> str:
    """
    XGBoost와 LightGBM의 fold별 train/valid logloss 곡선을 저장한다.

    `run_loso_cv(..., include_losses=True)`에서 수집한 eval history를 사용한다.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)
    fold_cmap = mcm.get_cmap("tab10")

    for i, season in enumerate(test_seasons):
        # 마지막 fold는 굵게 표시해 최신 시즌 검증 흐름을 더 잘 보이게 한다.
        color = fold_cmap(i)
        lw = 2.2 if season == test_seasons[-1] else 1.0
        alpha = 1.0 if season == test_seasons[-1] else 0.40

        xr = xgb_losses[season]
        tr_key = list(xr.keys())[0]
        va_key = list(xr.keys())[1]
        rds = range(1, len(xr[tr_key]["logloss"]) + 1)
        ax1.plot(rds, xr[tr_key]["logloss"], color=color, lw=lw, alpha=alpha, ls="-")
        ax1.plot(rds, xr[va_key]["logloss"], color=color, lw=lw, alpha=alpha, ls="--")

        lr_ = lgbm_losses[season]
        tr_l = lr_["train"]["binary_logloss"]
        va_l = lr_["valid"]["binary_logloss"]
        rds_l = range(1, len(tr_l) + 1)
        ax2.plot(rds_l, tr_l, color=color, lw=lw, alpha=alpha, ls="-")
        ax2.plot(rds_l, va_l, color=color, lw=lw, alpha=alpha, ls="--")

    legend_els = (
        [Line2D([0], [0], color=fold_cmap(i), lw=1.5, label=str(s)) for i, s in enumerate(test_seasons)]
        + [Line2D([0], [0], color="#555", lw=1.5, ls="-", label="Train"), Line2D([0], [0], color="#555", lw=1.5, ls="--", label="Valid")]
    )
    for ax, title in [(ax1, "XGBoost 로스 커브"), (ax2, "LightGBM 로스 커브")]:
        ax.set_xlabel("부스팅 라운드", fontsize=11, color="#444444")
        ax.set_ylabel("Log Loss", fontsize=11, color="#444444")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
        _soft_axes(ax)

    ax2.legend(handles=legend_els, fontsize=8.5, loc="upper right", framealpha=0.9, edgecolor="#DDDDDD", ncol=2)
    fig.suptitle(
        f"LOSO-CV 로스 커브  (각 선 = 1 폴드 | 굵은 선 = {test_seasons[-1]} 폴드 | 실선=Train / 점선=Valid)",
        fontsize=13,
        fontweight="bold",
        color="#1B1B1B",
        y=1.02,
    )
    plt.tight_layout()
    out_path = _save_fig(Path(valdir) / "val_loss_curves.png")
    print(f"[V3] 로스 커브 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 검증 V4: 캘리브레이션 + 확률 분포
# ─────────────────────────────────────────────
def save_calibration_chart(all_rows, valdir: str | Path) -> str:
    """
    전체 OOF 예측의 확률 보정 상태와 클래스별 확률 분포를 저장한다.

    reliability diagram은 예측 확률이 실제 양성 비율과 얼마나 맞는지 확인하는 용도다.
    """
    prob_all = all_rows["prob"].values
    actual_all = all_rows["postseason"].values
    frac_pos, mean_pred = calibration_curve(actual_all, prob_all, n_bins=8, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="완벽 캘리브레이션")
    ax1.plot(mean_pred, frac_pos, "o-", color=ACCENT, lw=2.0, ms=8, label="Strategy C")
    for xv, yv in zip(mean_pred, frac_pos):
        diff = yv - xv
        color = GREEN if abs(diff) < 0.05 else WARM_RED
        ax1.annotate(f"{diff:+.2f}", (xv, yv), xytext=(xv + 0.01, yv + 0.025), fontsize=8, color=color, fontweight="bold")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("예측 확률", fontsize=11, color="#444444")
    ax1.set_ylabel("실제 양성 비율", fontsize=11, color="#444444")
    ax1.set_title("Reliability Diagram", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
    ax1.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _soft_axes(ax1, grid_axis="both")

    bins = np.linspace(0, 1, 25)
    ax2.hist(prob_all[actual_all == 0], bins=bins, color=WARM_RED, alpha=0.65, label="비진출 (0)", density=True)
    ax2.hist(prob_all[actual_all == 1], bins=bins, color=GREEN, alpha=0.65, label="진출 (1)", density=True)
    ax2.axvline(0.5, color="#555555", lw=1.2, ls="--", alpha=0.7)
    ax2.set_xlabel("예측 확률", fontsize=11, color="#444444")
    ax2.set_ylabel("밀도", fontsize=11, color="#444444")
    ax2.set_title("예측 확률 분포 (클래스별)", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
    ax2.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _soft_axes(ax2)

    fig.suptitle("예측 확률 캘리브레이션 및 분포 (LOSO-CV 전체)", fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
    plt.tight_layout()
    out_path = _save_fig(Path(valdir) / "val_calibration.png")
    print(f"[V4] 캘리브레이션 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 검증 V5: 체크포인트 적중률
# ─────────────────────────────────────────────
def save_checkpoint_chart(all_rows, actual_top5, valdir: str | Path, test_seasons=TEST_SEASONS) -> str:
    """
    시즌 진행률 50%, 75%, 90%, 최종 시점에서 예측 top5 적중 수를 저장한다.

    각 checkpoint별로 팀별 최신 row를 잡고, 예측 상위 5팀이 실제 postseason 팀과 얼마나 겹치는지 계산한다.
    """
    checkpoints = {"50% 시점": 0.50, "75% 시점": 0.75, "90% 시점": 0.90, "최종 시점": 1.01}
    cp_results = {}
    for season in test_seasons:
        if season not in actual_top5:
            continue
        season_rows = all_rows[all_rows["season"] == season]
        cp_results[season] = checkpoint_hits(season_rows, "prob", actual_top5[season], checkpoints)

    cp_labels = list(checkpoints.keys())
    cp_hit_avg = {lbl: [cp_results[s][lbl]["hit"] for s in test_seasons if s in cp_results] for lbl in cp_labels}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    cp_means = [np.mean(cp_hit_avg[lbl]) if cp_hit_avg[lbl] else 0 for lbl in cp_labels]
    cp_stds = [np.std(cp_hit_avg[lbl]) if cp_hit_avg[lbl] else 0 for lbl in cp_labels]
    cp_colors = [GREEN if m >= 4 else (ORANGE if m >= 3 else WARM_RED) for m in cp_means]
    bars_cp = ax1.bar(cp_labels, cp_means, color=cp_colors, alpha=0.85, width=0.5, edgecolor="white")
    ax1.errorbar(cp_labels, cp_means, yerr=cp_stds, fmt="none", capsize=5, ecolor="#555555", elinewidth=1.5)
    ax1.axhline(3, color=ORANGE, lw=1.2, ls="--", alpha=0.7, label="3/5 기준")
    ax1.axhline(4, color=GREEN, lw=1.2, ls="--", alpha=0.7, label="4/5 기준")
    for bar, mean in zip(bars_cp, cp_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, mean + 0.06, f"{mean:.2f}/5", ha="center", va="bottom", fontsize=10, color="#333333", fontweight="bold")
    ax1.set_ylim(0, 5.8)
    ax1.set_ylabel("평균 적중 팀 수 (/5)", fontsize=11, color="#444444")
    ax1.set_title("시점별 평균 적중 수", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
    ax1.legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
    _soft_axes(ax1)

    valid_seasons = [s for s in test_seasons if s in cp_results]
    final_hits = [cp_results[s]["최종 시점"]["hit"] for s in valid_seasons]
    hit_colors = [GREEN if h >= 4 else (ORANGE if h == 3 else WARM_RED) for h in final_hits]
    bars_fin = ax2.bar(range(len(valid_seasons)), final_hits, color=hit_colors, alpha=0.85, width=0.55, edgecolor="white")
    for bar, val in zip(bars_fin, final_hits):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"{val}/5", ha="center", va="bottom", fontsize=10, color="#333333", fontweight="bold")
    ax2.set_xticks(range(len(valid_seasons)))
    ax2.set_xticklabels([str(s) for s in valid_seasons], fontsize=10)
    ax2.set_ylim(0, 5.8)
    ax2.set_ylabel("적중 팀 수 (/5)", fontsize=11, color="#444444")
    ax2.set_title("시즌별 최종 시점 적중 수", fontsize=12, fontweight="bold", color="#1B1B1B", pad=14)
    _soft_axes(ax2)

    fig.suptitle("체크포인트 적중률 — 포스트시즌 상위 5팀 예측 (LOSO-CV)", fontsize=14, fontweight="bold", color="#1B1B1B", y=1.02)
    plt.tight_layout()
    out_path = _save_fig(Path(valdir) / "val_checkpoint.png")
    print(f"[V5] 체크포인트 저장: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 검증 차트 일괄 저장
# ─────────────────────────────────────────────
def save_validation_charts(
    cv_df,
    all_rows,
    xgb_losses,
    lgbm_losses,
    actual_top5,
    valdir: str | Path,
) -> dict[str, str]:
    """검증 차트 V1~V5를 모두 저장하고 산출물 경로를 반환한다."""
    paths = {
        "scorecard": save_scorecard_chart(cv_df, valdir),
        "overfit_gap": save_overfit_gap_chart(cv_df, valdir),
        "loss_curves": save_loss_curves_chart(xgb_losses, lgbm_losses, valdir),
        "calibration": save_calibration_chart(all_rows, valdir),
        "checkpoint": save_checkpoint_chart(all_rows, actual_top5, valdir),
    }
    print("\n검증 완료.")
    print(f"  V1 스코어카드    : {paths['scorecard']}")
    print(f"  V2 과적합 갭     : {paths['overfit_gap']}")
    print(f"  V3 로스 커브     : {paths['loss_curves']}")
    print(f"  V4 캘리브레이션  : {paths['calibration']}")
    print(f"  V5 체크포인트    : {paths['checkpoint']}\n")
    return paths
