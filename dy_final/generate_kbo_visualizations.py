# -*- coding: utf-8 -*-
"""Generate PNG visualization assets for the KBO postseason project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "data"
OUT_DIR = ROOT_DIR / "kbo_outputs"
YEARS = [2022, 2023, 2024, 2025, 2026]


def setup_plot() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        fonts = [f.fname for f in fm.fontManager.ttflist if "Nanum" in f.name]
        if fonts:
            plt.rcParams["font.family"] = fm.FontProperties(fname=fonts[0]).get_name()


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV read failed: {path}")


def load_stats() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = DATA_DIR / "processed" / str(year) / "team_stats_clean.csv"
        if path.exists():
            frames.append(read_csv(path))
    stats = pd.concat(frames, ignore_index=True)

    snap_path = OUT_DIR / "2026_team_snapshot.csv"
    if snap_path.exists():
        snap = read_csv(snap_path)
        current = pd.DataFrame(
            {
                "season": snap["season"],
                "team": snap["team"],
                "team_avg": snap.get("hit_AVG"),
                "team_runs": snap.get("hit_R"),
                "team_hr": snap.get("hit_HR"),
                "team_rbi": snap.get("hit_RBI"),
                "team_era": snap.get("pit_ERA"),
                "team_whip": snap.get("pit_WHIP"),
                "team_runs_allowed": snap.get("pit_R"),
                "team_so_pitcher": snap.get("pit_SO"),
                "team_bb_allowed": snap.get("pit_BB"),
                "team_ip": snap.get("pit_IP"),
                "team_errors": snap.get("def_E"),
                "team_fpct": snap.get("def_FPCT"),
                "team_sb": snap.get("run_SB"),
                "team_sb_rate": snap.get("run_SB%"),
                "run_differential": snap.get("hit_R") - snap.get("pit_R"),
                "iso": (snap.get("hit_TB") / snap.get("hit_AB")) - snap.get("hit_AVG"),
            }
        )
        current["team_ops"] = np.nan
        current["k_bb_ratio"] = current["team_so_pitcher"] / current["team_bb_allowed"].replace(0, np.nan)
        stats = stats[stats["season"] != 2026]
        stats = pd.concat([stats, current], ignore_index=True, sort=False)
    return stats


def load_ranks() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = DATA_DIR / "processed" / str(year) / "team_final_rank_clean.csv"
        if path.exists():
            frames.append(read_csv(path))
    ranks = pd.concat(frames, ignore_index=True)

    snap_path = OUT_DIR / "2026_team_snapshot.csv"
    if snap_path.exists():
        snap = read_csv(snap_path)
        cur = snap[["season", "team", "rank"]].rename(columns={"rank": "final_rank"}).copy()
        cur["postseason"] = np.nan
        ranks = ranks[ranks["season"] != 2026]
        ranks = pd.concat([ranks, cur], ignore_index=True, sort=False)
    return ranks


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(ROOT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def plot_01_league_trend(stats: pd.DataFrame) -> None:
    league = stats.groupby("season").agg(
        avg=("team_avg", "mean"),
        era=("team_era", "mean"),
        ops=("team_ops", "mean"),
        hr=("team_hr", "sum"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    items = [("avg", "리그 평균 타율", "#1D9E75"), ("era", "리그 평균 ERA", "#D85A30"), ("hr", "리그 홈런 합계", "#534AB7")]
    for ax, (col, title, color) in zip(axes, items):
        ax.plot(league["season"], league[col], marker="o", color=color, linewidth=2.5)
        for x, y in zip(league["season"], league[col]):
            label = f"{y:.3f}" if col == "avg" else f"{y:.1f}"
            ax.text(x, y, label, ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("KBO 리그 환경 5년 트렌드", fontweight="bold")
    save_fig("01_league_trend.png")


def plot_02_rank_heatmap(ranks: pd.DataFrame) -> None:
    pivot = ranks.pivot_table(index="team", columns="season", values="final_rank", aggfunc="first")
    pivot = pivot[[c for c in YEARS if c in pivot.columns]]
    pivot = pivot.sort_values(2026 if 2026 in pivot.columns else pivot.columns[-1])
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", vmin=1, vmax=10, linewidths=.5, cbar_kws={"label": "순위"})
    plt.title("팀별 연도별 순위 변화 (2022~2026)", fontweight="bold")
    plt.xlabel("")
    plt.ylabel("")
    save_fig("02_rank_heatmap.png")


def plot_03_era_winrate(stats: pd.DataFrame, ranks: pd.DataFrame) -> None:
    df = stats.merge(ranks[["season", "team", "final_rank", "postseason"]], on=["season", "team"], how="left")
    final = []
    for year in YEARS:
        p = DATA_DIR / "raw" / str(year) / "team_final_rank.csv"
        if p.exists():
            raw = read_csv(p).rename(columns={"팀명": "team", "승률": "win_rate"})
            raw["season"] = year
            final.append(raw[["season", "team", "win_rate"]])
    if final:
        df = df.merge(pd.concat(final, ignore_index=True), on=["season", "team"], how="left")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df[df["season"] < 2026], x="team_era", y="win_rate", hue="season", size="postseason", sizes=(60, 140), palette="viridis")
    plt.title("팀 ERA와 승률 관계 (2022~2025)", fontweight="bold")
    plt.xlabel("팀 ERA")
    plt.ylabel("승률")
    save_fig("03_era_winrate.png")


def plot_04_playoff_boxplot(stats: pd.DataFrame, ranks: pd.DataFrame) -> None:
    df = stats.merge(ranks[["season", "team", "postseason"]], on=["season", "team"], how="left")
    df = df[(df["season"] < 2026) & df["postseason"].notna()].copy()
    df["가을야구"] = df["postseason"].map({1: "진출", 0: "미진출"})
    cols = [("team_era", "ERA"), ("team_ops", "OPS"), ("run_differential", "득실차")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (col, title) in zip(axes, cols):
        sns.boxplot(data=df, x="가을야구", y=col, ax=ax, palette={"진출": "#1D9E75", "미진출": "#D85A30"})
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("가을야구 진출/미진출 팀 지표 비교", fontweight="bold")
    save_fig("04_playoff_boxplot.png")


def plot_05_transfer_placeholder() -> None:
    movement = DATA_DIR / "2026" / "2026_선수_이동_현황.csv"
    fig, ax = plt.subplots(figsize=(8, 4))
    if movement.exists():
        df = read_csv(movement)
        if not df.empty and "항목" in df.columns:
            cnt = df["항목"].value_counts().head(10)
            cnt.sort_values().plot(kind="barh", ax=ax, color="#534AB7")
            ax.set_title("2026 선수 이동 현황", fontweight="bold")
            ax.set_xlabel("건수")
        else:
            ax.text(.5, .5, "현재 선수 이동 현황 데이터가 비어 있습니다.", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
    else:
        ax.text(.5, .5, "선수 이동 현황 CSV가 없습니다.", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    save_fig("05_transfer.png")


def plot_06_feature_importance() -> None:
    df = read_csv(OUT_DIR / "feature_importance_coefficients.csv").head(12).copy()
    df = df.sort_values("abs_coefficient")
    plt.figure(figsize=(9, 5))
    colors = np.where(df["coefficient"] >= 0, "#1D9E75", "#D85A30")
    plt.barh(df["feature_ko"], df["abs_coefficient"], color=colors)
    plt.title("가을야구 예측 피처 영향도", fontweight="bold")
    plt.xlabel("절대 계수")
    save_fig("06_feature_importance.png")


def plot_07_playoff_prob() -> None:
    pred = read_csv(OUT_DIR / "2026_postseason_predictions.csv").sort_values("postseason_probability_pct", ascending=False)
    plt.figure(figsize=(10, 5))
    colors = np.where(pred["prediction_label"] == "진출", "#1D9E75", "#D85A30")
    bars = plt.bar(pred["team"], pred["postseason_probability_pct"], color=colors)
    plt.axhline(50, color="gray", linestyle="--", linewidth=1)
    for bar, value in zip(bars, pred["postseason_probability_pct"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    plt.ylim(0, 105)
    plt.title("2026 가을야구 진출 확률", fontweight="bold")
    plt.ylabel("확률 (%)")
    save_fig("07_playoff_prob.png")


def plot_08_april_vs_final() -> None:
    april = read_csv(OUT_DIR / "april_rank_insight.csv")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(april["season"].astype(str), april["current_top5_overlap_rate"] * 100, color="#185FA5", alpha=.85, label="4월 Top5 적중률")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("적중률 (%)")
    ax2 = ax1.twinx()
    ax2.plot(april["season"].astype(str), april["rank_final_spearman"], color="#D85A30", marker="o", linewidth=2, label="순위 상관")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Spearman 상관")
    ax1.set_title("4월 순위의 최종 순위 예측력", fontweight="bold")
    save_fig("08_apr_vs_final.png")


def plot_09_cluster_pca(stats: pd.DataFrame, ranks: pd.DataFrame) -> None:
    df = stats.merge(ranks[["season", "team", "final_rank"]], on=["season", "team"], how="left")
    features = ["team_avg", "team_era", "team_whip", "team_hr", "run_differential", "team_fpct"]
    base = df[df["season"] < 2026].dropna(subset=features).copy()
    x = StandardScaler().fit_transform(base[features])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    base["cluster"] = km.fit_predict(x)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)
    base["PC1"] = coords[:, 0]
    base["PC2"] = coords[:, 1]
    order = base.groupby("cluster")["final_rank"].mean().sort_values().index.tolist()
    labels = {order[0]: "강팀형", order[-1]: "약팀형"}
    for c in base["cluster"].unique():
        labels.setdefault(c, "중위권형")
    base["팀유형"] = base["cluster"].map(labels)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=base, x="PC1", y="PC2", hue="팀유형", style="season", s=90, palette={"강팀형": "#1D9E75", "중위권형": "#534AB7", "약팀형": "#D85A30"})
    plt.title("팀 유형 클러스터링 PCA (2022~2025)", fontweight="bold")
    save_fig("09_cluster_pca.png")


def main() -> None:
    setup_plot()
    stats = load_stats()
    ranks = load_ranks()
    plot_01_league_trend(stats)
    plot_02_rank_heatmap(ranks)
    plot_03_era_winrate(stats, ranks)
    plot_04_playoff_boxplot(stats, ranks)
    plot_05_transfer_placeholder()
    plot_06_feature_importance()
    plot_07_playoff_prob()
    plot_08_april_vs_final()
    plot_09_cluster_pca(stats, ranks)
    print(f"Generated visualization PNG files in {ROOT_DIR}")


if __name__ == "__main__":
    main()
