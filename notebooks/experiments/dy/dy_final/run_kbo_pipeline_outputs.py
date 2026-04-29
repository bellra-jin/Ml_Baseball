# -*- coding: utf-8 -*-
"""Refresh pipeline outputs and dashboard-compatible CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import notebooks.experiments.dy.dy_final.kbo_postseason_pipeline as pipe
import notebooks.experiments.dy.dy_final.generate_kbo_visualizations as viz

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parent
DEFAULT_DATA_DIR = REPO_DIR / "data"
DEFAULT_OUT_DIR = ROOT_DIR / "kbo_outputs"


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV read failed: {path}")


def raw_rank_rows(data_dir: Path, years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        candidates = [
            data_dir / "raw" / str(year) / "team_final_rank.csv",
            data_dir / str(year) / "팀_순위.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        df = read_csv(path).rename(
            columns={
                "팀명": "team",
                "순위": "rank",
                "경기": "games",
                "승": "wins",
                "패": "losses",
                "승률": "win_rate",
            }
        )
        df["season"] = year
        frames.append(df[["season", "team", "rank", "games", "wins", "losses", "win_rate"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export_dashboard_csvs(data_dir: Path, out_dir: Path) -> None:
    pred = read_csv(out_dir / "2026_postseason_predictions.csv")
    snap = read_csv(out_dir / "2026_team_snapshot.csv")
    stats = viz.load_stats()
    ranks = viz.load_ranks()

    pred_ko = pred.merge(
        snap[["team", "current_era", "current_whip"]],
        on="team",
        how="left",
    )
    pred_ko = pred_ko.rename(
        columns={
            "team": "팀명",
            "rank": "순위",
            "wins": "승",
            "losses": "패",
            "win_rate": "승률",
            "current_era": "ERA",
            "current_whip": "WHIP",
            "postseason_probability_pct": "가을야구_확률(%)",
            "prediction_label": "예측",
        }
    )
    pred_ko["OPS"] = pd.NA
    pred_ko["예측"] = pred_ko["예측"].map({"진출": "✅ 진출", "미진출": "❌ 미진출"}).fillna(pred_ko["예측"])
    pred_ko[["팀명", "순위", "승", "패", "승률", "ERA", "OPS", "WHIP", "가을야구_확률(%)", "예측"]].to_csv(
        ROOT_DIR / "2026_가을야구_예측결과.csv",
        index=True,
        encoding="utf-8-sig",
    )

    rank_rows = raw_rank_rows(data_dir, [2022, 2023, 2024, 2025])
    current = snap.rename(
        columns={
            "season": "season",
            "team": "team",
            "rank": "rank",
            "games": "games",
            "wins": "wins",
            "losses": "losses",
            "win_rate": "win_rate",
        }
    )[["season", "team", "rank", "games", "wins", "losses", "win_rate"]]
    rank_rows = pd.concat([rank_rows, current], ignore_index=True, sort=False)

    master = rank_rows.merge(stats, on=["season", "team"], how="left")
    master = master.rename(
        columns={
            "season": "연도",
            "team": "팀명",
            "rank": "순위",
            "games": "경기",
            "wins": "승",
            "losses": "패",
            "win_rate": "승률",
            "team_era": "pit_ERA",
            "team_ops": "bat_OPS",
            "team_whip": "pit_WHIP",
            "team_runs": "bat_R",
            "team_runs_allowed": "pit_R",
        }
    )
    master.to_csv(ROOT_DIR / "team_master_2022_2026.csv", index=False, encoding="utf-8-sig")

    league_bat = stats.groupby("season").agg(
        리그_AVG=("team_avg", "mean"),
        리그_OPS=("team_ops", "mean"),
        리그_HR평균=("team_hr", "mean"),
    ).reset_index().rename(columns={"season": "연도"})
    league_bat.to_csv(ROOT_DIR / "리그_타격환경.csv", index=False, encoding="utf-8-sig")

    def k9(group: pd.DataFrame) -> float:
        so = pd.to_numeric(group["team_so_pitcher"], errors="coerce").sum()
        ip = pd.to_numeric(group["team_ip"], errors="coerce").sum()
        return (so / ip * 9) if ip else float("nan")

    league_pit = stats.groupby("season").agg(
        리그_ERA=("team_era", "mean"),
        리그_WHIP=("team_whip", "mean"),
        규정투수수=("team", "count"),
    ).reset_index().rename(columns={"season": "연도"})
    league_pit["리그_K9"] = stats.groupby("season").apply(k9, include_groups=False).values
    league_pit.to_csv(ROOT_DIR / "리그_투구환경.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    artifacts = pipe.run_pipeline(args.data_dir, args.out_dir)
    export_dashboard_csvs(args.data_dir, args.out_dir)
    print(artifacts["summary"])


if __name__ == "__main__":
    main()
