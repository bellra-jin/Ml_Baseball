
# -*- coding: utf-8 -*-
"""DY experiment pipeline built from data/raw and data/processed.

Outputs are written next to this file under notebooks/experiments/dy.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_DIR = EXPERIMENT_DIR.parents[2]
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DY_FINAL_DIR = EXPERIMENT_DIR / "dy_final"
if not DY_FINAL_DIR.exists():
    DY_FINAL_DIR = REPO_DIR / "dy_final"
OUT_DIR = EXPERIMENT_DIR / "kbo_outputs"
YEARS = [2022, 2023, 2024, 2025, 2026]
TRAIN_YEARS = [2022, 2023, 2024, 2025]

FEATURES = [
    "rank",
    "win_rate",
    "games_behind",
    "recent10_win_rate",
    "home_win_rate",
    "away_win_rate",
    "games_played_ratio",
    "games_behind_5th",
    "wins_to_5th",
    "prev_team_era",
    "prev_team_whip",
    "prev_run_differential",
    "prev_pythagorean_win_rate",
    "prev_k_bb_ratio",
    "prev_team_ops",
    "prev_top5_hitter_ops_avg",
    "avg3yr_run_differential",
    "trend_run_differential",
    "avg3yr_team_era",
    "trend_team_era",
]

FEATURE_KO = {
    "rank": "현재 순위",
    "win_rate": "현재 승률",
    "games_behind": "현재 게임차",
    "recent10_win_rate": "최근 10경기 승률",
    "home_win_rate": "홈 승률",
    "away_win_rate": "원정 승률",
    "games_played_ratio": "시즌 진행률",
    "games_behind_5th": "5위와 게임차",
    "wins_to_5th": "5위 추격 필요승",
    "prev_team_era": "전년도 ERA",
    "prev_team_whip": "전년도 WHIP",
    "prev_run_differential": "전년도 득실차",
    "prev_pythagorean_win_rate": "전년도 피타고리안 승률",
    "prev_k_bb_ratio": "전년도 K/BB",
    "prev_team_ops": "전년도 OPS",
    "prev_top5_hitter_ops_avg": "전년도 상위5타자 OPS",
    "avg3yr_run_differential": "최근3년 평균 득실차",
    "trend_run_differential": "득실차 추세",
    "avg3yr_team_era": "최근3년 평균 ERA",
    "trend_team_era": "ERA 추세",
}


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV read failed: {path}")


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))


def numeric_frame(df: pd.DataFrame, features: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame(index=df.index)
    for feature in features:
        x[feature] = pd.to_numeric(df[feature], errors="coerce") if feature in df.columns else np.nan
    if medians is None:
        medians = x.median(numeric_only=True).fillna(0)
    x = x.fillna(medians).fillna(0)
    return x, medians


def fit_logistic(train: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    y = pd.to_numeric(train["postseason"], errors="coerce").fillna(0).to_numpy(dtype=float)
    x_df, medians = numeric_frame(train, features)
    mean = x_df.mean()
    std = x_df.std(ddof=0).replace(0, 1)
    x = ((x_df - mean) / std).to_numpy(dtype=float)
    w = np.zeros(x.shape[1], dtype=float)
    p0 = min(max(float(y.mean()), 1e-4), 1 - 1e-4)
    b = math.log(p0 / (1 - p0))
    lr = 0.04
    l2 = 0.01
    for _ in range(2500):
        p = sigmoid(x @ w + b)
        grad_w = (x.T @ (p - y)) / len(y) + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return {"features": features, "medians": medians, "mean": mean, "std": std, "w": w, "b": b}


def predict_logistic(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x_df, _ = numeric_frame(df, model["features"], model["medians"])
    x = ((x_df - model["mean"]) / model["std"]).to_numpy(dtype=float)
    return sigmoid(x @ model["w"] + model["b"])


def standing_probability(df: pd.DataFrame) -> np.ndarray:
    rank = pd.to_numeric(df.get("rank"), errors="coerce").fillna(10).to_numpy(float)
    win_rate = pd.to_numeric(df.get("win_rate"), errors="coerce").fillna(0.5).to_numpy(float)
    games_behind = pd.to_numeric(df.get("games_behind"), errors="coerce").fillna(10).to_numpy(float)
    recent = pd.to_numeric(df.get("recent10_win_rate"), errors="coerce").fillna(win_rate.mean()).to_numpy(float)
    z = 1.0 * (5.5 - rank) + 4.0 * (win_rate - 0.5) + 1.6 * (recent - 0.5) - 0.16 * games_behind
    return sigmoid(z)


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values(["team", "date"])
    latest = data.groupby("team", as_index=False).tail(1).sort_values(["rank", "win_rate"], ascending=[True, False])
    return latest.reset_index(drop=True)


def load_processed_model_data() -> pd.DataFrame:
    frames = []
    for year in TRAIN_YEARS:
        path = PROCESSED_DIR / str(year) / f"train_dataset_{year}.csv"
        if path.exists():
            df = read_csv(path).copy()
            df["source"] = f"processed/train_dataset_{year}"
            frames.append(df)
    daily_2026 = read_csv(PROCESSED_DIR / "2026" / "team_daily_rank_clean.csv")
    prev_2026 = read_csv(PROCESSED_DIR / "2026" / "prev_features_from_2025.csv")
    multi_2026 = read_csv(PROCESSED_DIR / "2026" / "multi_year_features_2026.csv")
    data_2026 = daily_2026.merge(prev_2026, on=["season", "team"], how="left")
    data_2026 = data_2026.merge(multi_2026, on=["season", "team"], how="left").copy()
    data_2026["final_rank"] = np.nan
    data_2026["postseason"] = np.nan
    data_2026["source"] = "processed/2026_daily_plus_prev_features"
    frames.append(data_2026)
    model_df = pd.concat(frames, ignore_index=True, sort=False)
    return model_df


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 2:
        return float("nan")
    lrank = pair["left"].rank(method="average")
    rrank = pair["right"].rank(method="average")
    return float(lrank.corr(rrank, method="pearson"))


def build_validation(model_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in TRAIN_YEARS:
        train = model_df[(model_df["season"] != season) & (model_df["season"].isin(TRAIN_YEARS))]
        train = train[(train["postseason"].notna()) & (pd.to_numeric(train["games"], errors="coerce") >= 10)]
        test = model_df[(model_df["season"] == season) & (model_df["postseason"].notna())].copy()
        test = test[pd.to_numeric(test["games"], errors="coerce") >= 10]
        if train.empty or test.empty:
            continue
        model = fit_logistic(train, FEATURES)
        test["prob"] = predict_logistic(model, test)
        latest = latest_snapshot(test)
        top5 = set(latest.sort_values("prob", ascending=False).head(5)["team"])
        actual = set(latest[latest["postseason"] == 1]["team"])
        rows.append(
            {
                "season": season,
                "rows": len(test),
                "auc": round(auc_score(test["postseason"].to_numpy(), test["prob"].to_numpy()), 3),
                "latest_top5_overlap": len(top5 & actual),
                "latest_top5_overlap_rate": round(len(top5 & actual) / 5, 3),
            }
        )
    return pd.DataFrame(rows)


def build_april_insight() -> pd.DataFrame:
    rows = []
    for season in [2023, 2024, 2025]:
        daily = read_csv(PROCESSED_DIR / str(season) / "team_daily_rank_clean.csv")
        final = read_csv(PROCESSED_DIR / str(season) / "team_final_rank_clean.csv")
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        cutoff = pd.Timestamp(year=season, month=4, day=30)
        april = daily[daily["date"] <= cutoff]
        if april.empty:
            continue
        latest_date = april["date"].max()
        snap = april[april["date"] == latest_date]
        cur_top5 = set(snap.sort_values("rank").head(5)["team"])
        final_top5 = set(final.sort_values("final_rank").head(5)["team"])
        merged = snap[["team", "rank"]].merge(final[["team", "final_rank"]], on="team", how="inner")
        rows.append(
            {
                "season": season,
                "april_latest_date": latest_date.date().isoformat(),
                "current_top5_overlap": len(cur_top5 & final_top5),
                "current_top5_overlap_rate": round(len(cur_top5 & final_top5) / 5, 3),
                "rank_final_spearman": round(spearman_corr(merged["rank"], merged["final_rank"]), 3),
            }
        )
    return pd.DataFrame(rows)


def build_team_master() -> pd.DataFrame:
    rank_frames = []
    for year in YEARS:
        rank_path = RAW_DIR / str(year) / "team_final_rank.csv"
        if not rank_path.exists():
            continue
        r = read_csv(rank_path)
        r = r.rename(columns={"순위": "순위", "팀명": "팀명", "경기": "경기", "승": "승", "패": "패", "승률": "승률"})
        keep = [c for c in ["순위", "팀명", "경기", "승", "패", "승률"] if c in r.columns]
        r = r[keep].copy()
        r.insert(0, "연도", year)
        rank_frames.append(r)
    ranks = pd.concat(rank_frames, ignore_index=True)

    stat_frames = []
    for year in YEARS:
        stat_path = PROCESSED_DIR / str(year) / "team_stats_clean.csv"
        if not stat_path.exists():
            continue
        s = read_csv(stat_path).rename(columns={"season": "연도", "team": "팀명"})
        stat_frames.append(s)
    stats = pd.concat(stat_frames, ignore_index=True)
    master = ranks.merge(stats, on=["연도", "팀명"], how="left")
    alias_map = {
        "team_era": "pit_ERA",
        "team_whip": "pit_WHIP",
        "team_ops": "bat_OPS",
        "team_runs": "bat_R",
        "team_hits": "team_hits",
        "team_hr": "team_hr",
        "team_rbi": "team_rbi",
    }
    for src, dst in alias_map.items():
        if src in master.columns and dst not in master.columns:
            master[dst] = master[src]
    master = master.sort_values(["연도", "순위", "팀명"]).reset_index(drop=True)
    return master


def export_dashboard_csvs(pred: pd.DataFrame, team_master: pd.DataFrame) -> None:
    stats_2026 = team_master[team_master["연도"] == 2026][["팀명", "pit_ERA", "bat_OPS", "pit_WHIP"]].copy()
    pred_ko = pred.merge(stats_2026, left_on="team", right_on="팀명", how="left")
    pred_ko = pd.DataFrame(
        {
            "팀명": pred_ko["team"],
            "순위": pred_ko["rank"],
            "승": pred_ko["wins"],
            "패": pred_ko["losses"],
            "승률": pred_ko["win_rate"],
            "ERA": pred_ko["pit_ERA"],
            "OPS": pred_ko["bat_OPS"],
            "WHIP": pred_ko["pit_WHIP"],
            "가을야구_확률(%)": pred_ko["postseason_probability_pct"],
            "예측": pred_ko["prediction_label"],
        }
    )
    save_csv(pred_ko, EXPERIMENT_DIR / "2026_가을야구_예측결과.csv")
    save_csv(team_master, EXPERIMENT_DIR / "team_master_2022_2026.csv")
    league_bat = team_master.groupby("연도", as_index=False).agg(
        리그_AVG=("team_avg", "mean"),
        리그_OPS=("bat_OPS", "mean"),
        리그_HR평균=("team_hr", "mean"),
    )
    league_pit = team_master.groupby("연도", as_index=False).agg(
        리그_ERA=("pit_ERA", "mean"),
        리그_WHIP=("pit_WHIP", "mean"),
        규정투수수=("팀명", "count"),
        리그_K9=("k_bb_ratio", "mean"),
    )
    save_csv(league_bat, EXPERIMENT_DIR / "리그_타격환경.csv")
    save_csv(league_pit, EXPERIMENT_DIR / "리그_투구환경.csv")


def generate_pngs_and_html() -> None:
    if str(DY_FINAL_DIR) not in sys.path:
        sys.path.insert(0, str(DY_FINAL_DIR))
    import generate_kbo_visualizations_pretty as viz

    viz.ROOT_DIR = EXPERIMENT_DIR
    viz.REPO_DIR = REPO_DIR
    viz.DATA_DIR = DATA_DIR
    viz.OUT_DIR = OUT_DIR
    viz.main()

    import generate_kbo_master_dashboard as dash

    dash.ROOT_DIR = EXPERIMENT_DIR
    dash.DATA_DIR = EXPERIMENT_DIR / "data_"
    dash.OUT_HTML = EXPERIMENT_DIR / "kbo_2022_2026_master_dashboard.html"
    dash.main()

    import dy_chart_dashboard as chart_dash

    chart_dash.render_dashboard()


def run_all() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_df = load_processed_model_data()
    save_csv(model_df, OUT_DIR / "model_dataset_2022_2026.csv")

    train = model_df[(model_df["season"].isin(TRAIN_YEARS)) & (model_df["postseason"].notna())].copy()
    train = train[pd.to_numeric(train["games"], errors="coerce") >= 10]
    model = fit_logistic(train, FEATURES)

    latest_2026 = latest_snapshot(model_df[model_df["season"] == 2026].copy())
    pred = latest_2026[["season", "date", "team", "rank", "games", "wins", "losses", "draws", "win_rate", "games_behind"]].copy()
    pred["model_probability"] = predict_logistic(model, latest_2026)
    pred["standing_probability"] = standing_probability(latest_2026)
    pred["postseason_probability"] = 0.55 * pred["model_probability"] + 0.45 * pred["standing_probability"]
    for col in ["model_probability", "standing_probability", "postseason_probability"]:
        pred[col + "_pct"] = (pred[col] * 100).round(1)
    pred = pred.sort_values("postseason_probability", ascending=False).reset_index(drop=True)
    pred["predicted_top5"] = 0
    pred.loc[:4, "predicted_top5"] = 1
    pred["prediction_label"] = np.where(pred["predicted_top5"] == 1, "진출", "미진출")
    save_csv(pred, OUT_DIR / "2026_postseason_predictions.csv")
    save_csv(latest_2026, OUT_DIR / "2026_team_snapshot.csv")

    fi = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_ko": [FEATURE_KO.get(f, f) for f in FEATURES],
            "coefficient": model["w"],
            "abs_coefficient": np.abs(model["w"]),
        }
    ).sort_values("abs_coefficient", ascending=False)
    fi["direction"] = np.where(fi["coefficient"] >= 0, "positive", "negative")
    save_csv(fi, OUT_DIR / "feature_importance_coefficients.csv")

    validation = build_validation(model_df)
    april = build_april_insight()
    save_csv(validation, OUT_DIR / "validation_leave_one_season.csv")
    save_csv(april, OUT_DIR / "april_rank_insight.csv")

    team_master = build_team_master()
    export_dashboard_csvs(pred, team_master)
    generate_pngs_and_html()

    summary = {
        "input_raw_dir": str(RAW_DIR),
        "input_processed_dir": str(PROCESSED_DIR),
        "output_dir": str(EXPERIMENT_DIR),
        "years": YEARS,
        "training_rows": int(len(train)),
        "latest_2026_date": str(pd.to_datetime(pred["date"]).max().date()),
        "predicted_top5": pred.head(5)["team"].tolist(),
        "generated_pngs": sorted(p.name for p in EXPERIMENT_DIR.glob("0*.png")),
    }
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(), ensure_ascii=False, indent=2))
