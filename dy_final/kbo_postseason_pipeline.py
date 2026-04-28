# -*- coding: utf-8 -*-
"""KBO postseason preprocessing and lightweight prediction pipeline.

The script reads the local KBO CSV files, builds leakage-aware features, trains a
small NumPy logistic model, and exports analysis-ready tables.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\data")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "kbo_outputs"
REGULAR_SEASON_GAMES = 144


FILES = {
    "team_rank": "팀_순위.csv",
    "team_daily_rank": "팀_일자별순위.csv",
    "team_hitter": "팀_타자_기본기록.csv",
    "team_pitcher": "팀_투수_기본기록.csv",
    "team_defense": "팀_수비_기본기록.csv",
    "team_runner": "팀_주루_기본기록.csv",
    "hitter_basic": "타자_기본기록.csv",
    "hitter_detail": "타자_세부기록.csv",
    "pitcher_basic": "투수_기본기록.csv",
    "pitcher_detail": "투수_세부기록.csv",
}


COL = {
    "season": "연도",
    "date": "날짜",
    "rank": "순위",
    "team": "팀명",
    "games": "경기",
    "wins": "승",
    "losses": "패",
    "draws": "무",
    "win_rate": "승률",
    "games_behind": "게임차",
    "recent10": "최근10경기",
    "streak": "연속",
    "home": "홈",
    "away": "방문",
    "player": "선수명",
}


FEATURES = [
    "rank",
    "games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "games_behind",
    "recent10_win_rate",
    "home_win_rate",
    "away_win_rate",
    "games_played_ratio",
    "month",
    "prev_final_rank",
    "prev_postseason",
    "prev_win_rate",
    "prev_avg",
    "prev_runs_per_game",
    "prev_hr_per_game",
    "prev_era",
    "prev_whip",
    "prev_so_per9",
    "prev_bb_per9",
    "prev_ra_per_game",
    "prev_fpct",
    "prev_errors_per_game",
    "prev_sb_rate",
    "prev_sb_per_game",
    "prev_top5_hitter_hr",
    "prev_top5_hitter_rbi",
    "prev_top5_hitter_xr",
    "prev_top5_hitter_gpa",
    "prev_top5_pitcher_ip",
    "prev_top5_pitcher_so",
    "prev_top5_pitcher_era",
    "prev_top5_pitcher_whip",
]


KOREAN_FEATURE_NAMES = {
    "rank": "현재 순위",
    "games": "현재 경기수",
    "wins": "현재 승",
    "losses": "현재 패",
    "draws": "현재 무",
    "win_rate": "현재 승률",
    "games_behind": "현재 게임차",
    "recent10_win_rate": "최근10경기 승률",
    "home_win_rate": "홈 승률",
    "away_win_rate": "원정 승률",
    "games_played_ratio": "시즌 진행률",
    "month": "월",
    "prev_final_rank": "전년도 최종순위",
    "prev_postseason": "전년도 가을야구",
    "prev_win_rate": "전년도 승률",
    "prev_avg": "전년도 팀 타율",
    "prev_runs_per_game": "전년도 득점/G",
    "prev_hr_per_game": "전년도 홈런/G",
    "prev_era": "전년도 ERA",
    "prev_whip": "전년도 WHIP",
    "prev_so_per9": "전년도 K/9",
    "prev_bb_per9": "전년도 BB/9",
    "prev_ra_per_game": "전년도 실점/G",
    "prev_fpct": "전년도 수비율",
    "prev_errors_per_game": "전년도 실책/G",
    "prev_sb_rate": "전년도 도루성공률",
    "prev_sb_per_game": "전년도 도루/G",
    "prev_top5_hitter_hr": "전년도 상위5타자 HR",
    "prev_top5_hitter_rbi": "전년도 상위5타자 RBI",
    "prev_top5_hitter_xr": "전년도 상위5타자 XR",
    "prev_top5_hitter_gpa": "전년도 상위5타자 GPA",
    "prev_top5_pitcher_ip": "전년도 상위5투수 IP",
    "prev_top5_pitcher_so": "전년도 상위5투수 SO",
    "prev_top5_pitcher_era": "전년도 상위5투수 ERA",
    "prev_top5_pitcher_whip": "전년도 상위5투수 WHIP",
}


@dataclass
class LogisticModel:
    features: list[str]
    mean: pd.Series
    std: pd.Series
    weights: np.ndarray

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = prepare_matrix(df, self.features, self.mean, self.std)
        z = x @ self.weights
        return sigmoid(z)


def standing_probability(df: pd.DataFrame) -> pd.Series:
    """Stable current-form score from rank, win rate, games behind, and recent form."""
    out = pd.Series(index=df.index, dtype="float64")
    group_cols = ["season"]
    if "date" in df.columns:
        group_cols.append("date")

    for _, group in df.groupby(group_cols, dropna=False):
        rank_score = 1 - ((group["rank"] - 1) / 9)
        rank_score = rank_score.clip(0, 1)
        win_score = pd.to_numeric(group["win_rate"], errors="coerce").clip(0, 1)
        max_gb = pd.to_numeric(group["games_behind"], errors="coerce").max()
        if pd.isna(max_gb) or max_gb <= 0:
            gb_score = pd.Series(1.0, index=group.index)
        else:
            gb_score = (1 - group["games_behind"] / max_gb).clip(0, 1)
        recent = pd.to_numeric(group["recent10_win_rate"], errors="coerce").fillna(win_score).clip(0, 1)
        score = 0.42 * rank_score + 0.28 * win_score + 0.20 * gb_score + 0.10 * recent
        out.loc[group.index] = score.clip(0, 1)
    return out


def ensemble_probability(model: LogisticModel, df: pd.DataFrame, model_weight: float = 0.45) -> pd.Series:
    model_prob = pd.Series(model.predict_proba(df), index=df.index)
    standing_prob = standing_probability(df)
    return (model_weight * model_prob + (1 - model_weight) * standing_prob).clip(0, 1)


def read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise last_error if last_error else FileNotFoundError(path)


def to_num(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if text in {"", "-", "--"}:
        return np.nan
    text = text.replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_ip(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--"}:
        return np.nan
    if " " in text:
        base, frac = text.split(" ", 1)
        base_val = float(base)
        if "/" in frac:
            num, den = frac.split("/", 1)
            return base_val + float(num) / float(den)
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    # KBO sometimes encodes third-innings as 5.1 or 5.2.
    if re.fullmatch(r"\d+\.[12]", text):
        whole, third = text.split(".")
        return float(whole) + float(third) / 3.0
    return float(text)


def parse_recent10(value) -> tuple[float, float, float, float]:
    text = "" if pd.isna(value) else str(value)
    m = re.search(r"(\d+)승(\d+)무(\d+)패", text)
    if not m:
        return np.nan, np.nan, np.nan, np.nan
    wins, draws, losses = map(float, m.groups())
    denom = wins + losses
    rate = wins / denom if denom else np.nan
    return wins, draws, losses, rate


def parse_record(value) -> tuple[float, float, float, float]:
    text = "" if pd.isna(value) else str(value).strip()
    m = re.search(r"(\d+)-(\d+)-(\d+)", text)
    if not m:
        return np.nan, np.nan, np.nan, np.nan
    wins, draws, losses = map(float, m.groups())
    denom = wins + losses
    rate = wins / denom if denom else np.nan
    return wins, draws, losses, rate


def add_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(to_num)
    return df


def clean_rank_table(df: pd.DataFrame, season: int, daily: bool) -> pd.DataFrame:
    df = df.copy()
    df[COL["season"]] = season
    numeric = [
        COL["rank"],
        COL["games"],
        COL["wins"],
        COL["losses"],
        COL["draws"],
        COL["win_rate"],
        COL["games_behind"],
    ]
    df = add_numeric_columns(df, numeric)
    if daily:
        df[COL["date"]] = pd.to_datetime(
            df[COL["date"]].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        df[COL["date"]] = pd.NaT

    recent = df[COL["recent10"]].map(parse_recent10)
    df["recent10_wins"] = [x[0] for x in recent]
    df["recent10_draws"] = [x[1] for x in recent]
    df["recent10_losses"] = [x[2] for x in recent]
    df["recent10_win_rate"] = [x[3] for x in recent]

    home = df[COL["home"]].map(parse_record)
    away = df[COL["away"]].map(parse_record)
    for prefix, parsed in (("home", home), ("away", away)):
        df[f"{prefix}_wins"] = [x[0] for x in parsed]
        df[f"{prefix}_draws"] = [x[1] for x in parsed]
        df[f"{prefix}_losses"] = [x[2] for x in parsed]
        df[f"{prefix}_win_rate"] = [x[3] for x in parsed]

    df["games_played_ratio"] = df[COL["games"]] / REGULAR_SEASON_GAMES
    df["month"] = df[COL["date"]].dt.month if daily else np.nan
    return df.rename(
        columns={
            COL["rank"]: "rank",
            COL["team"]: "team",
            COL["games"]: "games",
            COL["wins"]: "wins",
            COL["losses"]: "losses",
            COL["draws"]: "draws",
            COL["win_rate"]: "win_rate",
            COL["games_behind"]: "games_behind",
            COL["season"]: "season",
            COL["date"]: "date",
        }
    )


def read_year_file(data_dir: Path, season: int, key: str) -> pd.DataFrame | None:
    path = data_dir / str(season) / FILES[key]
    if not path.exists():
        return None
    df = read_csv(path)
    df[COL["season"]] = season
    return df


def numeric_team_stats(df: pd.DataFrame | None, season: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"season": [], "team": []})
    df = df.copy().rename(columns={COL["team"]: "team", COL["season"]: "season"})
    for col in df.columns:
        if col not in {"team", "season"}:
            if col == "IP":
                df[col] = df[col].map(parse_ip)
            else:
                df[col] = df[col].map(to_num)
    return df


def load_final_rank(data_dir: Path, season: int) -> pd.DataFrame:
    df = read_year_file(data_dir, season, "team_rank")
    if df is None:
        return pd.DataFrame()
    clean = clean_rank_table(df, season, daily=False)
    return clean[["season", "team", "rank", "games", "wins", "losses", "draws", "win_rate", "games_behind"]].rename(
        columns={
            "rank": "final_rank",
            "games": "final_games",
            "wins": "final_wins",
            "losses": "final_losses",
            "draws": "final_draws",
            "win_rate": "final_win_rate",
            "games_behind": "final_games_behind",
        }
    )


def build_team_prev_features(data_dir: Path, prev_season: int) -> pd.DataFrame:
    hitter = numeric_team_stats(read_year_file(data_dir, prev_season, "team_hitter"), prev_season)
    pitcher = numeric_team_stats(read_year_file(data_dir, prev_season, "team_pitcher"), prev_season)
    defense = numeric_team_stats(read_year_file(data_dir, prev_season, "team_defense"), prev_season)
    runner = numeric_team_stats(read_year_file(data_dir, prev_season, "team_runner"), prev_season)
    final_rank = load_final_rank(data_dir, prev_season)

    teams = pd.DataFrame({"team": sorted(set().union(*(set(x["team"]) for x in [hitter, pitcher, defense, runner, final_rank] if "team" in x)))})
    teams["prev_season"] = prev_season

    merged = teams.merge(hitter.add_prefix("hit_"), left_on="team", right_on="hit_team", how="left")
    merged = merged.merge(pitcher.add_prefix("pit_"), left_on="team", right_on="pit_team", how="left")
    merged = merged.merge(defense.add_prefix("def_"), left_on="team", right_on="def_team", how="left")
    merged = merged.merge(runner.add_prefix("run_"), left_on="team", right_on="run_team", how="left")
    merged = merged.merge(final_rank.add_prefix("rank_"), left_on="team", right_on="rank_team", how="left")

    out = pd.DataFrame({"team": merged["team"], "prev_season": prev_season, "season": prev_season + 1})
    out["prev_final_rank"] = merged.get("rank_final_rank")
    out["prev_postseason"] = (out["prev_final_rank"] <= 5).astype(float)
    out["prev_win_rate"] = merged.get("rank_final_win_rate")
    out["prev_avg"] = merged.get("hit_AVG")
    out["prev_runs_per_game"] = safe_div(merged.get("hit_R"), merged.get("hit_G"))
    out["prev_hr_per_game"] = safe_div(merged.get("hit_HR"), merged.get("hit_G"))
    out["prev_rbi_per_game"] = safe_div(merged.get("hit_RBI"), merged.get("hit_G"))
    out["prev_era"] = merged.get("pit_ERA")
    out["prev_whip"] = merged.get("pit_WHIP")
    out["prev_so_per9"] = safe_div(merged.get("pit_SO") * 9, merged.get("pit_IP"))
    out["prev_bb_per9"] = safe_div(merged.get("pit_BB") * 9, merged.get("pit_IP"))
    out["prev_ra_per_game"] = safe_div(merged.get("pit_R"), merged.get("pit_G"))
    out["prev_fpct"] = merged.get("def_FPCT")
    out["prev_errors_per_game"] = safe_div(merged.get("def_E"), merged.get("def_G"))
    out["prev_sb_rate"] = merged.get("run_SB%")
    out["prev_sb_per_game"] = safe_div(merged.get("run_SB"), merged.get("run_G"))
    return out


def safe_div(a, b):
    if a is None or b is None:
        return np.nan
    aa = pd.Series(a, dtype="float64")
    bb = pd.Series(b, dtype="float64")
    return aa / bb.replace({0: np.nan})


def clean_player_table(df: pd.DataFrame | None, season: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().rename(
        columns={COL["team"]: "team", COL["player"]: "player", COL["season"]: "season"}
    )
    for col in df.columns:
        if col in {"team", "player", "season", "POS"}:
            continue
        if col == "IP":
            df[col] = df[col].map(parse_ip)
        else:
            df[col] = df[col].map(to_num)
    return df


def top_n_by_team(df: pd.DataFrame, sort_col: str, n: int = 5, ascending: bool = False) -> pd.DataFrame:
    if df.empty or sort_col not in df.columns:
        return pd.DataFrame()
    return (
        df.sort_values(["team", sort_col], ascending=[True, ascending])
        .groupby("team", as_index=False, group_keys=False)
        .head(n)
    )


def build_player_prev_features(data_dir: Path, prev_season: int) -> pd.DataFrame:
    hb = clean_player_table(read_year_file(data_dir, prev_season, "hitter_basic"), prev_season)
    hd = clean_player_table(read_year_file(data_dir, prev_season, "hitter_detail"), prev_season)
    pb = clean_player_table(read_year_file(data_dir, prev_season, "pitcher_basic"), prev_season)

    teams = set(hb.get("team", pd.Series(dtype=object))).union(set(pb.get("team", pd.Series(dtype=object))))
    out = pd.DataFrame({"team": sorted(teams), "prev_season": prev_season, "season": prev_season + 1})

    if not hb.empty:
        top_hit = top_n_by_team(hb, "PA", n=5, ascending=False)
        hit_agg = top_hit.groupby("team").agg(
            prev_top5_hitter_hr=("HR", "sum"),
            prev_top5_hitter_rbi=("RBI", "sum"),
            prev_top5_hitter_hits=("H", "sum"),
            prev_top5_hitter_tb=("TB", "sum"),
            prev_top5_hitter_pa=("PA", "sum"),
        )
        out = out.merge(hit_agg.reset_index(), on="team", how="left")

    if not hd.empty:
        top_detail = top_n_by_team(hd, "XR", n=5, ascending=False)
        detail_agg = top_detail.groupby("team").agg(
            prev_top5_hitter_xr=("XR", "sum"),
            prev_top5_hitter_gpa=("GPA", "mean"),
            prev_top5_hitter_isop=("ISOP", "mean"),
        )
        out = out.merge(detail_agg.reset_index(), on="team", how="left")

    if not pb.empty:
        top_pit = top_n_by_team(pb, "IP", n=5, ascending=False)
        pit_agg = top_pit.groupby("team").agg(
            prev_top5_pitcher_ip=("IP", "sum"),
            prev_top5_pitcher_so=("SO", "sum"),
            prev_top5_pitcher_bb=("BB", "sum"),
            prev_top5_pitcher_hr=("HR", "sum"),
            prev_top5_pitcher_era=("ERA", "mean"),
            prev_top5_pitcher_whip=("WHIP", "mean"),
        )
        out = out.merge(pit_agg.reset_index(), on="team", how="left")

    return out


def build_prev_features(data_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        prev = season - 1
        team_features = build_team_prev_features(data_dir, prev)
        player_features = build_player_prev_features(data_dir, prev)
        merged = team_features.merge(
            player_features.drop(columns=["prev_season"], errors="ignore"),
            on=["season", "team"],
            how="left",
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def load_daily_rank(data_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        df = read_year_file(data_dir, season, "team_daily_rank")
        if df is None:
            continue
        frames.append(clean_rank_table(df, season, daily=True))
    return pd.concat(frames, ignore_index=True)


def load_final_labels(data_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        final = load_final_rank(data_dir, season)
        if final.empty:
            continue
        final["postseason"] = (final["final_rank"] <= 5).astype(int)
        frames.append(final[["season", "team", "final_rank", "postseason"]])
    return pd.concat(frames, ignore_index=True)


def build_model_dataset(data_dir: Path, seasons: list[int]) -> pd.DataFrame:
    daily = load_daily_rank(data_dir, seasons)
    labels = load_final_labels(data_dir, [s for s in seasons if s < 2026])
    prev_features = build_prev_features(data_dir, seasons)
    df = daily.merge(labels, on=["season", "team"], how="left")
    df = df.merge(prev_features, on=["season", "team"], how="left")
    return df


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def prepare_matrix(df: pd.DataFrame, features: list[str], mean=None, std=None) -> np.ndarray:
    x = df.reindex(columns=features).copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std(ddof=0).replace({0: 1.0})
    x = x.fillna(mean).fillna(0.0)
    x = (x - mean) / std.replace({0: 1.0})
    x.insert(0, "intercept", 1.0)
    return x.to_numpy(dtype=float)


def fit_logistic(train_df: pd.DataFrame, features: list[str], target: str = "postseason") -> LogisticModel:
    usable = train_df.dropna(subset=[target]).copy()
    usable = usable[usable["games"] >= 10].copy()
    x_raw = usable.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    mean = x_raw.mean()
    std = x_raw.std(ddof=0).replace({0: 1.0})
    x = prepare_matrix(usable, features, mean, std)
    y = usable[target].to_numpy(dtype=float)

    # Class-balanced weights help because each daily snapshot has five positive
    # and five negative teams, but missing rows can still create slight skew.
    pos = max(y.sum(), 1.0)
    neg = max(len(y) - y.sum(), 1.0)
    sample_w = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))

    weights = np.zeros(x.shape[1], dtype=float)
    lr = 0.08
    reg = 0.05
    for _ in range(3500):
        p = sigmoid(x @ weights)
        grad = (x.T @ ((p - y) * sample_w)) / len(y)
        grad[1:] += reg * weights[1:]
        weights -= lr * grad
    return LogisticModel(features=features, mean=mean, std=std, weights=weights)


def evaluate_leave_one_season(model_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seasons = sorted(int(s) for s in model_df["season"].dropna().unique() if int(s) < 2026)
    for test_season in seasons:
        train = model_df[(model_df["season"] != test_season) & model_df["postseason"].notna()]
        test = model_df[(model_df["season"] == test_season) & model_df["postseason"].notna()]
        if train.empty or test.empty:
            continue
        model = fit_logistic(train, FEATURES)
        for label, subset in [
            ("daily_all_after_10g", test[test["games"] >= 10]),
            ("april_latest", latest_april_snapshot(test)),
            ("season_latest", latest_snapshot(test)),
        ]:
            if subset.empty:
                continue
            scored = subset.copy()
            scored["model_prob"] = model.predict_proba(scored)
            scored["standing_prob"] = standing_probability(scored)
            scored["prob"] = ensemble_probability(model, scored)
            daily_acc = float(((scored["prob"] >= 0.5).astype(int) == scored["postseason"]).mean())
            if label == "daily_all_after_10g":
                rows.append(
                    {
                        "test_season": test_season,
                        "slice": label,
                        "rows": len(scored),
                        "team_accuracy": daily_acc,
                        "top5_overlap": np.nan,
                        "baseline_current_top5_overlap": np.nan,
                    }
                )
            else:
                pred_top5 = set(scored.sort_values("prob", ascending=False).head(5)["team"])
                actual_top5 = set(scored[scored["postseason"] == 1]["team"])
                baseline_top5 = set(scored.sort_values("rank").head(5)["team"])
                rows.append(
                    {
                        "test_season": test_season,
                        "slice": label,
                        "rows": len(scored),
                        "team_accuracy": daily_acc,
                        "top5_overlap": len(pred_top5 & actual_top5),
                        "baseline_current_top5_overlap": len(baseline_top5 & actual_top5),
                    }
                )
    return pd.DataFrame(rows)


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    latest = df["date"].max()
    return df[df["date"] == latest].copy()


def latest_april_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    apr = df[df["date"].dt.month == 4].copy()
    if apr.empty:
        return apr
    return apr[apr["date"] == apr["date"].max()].copy()


def april_rank_insight(model_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in model_df[model_df["season"] < 2026].groupby("season"):
        snap = latest_april_snapshot(group)
        if snap.empty:
            continue
        pred_top5 = set(snap.sort_values("rank").head(5)["team"])
        actual_top5 = set(snap[snap["postseason"] == 1]["team"])
        corr = snap[["rank", "final_rank"]].corr(method="spearman").iloc[0, 1]
        rows.append(
            {
                "season": int(season),
                "april_latest_date": snap["date"].max().strftime("%Y-%m-%d"),
                "current_top5_overlap": len(pred_top5 & actual_top5),
                "current_top5_overlap_rate": len(pred_top5 & actual_top5) / 5,
                "rank_final_spearman": corr,
            }
        )
    return pd.DataFrame(rows)


def coefficient_importance(model: LogisticModel) -> pd.DataFrame:
    rows = []
    for feature, coef in zip(model.features, model.weights[1:]):
        rows.append(
            {
                "feature": feature,
                "feature_ko": KOREAN_FEATURE_NAMES.get(feature, feature),
                "coefficient": coef,
                "abs_coefficient": abs(coef),
                "direction": "positive" if coef > 0 else "negative",
            }
        )
    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def build_2026_team_snapshot(data_dir: Path) -> pd.DataFrame:
    season = 2026
    rank = clean_rank_table(read_year_file(data_dir, season, "team_rank"), season, daily=False)
    hit = numeric_team_stats(read_year_file(data_dir, season, "team_hitter"), season).add_prefix("hit_")
    pit = numeric_team_stats(read_year_file(data_dir, season, "team_pitcher"), season).add_prefix("pit_")
    defense = numeric_team_stats(read_year_file(data_dir, season, "team_defense"), season).add_prefix("def_")
    runner = numeric_team_stats(read_year_file(data_dir, season, "team_runner"), season).add_prefix("run_")
    out = rank.merge(hit, left_on="team", right_on="hit_team", how="left")
    out = out.merge(pit, left_on="team", right_on="pit_team", how="left")
    out = out.merge(defense, left_on="team", right_on="def_team", how="left")
    out = out.merge(runner, left_on="team", right_on="run_team", how="left")
    out["current_runs_per_game"] = safe_div(out.get("hit_R"), out.get("hit_G"))
    out["current_hr_per_game"] = safe_div(out.get("hit_HR"), out.get("hit_G"))
    out["current_era"] = out.get("pit_ERA")
    out["current_whip"] = out.get("pit_WHIP")
    out["current_errors_per_game"] = safe_div(out.get("def_E"), out.get("def_G"))
    out["current_sb_rate"] = out.get("run_SB%")
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_pipeline(data_dir: Path = DEFAULT_DATA_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = [2023, 2024, 2025, 2026]
    model_df = build_model_dataset(data_dir, seasons)
    train_df = model_df[(model_df["season"] < 2026) & model_df["postseason"].notna()]
    model = fit_logistic(train_df, FEATURES)

    latest_2026 = latest_snapshot(model_df[model_df["season"] == 2026])
    pred = latest_2026[["season", "date", "team", "rank", "games", "wins", "losses", "draws", "win_rate", "games_behind"]].copy()
    pred["model_probability"] = model.predict_proba(latest_2026)
    pred["standing_probability"] = standing_probability(latest_2026)
    pred["postseason_probability"] = ensemble_probability(model, latest_2026)
    pred["model_probability_pct"] = (pred["model_probability"] * 100).round(1)
    pred["standing_probability_pct"] = (pred["standing_probability"] * 100).round(1)
    pred["postseason_probability_pct"] = (pred["postseason_probability"] * 100).round(1)
    pred = pred.sort_values("postseason_probability", ascending=False).reset_index(drop=True)
    pred["predicted_top5"] = 0
    pred.loc[pred.index[:5], "predicted_top5"] = 1
    pred["prediction_label"] = np.where(pred["predicted_top5"] == 1, "진출", "미진출")

    validation = evaluate_leave_one_season(model_df)
    april = april_rank_insight(model_df)
    importance = coefficient_importance(model)
    team_snapshot = build_2026_team_snapshot(data_dir)

    save_csv(model_df, out_dir / "model_dataset_2023_2026.csv")
    save_csv(pred, out_dir / "2026_postseason_predictions.csv")
    save_csv(validation, out_dir / "validation_leave_one_season.csv")
    save_csv(april, out_dir / "april_rank_insight.csv")
    save_csv(importance, out_dir / "feature_importance_coefficients.csv")
    save_csv(team_snapshot, out_dir / "2026_team_snapshot.csv")

    latest_date = pred["date"].max()
    latest_date_str = latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else ""
    summary = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "training_rows": int(len(train_df[train_df["games"] >= 10])),
        "model_rows_total": int(len(model_df)),
        "latest_2026_date": latest_date_str,
        "predicted_top5": pred.head(5)["team"].tolist(),
        "april_current_top5_overlap_mean": float(april["current_top5_overlap"].mean()) if not april.empty else math.nan,
        "validation_april_ensemble_top5_overlap_mean": float(
            validation.loc[validation["slice"] == "april_latest", "top5_overlap"].mean()
        )
        if not validation.empty
        else math.nan,
        "validation_april_baseline_top5_overlap_mean": float(
            validation.loc[validation["slice"] == "april_latest", "baseline_current_top5_overlap"].mean()
        )
        if not validation.empty
        else math.nan,
    }

    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown_report(out_dir, pred, validation, april, importance, summary)
    return {
        "summary": summary,
        "predictions": pred,
        "validation": validation,
        "april": april,
        "importance": importance,
        "team_snapshot": team_snapshot,
        "model_dataset": model_df,
    }


def write_markdown_report(
    out_dir: Path,
    pred: pd.DataFrame,
    validation: pd.DataFrame,
    april: pd.DataFrame,
    importance: pd.DataFrame,
    summary: dict,
) -> None:
    top5 = ", ".join(summary["predicted_top5"])
    top_importance = importance.head(8)[["feature_ko", "coefficient"]]
    pred_table = pred[
        [
            "team",
            "rank",
            "wins",
            "losses",
            "win_rate",
            "games_behind",
            "model_probability_pct",
            "standing_probability_pct",
            "postseason_probability_pct",
            "prediction_label",
        ]
    ].rename(
        columns={
            "team": "팀",
            "rank": "현재순위",
            "wins": "승",
            "losses": "패",
            "win_rate": "승률",
            "games_behind": "게임차",
            "model_probability_pct": "모델확률(%)",
            "standing_probability_pct": "스탠딩확률(%)",
            "postseason_probability_pct": "가을야구확률(%)",
            "prediction_label": "예측",
        }
    )
    lines = [
        "# KBO 가을야구 예측 전처리·심층 분석 요약",
        "",
        f"- 2026 최신 로컬 일자별 순위 기준일: {summary['latest_2026_date']}",
        f"- 예측 진출 Top5: {top5}",
        f"- 학습 데이터: 2023~2025 일자별 팀 스냅샷 {summary['training_rows']:,}행(10경기 이상)",
        "- 훈련 피처는 당일 순위 상태와 전년도 팀/선수 지표만 사용해 최종 시즌 결과 누수를 피했습니다.",
        "",
        "## 2026 예측",
        df_to_markdown(pred_table),
        "",
        "## 4월 순위의 예측력",
        df_to_markdown(april) if not april.empty else "4월 스냅샷 없음",
        "",
        "## 시즌 홀드아웃 검증",
        df_to_markdown(validation) if not validation.empty else "검증 결과 없음",
        "",
        "## 영향도가 큰 피처",
        "계수 방향은 표본 시즌이 3개뿐이라 인과로 해석하지 말고, 모델이 민감하게 본 신호 정도로만 봅니다.",
        df_to_markdown(top_importance),
        "",
        "## 추가 크롤링 필요 데이터",
        "- 선수 프로필 전체: 프로필/통산/일자별/경기별/상황별 기록은 선수 단위 부상·출전 지속성·역할 변화를 보려면 필요합니다.",
        "- 선수 이동 현황: 2026 전력 변화, FA/트레이드/방출 영향을 전년도 지표에 보정하려면 필요합니다.",
        "- 팀 세부기록: 현재 로컬 기본기록만으로도 예측은 가능하지만 OPS/출루율/장타율, 투수 피안타율 등 세부지표가 있으면 설명력이 좋아집니다.",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in tmp.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in tmp.columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    artifacts = run_pipeline()
    print(json.dumps(artifacts["summary"], ensure_ascii=False, indent=2))
