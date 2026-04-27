import numpy as np
import pandas as pd

from src.utils.parser import read_csv_korean, to_numeric_safe, ip_to_float
from src.utils.config import MIN_HITTER_PA


def clean_player_hitter_basic(path, season):
    """선수 타자 기본 기록을 전처리한다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "선수명": "player_name",
        "팀명": "team",
        "AVG": "hitter_avg",
        "G": "hitter_g",
        "PA": "hitter_pa",
        "AB": "hitter_ab",
        "R": "hitter_runs",
        "H": "hitter_hits",
        "2B": "hitter_2b",
        "3B": "hitter_3b",
        "HR": "hitter_hr",
        "TB": "hitter_tb",
        "RBI": "hitter_rbi",
        "SAC": "hitter_sac",
        "SF": "hitter_sf",
        "OBP": "hitter_obp",
        "SLG": "hitter_slg",
        "OPS": "hitter_ops",
        "BB": "hitter_bb",
        "SO": "hitter_so",
    })

    df["season"] = season

    for col in df.columns:
        if col not in ["season", "team", "player_name"]:
            df[col] = to_numeric_safe(df[col])

    df = df[df["hitter_pa"] >= MIN_HITTER_PA]

    return df


def clean_player_hitter_detail(path, season):
    """선수 타자 세부 기록을 전처리한다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "선수명": "player_name",
        "팀명": "team",
        "BB/K": "hitter_bb_k",
        "P/PA": "hitter_p_pa",
        "ISOP": "hitter_isop",
        "XR": "hitter_xr",
        "GPA": "hitter_gpa",
    })

    df["season"] = season

    for col in df.columns:
        if col not in ["season", "team", "player_name"]:
            df[col] = to_numeric_safe(df[col])

    return df


def make_hitter_summary(basic_df, detail_df=None):
    """선수 타자 기록을 팀 단위 요약 변수로 변환한다."""
    df = basic_df.copy()

    result = []

    for (season, team), g in df.groupby(["season", "team"]):
        # PA 기준 상위 5명(주전)을 고정하고 모든 지표를 같은 선수에서 계산한다.
        starters = g.sort_values("hitter_pa", ascending=False).head(5)

        row = {
            "season": season,
            "team": team,
            "top5_hitter_avg": starters["hitter_avg"].mean(),
            "top5_hitter_hr_sum": starters["hitter_hr"].sum(),
            "top5_hitter_rbi_sum": starters["hitter_rbi"].sum(),
        }

        if "hitter_ops" in starters.columns:
            row["top5_hitter_ops_avg"] = starters["hitter_ops"].mean()

        if "hitter_obp" in starters.columns:
            row["top5_hitter_obp_avg"] = starters["hitter_obp"].mean()

        if "hitter_slg" in starters.columns:
            row["top5_hitter_slg_avg"] = starters["hitter_slg"].mean()

        result.append(row)

    return pd.DataFrame(result)


def clean_player_pitcher_basic(path, season):
    """선수 투수 기본 기록을 전처리한다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "선수명": "player_name",
        "팀명": "team",
        "ERA": "pitcher_era",
        "G": "pitcher_g",
        "W": "pitcher_w",
        "L": "pitcher_l",
        "SV": "pitcher_sv",
        "HLD": "pitcher_hld",
        "WPCT": "pitcher_wpct",
        "IP": "pitcher_ip",
        "H": "pitcher_h_allowed",
        "HR": "pitcher_hr_allowed",
        "BB": "pitcher_bb_allowed",
        "HBP": "pitcher_hbp",
        "SO": "pitcher_so",
        "R": "pitcher_runs_allowed",
        "ER": "pitcher_er",
        "WHIP": "pitcher_whip",
    })

    df["season"] = season

    if "pitcher_ip" in df.columns:
        df["pitcher_ip"] = df["pitcher_ip"].apply(ip_to_float)

    for col in df.columns:
        if col not in ["season", "team", "player_name", "pitcher_ip"]:
            df[col] = to_numeric_safe(df[col])

    return df


def make_pitcher_summary(basic_df):
    """선수 투수 기록을 팀 단위 요약 변수로 변환한다."""
    df = basic_df.copy()

    result = []

    for (season, team), g in df.groupby(["season", "team"]):
        # 이닝 상위 5명을 주요 투수로 본다.
        top_ip = g.sort_values("pitcher_ip", ascending=False).head(5)

        row = {
            "season": season,
            "team": team,
            "top5_pitcher_era_avg": top_ip["pitcher_era"].mean(),
            "top5_pitcher_whip_avg": top_ip["pitcher_whip"].mean(),
            "top5_pitcher_ip_sum": top_ip["pitcher_ip"].sum(),
            "top5_pitcher_so_sum": top_ip["pitcher_so"].sum(),
        }

        result.append(row)

    return pd.DataFrame(result)