# src/preprocessing/clean_team_stats.py

import numpy as np

from src.utils.parser import read_csv_korean, to_numeric_safe, ip_to_float

def clean_team_hitter(path, season):
    """팀 타자 기본/세부 기록을 전처리한다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "팀명": "team",

        # 기본 기록
        "AVG": "team_avg",
        "R": "team_runs",
        "H": "team_hits",
        "HR": "team_hr",
        "RBI": "team_rbi",
        "TB": "team_tb",
        "PA": "team_pa",
        "AB": "team_ab",

        # 세부 기록
        "OBP": "team_obp",
        "SLG": "team_slg",
        "OPS": "team_ops",
        "BB": "team_bb",
        "SO": "team_so",
    })

    df["season"] = season

    use_cols = [
        "season", "team",

        "team_avg",
        "team_runs",
        "team_hits",
        "team_hr",
        "team_rbi",
        "team_tb",
        "team_pa",
        "team_ab",

        "team_obp",
        "team_slg",
        "team_ops",
        "team_bb",
        "team_so",
    ]

    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in use_cols:
        if col not in ["season", "team"]:
            df[col] = to_numeric_safe(df[col])

    return df[use_cols]

def clean_team_pitcher(path, season):
    """팀 투수 기본 기록을 전처리한다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "팀명": "team",
        "ERA": "team_era",
        "WHIP": "team_whip",
        "R": "team_runs_allowed",
        "ER": "team_er",
        "HR": "team_hr_allowed",
        "BB": "team_bb_allowed",
        "SO": "team_so_pitcher",
        "IP": "team_ip",
    })

    df["season"] = season

    use_cols = [
        "season", "team",
        "team_era", "team_whip", "team_runs_allowed",
        "team_er", "team_hr_allowed", "team_bb_allowed",
        "team_so_pitcher", "team_ip",
    ]

    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["team_ip"] = df["team_ip"].apply(ip_to_float)

    for col in use_cols:
        if col not in ["season", "team", "team_ip"]:
            df[col] = to_numeric_safe(df[col])

    return df[use_cols]


def clean_team_defense(path, season):
    """팀 수비 기본 기록을 전처리한다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "팀명": "team",
        "E": "team_errors",
        "FPCT": "team_fpct",
        "DP": "team_dp",
        "PB": "team_pb",
        "SB": "team_sb_allowed",
        "CS": "team_cs_defense",
        "CS%": "team_cs_rate",
    })

    df["season"] = season

    use_cols = [
        "season", "team",
        "team_errors", "team_fpct", "team_dp",
        "team_pb", "team_sb_allowed", "team_cs_defense", "team_cs_rate",
    ]

    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in use_cols:
        if col not in ["season", "team"]:
            df[col] = to_numeric_safe(df[col])

    return df[use_cols]


def clean_team_runner(path, season):
    """팀 주루 기본 기록을 전처리한다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "팀명": "team",
        "SBA": "team_sba",
        "SB": "team_sb",
        "CS": "team_cs",
        "SB%": "team_sb_rate",
        "OOB": "team_oob",
        "PKO": "team_runner_pko",
    })

    df["season"] = season

    use_cols = [
        "season", "team",
        "team_sba", "team_sb", "team_cs",
        "team_sb_rate", "team_oob", "team_runner_pko",
    ]

    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in use_cols:
        if col not in ["season", "team"]:
            df[col] = to_numeric_safe(df[col])

    return df[use_cols]


def merge_team_stats(hitter_df, pitcher_df, defense_df, runner_df):
    """팀 타자/투수/수비/주루 기록을 하나로 합친다."""
    team_stats = (
        hitter_df
        .merge(pitcher_df, on=["season", "team"], how="left")
        .merge(defense_df, on=["season", "team"], how="left")
        .merge(runner_df, on=["season", "team"], how="left")
    )

    return team_stats