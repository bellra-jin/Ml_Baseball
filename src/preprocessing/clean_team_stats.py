# src/preprocessing/clean_team_stats.py
#
# 팀 타자/투수/수비/주루 원본 기록을 전처리한다.
# 각 기록의 컬럼명을 영문 변수명으로 변경하고, 숫자형 변환 후
# 팀 단위 시즌 기록과 핵심 파생 지표를 생성한다.

import numpy as np

from src.utils.parser import read_csv_korean, to_numeric_safe, ip_to_float


# ─────────────────────────────────────────────
# 팀 타자 기록 전처리
# ─────────────────────────────────────────────

def clean_team_hitter(path, season):
    """팀 타자 기본/세부 기록을 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
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

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # 최종적으로 사용할 컬럼 목록
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

    # 원본 CSV에 없는 컬럼은 NaN으로 생성하여 컬럼 구조를 통일
    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    # season, team을 제외한 기록 컬럼을 숫자형으로 변환
    for col in use_cols:
        if col not in ["season", "team"]:
            df[col] = to_numeric_safe(df[col])

    # 필요한 컬럼만 선택해 반환
    return df[use_cols]


# ─────────────────────────────────────────────
# 선수 타자 기록 기반 팀 타자 기록 생성
# ─────────────────────────────────────────────

def clean_team_hitter_from_player(path, season):
    """팀 타자 파일이 없는 시즌은 선수 타격 기록을 팀 단위로 집계한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # 선수 타자 원본 컬럼명을 팀 집계용 영문 컬럼명으로 변경
    df = df.rename(columns={
        "팀명": "team",
        "R": "team_runs",
        "H": "team_hits",
        "HR": "team_hr",
        "RBI": "team_rbi",
        "TB": "team_tb",
        "PA": "team_pa",
        "AB": "team_ab",
        "BB": "team_bb",
        "SO": "team_so",
        "HBP": "team_hbp",
        "SF": "team_sf",
    })

    # 팀 단위로 합산할 누적 기록 컬럼 목록
    sum_cols = [
        "team_runs",
        "team_hits",
        "team_hr",
        "team_rbi",
        "team_tb",
        "team_pa",
        "team_ab",
        "team_bb",
        "team_so",
        "team_hbp",
        "team_sf",
    ]

    # 없는 컬럼은 0으로 만들고, 모든 합산 컬럼을 숫자형으로 변환
    for col in sum_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = to_numeric_safe(df[col]).fillna(0)

    # 선수별 기록을 팀 단위로 합산
    team = df.groupby("team", as_index=False)[sum_cols].sum()

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    team["season"] = season

    # 팀 타율 계산: 안타 / 타수
    team["team_avg"] = team["team_hits"] / team["team_ab"].replace(0, np.nan)

    # 팀 출루율 계산: (안타 + 볼넷 + 사구) / (타수 + 볼넷 + 사구 + 희생플라이)
    team["team_obp"] = (
        (team["team_hits"] + team["team_bb"] + team["team_hbp"])
        / (team["team_ab"] + team["team_bb"] + team["team_hbp"] + team["team_sf"]).replace(0, np.nan)
    )

    # 팀 장타율 계산: 총루타 / 타수
    team["team_slg"] = team["team_tb"] / team["team_ab"].replace(0, np.nan)

    # 팀 OPS 계산: 출루율 + 장타율
    team["team_ops"] = team["team_obp"] + team["team_slg"]

    # 최종적으로 사용할 컬럼 목록
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

    # 필요한 컬럼만 선택해 반환
    return team[use_cols]


# ─────────────────────────────────────────────
# 팀 투수 기록 전처리
# ─────────────────────────────────────────────

def clean_team_pitcher(path, season):
    """팀 투수 기본 기록을 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
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

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # 최종적으로 사용할 컬럼 목록
    use_cols = [
        "season", "team",
        "team_era", "team_whip", "team_runs_allowed",
        "team_er", "team_hr_allowed", "team_bb_allowed",
        "team_so_pitcher", "team_ip",
    ]

    # 원본 CSV에 없는 컬럼은 NaN으로 생성하여 컬럼 구조를 통일
    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 투구 이닝은 야구식 이닝 표기를 소수형으로 변환
    df["team_ip"] = df["team_ip"].apply(ip_to_float)

    # season, team, team_ip를 제외한 기록 컬럼을 숫자형으로 변환
    for col in use_cols:
        if col not in ["season", "team", "team_ip"]:
            df[col] = to_numeric_safe(df[col])

    # 필요한 컬럼만 선택해 반환
    return df[use_cols]


# ─────────────────────────────────────────────
# 팀 수비 기록 전처리
# ─────────────────────────────────────────────

def clean_team_defense(path, season):
    """팀 수비 기본 기록을 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
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

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # 최종적으로 사용할 컬럼 목록
    use_cols = [
        "season", "team",
        "team_errors", "team_fpct", "team_dp",
        "team_pb", "team_sb_allowed", "team_cs_defense", "team_cs_rate",
    ]

    # 원본 CSV에 없는 컬럼은 NaN으로 생성하여 컬럼 구조를 통일
    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    # season, team을 제외한 수비 기록 컬럼을 숫자형으로 변환
    for col in use_cols:
        if col not in ["season", "team"]:
            df[col] = to_numeric_safe(df[col])

    # 필요한 컬럼만 선택해 반환
    return df[use_cols]


# ─────────────────────────────────────────────
# 팀 주루 기록 전처리
# ─────────────────────────────────────────────

def clean_team_runner(path, season):
    """팀 주루 기본 기록을 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
    df = df.rename(columns={
        "팀명": "team",
        "SBA": "team_sba",
        "SB": "team_sb",
        "CS": "team_cs",
        "SB%": "team_sb_rate",
        "OOB": "team_oob",
        "PKO": "team_runner_pko",
    })

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # 최종적으로 사용할 컬럼 목록
    use_cols = [
        "season", "team",
        "team_sba", "team_sb", "team_cs",
        "team_sb_rate", "team_oob", "team_runner_pko",
    ]

    # 원본 CSV에 없는 컬럼은 NaN으로 생성하여 컬럼 구조를 통일
    for col in use_cols:
        if col not in df.columns:
            df[col] = np.nan

    # season, team을 제외한 주루 기록 컬럼을 숫자형으로 변환
    for col in use_cols:
        if col not in ["season", "team"]:
            df[col] = to_numeric_safe(df[col])

    # 필요한 컬럼만 선택해 반환
    return df[use_cols]


# ─────────────────────────────────────────────
# 팀 기록 병합 및 파생 변수 생성
# ─────────────────────────────────────────────

def merge_team_stats(hitter_df, pitcher_df, defense_df, runner_df):
    """팀 타자/투수/수비/주루 기록을 하나로 합친다."""

    # 팀 타자 기록을 기준으로 투수/수비/주루 기록을 병합
    team_stats = (
        hitter_df
        .merge(pitcher_df, on=["season", "team"], how="left")
        .merge(defense_df, on=["season", "team"], how="left")
        .merge(runner_df, on=["season", "team"], how="left")
    )

    # 득실차: 공격력과 수비력을 동시에 반영하는 핵심 전력 지표
    team_stats["run_differential"] = (
        team_stats["team_runs"] - team_stats["team_runs_allowed"]
    )

    # 피타고라스 승률: 득실차 기반 예상 승률 (운을 제거한 실력 지표)
    r = team_stats["team_runs"]
    ra = team_stats["team_runs_allowed"]
    team_stats["pythagorean_win_rate"] = r ** 2 / (r ** 2 + ra ** 2)

    # 삼진/볼넷 비율: 투수 제구력 지표 (ERA보다 운의 영향을 덜 받음)
    team_stats["k_bb_ratio"] = (
        team_stats["team_so_pitcher"]
        / team_stats["team_bb_allowed"].replace(0, np.nan)
    )

    # 순수 장타력: 단순 안타가 아닌 장타(홈런·2루타) 의존도
    team_stats["iso"] = team_stats["team_slg"] - team_stats["team_avg"]

    # 볼넷 비율: 선구안 기반 출루 능력 (운의 영향이 적음)
    team_stats["bb_rate"] = (
        team_stats["team_bb"]
        / team_stats["team_pa"].replace(0, np.nan)
    )

    # 병합 및 파생 변수 생성이 끝난 팀 시즌 기록 반환
    return team_stats