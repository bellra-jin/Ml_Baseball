# src/preprocessing/clean_player_stats.py
#
# 선수별 타자/투수 원본 기록을 읽어 모델링에 사용할 수 있는 형태로 정제한다.
# 컬럼명을 영문 변수명으로 변경하고, 숫자형 변환 및 팀 단위 요약 변수를 생성한다.

import numpy as np
import pandas as pd

from src.utils.parser import read_csv_korean, to_numeric_safe, ip_to_float
from src.utils.config import MIN_HITTER_PA


# ─────────────────────────────────────────────
# 선수 타자 기본 기록 전처리
# ─────────────────────────────────────────────

def clean_player_hitter_basic(path, season):
    """선수 타자 기본 기록을 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
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

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # season, team, player_name을 제외한 기록 컬럼을 숫자형으로 변환
    for col in df.columns:
        if col not in ["season", "team", "player_name"]:
            df[col] = to_numeric_safe(df[col])

    # 최소 타석 기준 미만 선수는 표본이 작으므로 제외
    df = df[df["hitter_pa"] >= MIN_HITTER_PA]

    # 정제된 선수 타자 기본 기록 반환
    return df


# ─────────────────────────────────────────────
# 선수 타자 세부 기록 전처리
# ─────────────────────────────────────────────

def clean_player_hitter_detail(path, season):
    """선수 타자 세부 기록을 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 세부 기록 컬럼명을 영문 컬럼명으로 변경
    df = df.rename(columns={
        "선수명": "player_name",
        "팀명": "team",
        "BB/K": "hitter_bb_k",
        "P/PA": "hitter_p_pa",
        "ISOP": "hitter_isop",
        "XR": "hitter_xr",
        "GPA": "hitter_gpa",
    })

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # season, team, player_name을 제외한 세부 지표 컬럼을 숫자형으로 변환
    for col in df.columns:
        if col not in ["season", "team", "player_name"]:
            df[col] = to_numeric_safe(df[col])

    # 정제된 선수 타자 세부 기록 반환
    return df


# ─────────────────────────────────────────────
# 팀별 타자 요약 변수 생성
# ─────────────────────────────────────────────

def make_hitter_summary(basic_df, detail_df=None):
    """선수 타자 기록을 팀 단위 요약 변수로 변환한다."""

    # 원본 DataFrame 변경을 피하기 위해 복사본 사용
    df = basic_df.copy()

    # 팀별 요약 결과를 담을 리스트
    result = []

    # 시즌과 팀 단위로 선수 기록 그룹화
    for (season, team), g in df.groupby(["season", "team"]):

        # PA 기준 상위 5명(주전)을 고정하고 모든 지표를 같은 선수에서 계산한다.
        starters = g.sort_values("hitter_pa", ascending=False).head(5)

        # 상위 5명 중 상위 3명을 중심 타선으로 간주
        cleanup = starters.head(3)

        # 팀별 기본 타자 요약 변수 생성
        row = {
            "season": season,
            "team": team,
            "top5_hitter_avg": starters["hitter_avg"].mean(),
            "top5_hitter_hr_sum": starters["hitter_hr"].sum(),
            "top5_hitter_rbi_sum": starters["hitter_rbi"].sum(),
        }

        # OPS 컬럼이 있는 경우 OPS 기반 요약 변수 생성
        if "hitter_ops" in starters.columns:
            top5_ops = starters["hitter_ops"].mean()
            top3_ops = cleanup["hitter_ops"].mean()

            # 주전 5명 평균 OPS
            row["top5_hitter_ops_avg"] = top5_ops

            # 중심 타선 3명 평균 OPS
            row["top3_hitter_ops_avg"] = top3_ops

            # top5가 0이면 NaN 처리
            row["ops_concentration"] = (
                top3_ops / top5_ops if top5_ops > 0 else float("nan")
            )

        # OBP 컬럼이 있는 경우 주전 5명 평균 출루율 생성
        if "hitter_obp" in starters.columns:
            row["top5_hitter_obp_avg"] = starters["hitter_obp"].mean()

        # SLG 컬럼이 있는 경우 주전 5명 평균 장타율 생성
        if "hitter_slg" in starters.columns:
            row["top5_hitter_slg_avg"] = starters["hitter_slg"].mean()

        # 한 팀의 요약 결과를 리스트에 추가
        result.append(row)

    # 팀별 타자 요약 결과를 DataFrame으로 반환
    return pd.DataFrame(result)


# ─────────────────────────────────────────────
# 선수 투수 기본 기록 전처리
# ─────────────────────────────────────────────

def clean_player_pitcher_basic(path, season):
    """선수 투수 기본 기록을 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
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

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # 투구 이닝은 123 1/3 같은 야구식 표기를 소수형으로 변환
    if "pitcher_ip" in df.columns:
        df["pitcher_ip"] = df["pitcher_ip"].apply(ip_to_float)

    # season, team, player_name, pitcher_ip를 제외한 기록 컬럼을 숫자형으로 변환
    for col in df.columns:
        if col not in ["season", "team", "player_name", "pitcher_ip"]:
            df[col] = to_numeric_safe(df[col])

    # 정제된 선수 투수 기본 기록 반환
    return df


# ─────────────────────────────────────────────
# 팀별 투수 요약 변수 생성
# ─────────────────────────────────────────────

def make_pitcher_summary(basic_df):
    """선수 투수 기록을 팀 단위 요약 변수로 변환한다."""

    # 원본 DataFrame 변경을 피하기 위해 복사본 사용
    df = basic_df.copy()

    # 팀별 요약 결과를 담을 리스트
    result = []

    # 시즌과 팀 단위로 선수 기록 그룹화
    for (season, team), g in df.groupby(["season", "team"]):

        # 이닝 상위 5명을 주요 투수(선발 로테이션)로 본다.
        top_ip = g.sort_values("pitcher_ip", ascending=False).head(5)

        # 주요 투수 5명의 평균 ERA 계산
        rotation_era_avg = top_ip["pitcher_era"].mean()

        # IP 60이닝 이상을 선발 기준으로 삼아 에이스(최저 ERA)를 뽑는다.
        starters = g[g["pitcher_ip"] >= 60]

        # 60이닝 이상 투수가 없으면 이닝 1위 투수를 에이스 후보로 사용
        if starters.empty:
            starters = top_ip.head(1)

        # 선발 후보 중 ERA가 가장 낮은 투수를 에이스로 간주
        ace_era = starters["pitcher_era"].min()

        # 팀별 투수 요약 변수 생성
        row = {
            "season": season,
            "team": team,
            "top5_pitcher_era_avg": rotation_era_avg,
            "top5_pitcher_whip_avg": top_ip["pitcher_whip"].mean(),
            "top5_pitcher_ip_sum": top_ip["pitcher_ip"].sum(),
            "top5_pitcher_so_sum": top_ip["pitcher_so"].sum(),
            "ace_era": ace_era,
            # 로테이션 평균 ERA - 에이스 ERA: 클수록 에이스 의존도 높음
            "ace_era_gap": rotation_era_avg - ace_era,
        }

        # 한 팀의 요약 결과를 리스트에 추가
        result.append(row)

    # 팀별 투수 요약 결과를 DataFrame으로 반환
    return pd.DataFrame(result)