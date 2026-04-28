# src/preprocessing/clean_team_rank.py
#
# KBO 팀 순위 관련 원본 CSV 데이터를 전처리한다.
# 일자별 순위 데이터에서는 날짜, 순위, 승률, 최근 10경기, 홈/원정 성적,
# 연승/연패 흐름 등을 정리하고, 최종 순위 데이터에서는 postseason 라벨을 생성한다.

import pandas as pd

from src.utils.config import TOTAL_GAMES
from src.utils.parser import (
    read_csv_korean,
    to_numeric_safe,
    parse_recent10,
    parse_home_away,
    parse_streak,
)


# ─────────────────────────────────────────────
# 팀 일자별 순위 데이터 전처리
# ─────────────────────────────────────────────

def clean_team_daily_rank(path, season):
    """팀 일자별 순위 데이터를 전처리한다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
    df = df.rename(columns={
        "날짜": "date",
        "순위": "rank",
        "팀명": "team",
        "경기": "games",
        "승": "wins",
        "패": "losses",
        "무": "draws",
        "승률": "win_rate",
        "게임차": "games_behind",
        "최근10경기": "recent_10",
        "연속": "streak",
        "홈": "home_record",
        "방문": "away_record",
    })

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # 날짜 변환
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")

    # 숫자 컬럼 변환
    num_cols = [
        "rank",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "games_behind",
    ]

    # 순위, 경기 수, 승패무, 승률, 게임차를 숫자형으로 변환
    for col in num_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # 최근 10경기 파생변수
    df[[
        "recent10_wins",
        "recent10_draws",
        "recent10_losses",
        "recent10_win_rate",
    ]] = df["recent_10"].apply(parse_recent10)

    # 홈 성적 파생변수
    df[[
        "home_wins",
        "home_draws",
        "home_losses",
        "home_win_rate",
    ]] = df["home_record"].apply(parse_home_away)

    # 원정 성적 파생변수
    df[[
        "away_wins",
        "away_draws",
        "away_losses",
        "away_win_rate",
    ]] = df["away_record"].apply(parse_home_away)

    # 연속 기록 파생변수
    df[[
        "streak_type",
        "streak_count",
    ]] = df["streak"].apply(parse_streak)

    # 시즌 진행률
    df["games_played_ratio"] = df["games"] / TOTAL_GAMES

    # 남은 경기 수: 후반부일수록 현재 순위의 신뢰도 증가
    df["remaining_games"] = TOTAL_GAMES - df["games"]

    # 홈/원정 승률 차이: 양수면 원정에 약한 팀
    df["home_away_win_diff"] = df["home_win_rate"] - df["away_win_rate"]

    # 날짜에서 월 추출
    df["month"] = df["date"].dt.month

    # 전처리된 팀 일자별 순위 데이터 반환
    return df


# ─────────────────────────────────────────────
# 팀 최종 순위 라벨 생성
# ─────────────────────────────────────────────

def clean_team_final_rank(path, season):
    """시즌 최종 순위 데이터로 postseason 라벨을 만든다."""

    # 한글 인코딩 문제를 방지하며 CSV 읽기
    df = read_csv_korean(path)

    # KBO 원본 컬럼명을 프로젝트에서 사용할 영문 컬럼명으로 변경
    df = df.rename(columns={
        "순위": "final_rank",
        "팀명": "team",
    })

    # 어느 시즌 데이터인지 구분하기 위한 season 컬럼 추가
    df["season"] = season

    # 최종 순위를 숫자형으로 변환
    df["final_rank"] = to_numeric_safe(df["final_rank"])

    # 최종 5위 이내면 포스트시즌 진출
    df["postseason"] = (df["final_rank"] <= 5).astype(int)

    # 모델 정답 라벨로 사용할 핵심 컬럼만 반환
    return df[["season", "team", "final_rank", "postseason"]]