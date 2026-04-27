import pandas as pd

from src.utils.config import TOTAL_GAMES
from src.utils.parser import (
    read_csv_korean,
    to_numeric_safe,
    parse_recent10,
    parse_home_away,
    parse_streak,
)


def clean_team_daily_rank(path, season):
    """팀 일자별 순위 데이터를 전처리한다."""
    df = read_csv_korean(path)

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

    return df


def clean_team_final_rank(path, season):
    """시즌 최종 순위 데이터로 postseason 라벨을 만든다."""
    df = read_csv_korean(path)

    df = df.rename(columns={
        "순위": "final_rank",
        "팀명": "team",
    })

    df["season"] = season
    df["final_rank"] = to_numeric_safe(df["final_rank"])

    # 최종 5위 이내면 포스트시즌 진출
    df["postseason"] = (df["final_rank"] <= 5).astype(int)

    return df[["season", "team", "final_rank", "postseason"]]