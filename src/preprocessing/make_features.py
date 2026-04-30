# src/preprocessing/make_features.py
#
# 전처리된 팀 순위/시즌 기록 데이터를 바탕으로
# 모델 학습에 사용할 파생 변수를 생성한다.
# 전년도(prev_) 변수, 다년도 평균(avg3yr_)·추세(trend_) 변수,
# 일자별 순위 기반 추세 변수를 만든다.

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 다년도 집계 대상 컬럼
# ─────────────────────────────────────────────

# 최근 3년 평균과 전년 대비 추세를 계산할 핵심 전력 지표 목록
MULTI_YEAR_COLS = [
    "pythagorean_win_rate",
    "run_differential",
    "team_era",
    "top5_pitcher_era_avg",
    "k_bb_ratio",
    "top5_hitter_ops_avg",
    "ace_era",
    "iso",
    "ops_concentration",
    "bb_rate",
]


# ─────────────────────────────────────────────
# 전년도 변수 생성
# ─────────────────────────────────────────────

def make_prev_features(stats_df):
    """
    현재 연도 기록을 다음 시즌의 전년도 기록으로 바꾼다.

    예:
    2022 팀 기록 → 2023 학습 데이터의 prev_ 변수
    2023 팀 기록 → 2024 학습 데이터의 prev_ 변수
    2024 팀 기록 → 2025 학습 데이터의 prev_ 변수
    2025 팀 기록 → 2026 예측 데이터의 prev_ 변수
    """

    # 원본 DataFrame 변경을 피하기 위해 복사본 사용
    df = stats_df.copy()

    # 현재 시즌 기록을 다음 시즌에서 사용할 수 있도록 season을 1 증가
    df["season"] = df["season"] + 1

    # season, team을 제외한 모든 기록 컬럼에 prev_ 접두사 추가
    rename_cols = {
        col: f"prev_{col}"
        for col in df.columns
        if col not in ["season", "team"]
    }

    # 컬럼명 변경 적용
    df = df.rename(columns=rename_cols)

    # 다음 시즌용 전년도 기록 반환
    return df


# ─────────────────────────────────────────────
# 다년도 평균·추세 변수 생성
# ─────────────────────────────────────────────

def make_multi_year_features(predict_year, seasons):
    """
    다년도 평균·추세 변수를 생성한다.

    Parameters
    ----------
    predict_year : int
        예측 대상 시즌
    seasons : list[DataFrame | None]
        [t-1, t-2, t-3] 순서. 없는 연도는 None.

    Returns
    -------
    DataFrame
        season, team + avg3yr_ / trend_ 컬럼
    """

    # t-1 시즌 데이터를 기준으로 팀 목록 생성
    base = seasons[0]
    teams = sorted(base["team"].unique())

    # 팀명을 index로 설정한 시즌별 DataFrame을 담을 리스트
    indexed = []

    # t-1, t-2, t-3 시즌 데이터를 순서대로 처리
    for s in seasons:

        # 해당 시즌 데이터가 없으면 None으로 유지
        if s is None:
            indexed.append(None)

        else:
            # 실제 존재하는 다년도 집계 대상 컬럼만 선택
            cols = [c for c in MULTI_YEAR_COLS if c in s.columns]

            # 팀 기준으로 값을 쉽게 가져오기 위해 team을 index로 설정
            indexed.append(s[["team"] + cols].set_index("team"))

    # 결과 DataFrame 생성: 예측 시즌과 팀 목록 포함
    result = pd.DataFrame({"season": predict_year, "team": teams})

    # 핵심 전력 지표별로 avg3yr_ / trend_ 변수 생성
    for col in MULTI_YEAR_COLS:

        # t-1, t-2, t-3 값을 담을 리스트
        series = []

        # 시즌별 데이터에서 해당 컬럼 값을 팀 순서에 맞춰 추출
        for sdf in indexed:

            # 시즌 데이터가 없거나 해당 컬럼이 없으면 NaN 배열 사용
            if sdf is None or col not in sdf.columns:
                series.append(np.full(len(teams), np.nan))

            else:
                # 팀 순서를 맞춰 값 추출 후 float 타입으로 변환
                series.append(sdf[col].reindex(teams).values.astype(float))

        # t-1, t-2, t-3 값 분리
        s0 = series[0]
        s1 = series[1] if len(series) > 1 else np.full(len(teams), np.nan)
        s2 = series[2] if len(series) > 2 else np.full(len(teams), np.nan)

        # t-1, t-2 데이터가 모두 있는 팀만 추세 계산 가능
        ok2 = ~np.isnan(s0) & ~np.isnan(s1)

        # t-1, t-2, t-3 데이터가 모두 있는 팀만 3년 평균 계산 가능
        ok3 = ok2 & ~np.isnan(s2)

        # 최근 3개 시즌 평균 변수 생성
        result[f"avg3yr_{col}"] = np.where(ok3, (s0 + s1 + s2) / 3, np.nan)

        # 전년 대비 변화량 변수 생성: t-1 - t-2
        result[f"trend_{col}"]  = np.where(ok2, s0 - s1, np.nan)

    # 다년도 평균·추세 변수 반환
    return result


# ─────────────────────────────────────────────
# 일자별 순위 기반 파생 변수 생성
# ─────────────────────────────────────────────

def make_daily_features(df):
    """
    일자별 순위 데이터에서 추세·경계선 피처를 생성한다.

    생성 변수:
    - win_rate_delta_30d : 30일 전 대비 승률 변화 (양수=상승)
    - rank_delta_30d     : 30일 전 대비 순위 변화 (양수=상승)
    - recent20_win_rate  : 최근 20경기 승률
    - recent30_win_rate  : 최근 30경기 승률
    - games_behind_5th   : 5위와의 게임차 (음수=5위권 내)
    """

    # 원본 DataFrame 변경을 피하기 위해 복사본 사용
    df = df.copy()

    # 날짜 컬럼을 datetime 타입으로 변환
    df["date"] = pd.to_datetime(df["date"])

    # 팀별 날짜 순서로 정렬
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # 30일 전 대비 승률 변화와 순위 변화 생성
    df = _add_delta_30d(df)

    # 최근 20경기 승률 생성
    df = _add_recent_n_win_rate(df, 20)

    # 최근 30경기 승률 생성
    df = _add_recent_n_win_rate(df, 30)

    # 5위와의 게임차 및 5위 추월 필요 승수 생성
    df = _add_games_behind_5th(df)

    # 일자별 파생 변수 생성 결과 반환
    return df


# ─────────────────────────────────────────────
# 30일 전 대비 변화량 생성
# ─────────────────────────────────────────────

def _add_delta_30d(df):
    """30일 전 대비 승률·순위 변화를 추가한다."""

    # 원본 DataFrame 변경을 피하기 위해 복사본 사용
    df = df.copy()

    # 결과 컬럼을 먼저 NaN으로 초기화
    df["win_rate_delta_30d"] = np.nan
    df["rank_delta_30d"] = np.nan

    # 팀별로 날짜 순서에 따라 30일 전 기록을 탐색
    for _, g in df.groupby("team"):

        # 해당 팀 데이터를 날짜순으로 정렬
        g = g.sort_values("date")

        # 계산에 사용할 날짜, 승률, 순위, 원본 인덱스 추출
        dates = g["date"].values
        wr = g["win_rate"].values
        rk = g["rank"].values
        idx = g.index

        # 각 날짜별로 30일 이전 가장 가까운 기록을 찾음
        for i, di in enumerate(idx):

            # 현재 날짜 기준 30일 전 날짜
            cutoff = dates[i] - np.timedelta64(30, "D")

            # 30일 전 이하에 해당하는 과거 기록 위치 탐색
            past = np.where(dates <= cutoff)[0]

            # 과거 기록이 없으면 계산하지 않고 넘어감
            if past.size == 0:
                continue

            # 30일 전 이하 중 가장 최근 기록 선택
            j = past[-1]

            # 현재 승률 - 과거 승률
            df.at[di, "win_rate_delta_30d"] = float(wr[i]) - float(wr[j])

            # 순위가 내려갈수록 숫자가 커지므로 부호 반전 → 양수=상승
            df.at[di, "rank_delta_30d"] = float(rk[j]) - float(rk[i])

    # 30일 전 대비 변화량이 추가된 DataFrame 반환
    return df


# ─────────────────────────────────────────────
# 최근 N경기 승률 생성
# ─────────────────────────────────────────────

def _add_recent_n_win_rate(df, n):
    """최근 n경기 승률을 누적 기록에서 역산한다."""

    # 생성할 컬럼명 예: recent20_win_rate
    col = f"recent{n}_win_rate"

    # 원본 DataFrame 변경을 피하기 위해 복사본 사용
    df = df.copy()

    # 결과 컬럼을 먼저 NaN으로 초기화
    df[col] = np.nan

    # 팀별로 누적 경기 수 기준 정렬 후 최근 n경기 성적 계산
    for _, g in df.groupby("team"):

        # 누적 경기 수 순서로 정렬
        g = g.sort_values("games")

        # 계산에 사용할 원본 인덱스와 누적 경기/승/패 배열 추출
        idx = g.index
        games_arr = g["games"].values
        wins_arr = g["wins"].values
        losses_arr = g["losses"].values

        # 각 시점별로 n경기 전 기록을 찾아 최근 n경기 승률 계산
        for i, di in enumerate(idx):

            # 현재 누적 경기 수에서 n경기를 뺀 목표 지점
            target = games_arr[i] - n

            # 목표 지점 이하의 과거 기록 위치 탐색
            past = np.where(games_arr <= target)[0]

            # 과거 기록이 없으면 계산하지 않고 넘어감
            if past.size == 0:
                continue

            # n경기 전 이하 중 가장 가까운 기록 선택
            j = past[-1]

            # 현재 누적 승수 - 과거 누적 승수
            recent_wins = int(wins_arr[i]) - int(wins_arr[j])

            # 현재 누적 패수 - 과거 누적 패수
            recent_losses = int(losses_arr[i]) - int(losses_arr[j])

            # 무승부는 제외하고 승+패 기준으로 승률 계산
            total = recent_wins + recent_losses

            # 계산 가능한 경기 수가 있을 때만 승률 저장
            if total > 0:
                df.at[di, col] = recent_wins / total

    # 최근 n경기 승률이 추가된 DataFrame 반환
    return df


# ─────────────────────────────────────────────
# 5위 기준 게임차 및 필요 승수 생성
# ─────────────────────────────────────────────

def _add_games_behind_5th(df):
    """5위와의 게임차 및 5위 추월에 필요한 승수를 추가한다."""

    # 원본 DataFrame 변경을 피하기 위해 복사본 사용
    df = df.copy()

    # 결과 컬럼을 먼저 NaN으로 초기화
    df["games_behind_5th"] = np.nan
    df["wins_to_5th"] = np.nan

    # 날짜별로 5위 팀을 찾아 각 팀과의 차이를 계산
    for _, day in df.groupby("date"):

        # 해당 날짜의 5위 팀 데이터 추출
        fifth = day[day["rank"] == 5]

        # 5위 팀 정보가 없으면 계산하지 않고 넘어감
        if fifth.empty:
            continue

        # 5위 팀 행 선택
        fifth_row = fifth.iloc[0]

        # 5위 팀 기준 게임차 계산
        gb = (
            (fifth_row["wins"] - day["wins"]) +
            (day["losses"] - fifth_row["losses"])
        ) / 2

        # 날짜 내 모든 팀에 대해 5위와의 게임차 저장
        df.loc[day.index, "games_behind_5th"] = gb.values

        # 5위 팀의 승수를 따라잡기 위해 필요한 최소 승수 (이미 앞서면 0)
        wins_needed = (fifth_row["wins"] - day["wins"]).clip(lower=0)

        # 날짜 내 모든 팀에 대해 5위 추월 필요 승수 저장
        df.loc[day.index, "wins_to_5th"] = wins_needed.values

    # 5위 기준 파생 변수가 추가된 DataFrame 반환
    return df