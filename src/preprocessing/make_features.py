import numpy as np
import pandas as pd


def make_prev_features(stats_df):
    """
    현재 연도 기록을 다음 시즌의 전년도 기록으로 바꾼다.

    예:
    2022 팀 기록 → 2023 학습 데이터의 prev_ 변수
    2023 팀 기록 → 2024 학습 데이터의 prev_ 변수
    2024 팀 기록 → 2025 학습 데이터의 prev_ 변수
    2025 팀 기록 → 2026 예측 데이터의 prev_ 변수
    """
    df = stats_df.copy()

    df["season"] = df["season"] + 1

    rename_cols = {
        col: f"prev_{col}"
        for col in df.columns
        if col not in ["season", "team"]
    }

    df = df.rename(columns=rename_cols)

    return df


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
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    df = _add_delta_30d(df)
    df = _add_recent_n_win_rate(df, 20)
    df = _add_recent_n_win_rate(df, 30)
    df = _add_games_behind_5th(df)

    return df


def _add_delta_30d(df):
    """30일 전 대비 승률·순위 변화를 추가한다."""
    df = df.copy()
    df["win_rate_delta_30d"] = np.nan
    df["rank_delta_30d"] = np.nan

    for _, g in df.groupby("team"):
        g = g.sort_values("date")
        dates = g["date"].values
        wr = g["win_rate"].values
        rk = g["rank"].values
        idx = g.index

        for i, di in enumerate(idx):
            cutoff = dates[i] - np.timedelta64(30, "D")
            past = np.where(dates <= cutoff)[0]
            if past.size == 0:
                continue
            j = past[-1]
            df.at[di, "win_rate_delta_30d"] = float(wr[i]) - float(wr[j])
            # 순위가 내려갈수록 숫자가 커지므로 부호 반전 → 양수=상승
            df.at[di, "rank_delta_30d"] = float(rk[j]) - float(rk[i])

    return df


def _add_recent_n_win_rate(df, n):
    """최근 n경기 승률을 누적 기록에서 역산한다."""
    col = f"recent{n}_win_rate"
    df = df.copy()
    df[col] = np.nan

    for _, g in df.groupby("team"):
        g = g.sort_values("games")
        idx = g.index
        games_arr = g["games"].values
        wins_arr = g["wins"].values
        losses_arr = g["losses"].values

        for i, di in enumerate(idx):
            target = games_arr[i] - n
            past = np.where(games_arr <= target)[0]
            if past.size == 0:
                continue
            j = past[-1]
            recent_wins = int(wins_arr[i]) - int(wins_arr[j])
            recent_losses = int(losses_arr[i]) - int(losses_arr[j])
            total = recent_wins + recent_losses
            if total > 0:
                df.at[di, col] = recent_wins / total

    return df


def _add_games_behind_5th(df):
    """5위와의 게임차 및 5위 추월에 필요한 승수를 추가한다."""
    df = df.copy()
    df["games_behind_5th"] = np.nan
    df["wins_to_5th"] = np.nan

    for _, day in df.groupby("date"):
        fifth = day[day["rank"] == 5]
        if fifth.empty:
            continue
        fifth_row = fifth.iloc[0]

        gb = (
            (fifth_row["wins"] - day["wins"]) +
            (day["losses"] - fifth_row["losses"])
        ) / 2
        df.loc[day.index, "games_behind_5th"] = gb.values

        # 5위 팀의 승수를 따라잡기 위해 필요한 최소 승수 (이미 앞서면 0)
        wins_needed = (fifth_row["wins"] - day["wins"]).clip(lower=0)
        df.loc[day.index, "wins_to_5th"] = wins_needed.values

    return df
