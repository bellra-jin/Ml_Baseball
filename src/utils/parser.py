# src/utils/parser.py

import re
import numpy as np
import pandas as pd


def read_csv_korean(path):
    """한글 CSV 인코딩 오류를 방지해서 CSV 파일을 읽는다."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def to_numeric_safe(series):
    """콤마, %, -, 공백이 섞인 값을 숫자형으로 변환한다."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "-": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def ip_to_float(value):
    """야구 이닝 표기값을 소수로 변환한다."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value in ["", "-"]:
        return np.nan

    # 예: 123 1/3
    if " " in value:
        whole, frac = value.split(" ")
        num, den = frac.split("/")
        return float(whole) + float(num) / float(den)

    # 예: 1/3
    if "/" in value:
        num, den = value.split("/")
        return float(num) / float(den)

    return pd.to_numeric(value, errors="coerce")


def parse_recent10(record):
    """최근10경기 문자열을 승/무/패/승률로 분리한다."""
    record = str(record)

    win = re.search(r"(\d+)승", record)
    draw = re.search(r"(\d+)무", record)
    loss = re.search(r"(\d+)패", record)

    wins = int(win.group(1)) if win else 0
    draws = int(draw.group(1)) if draw else 0
    losses = int(loss.group(1)) if loss else 0

    total = wins + draws + losses
    win_rate = wins / total if total > 0 else 0

    return pd.Series([wins, draws, losses, win_rate])


def parse_home_away(record):
    """홈/방문 성적 문자열을 승/무/패/승률로 분리한다."""
    record = str(record).strip()

    try:
        wins, draws, losses = map(int, record.split("-"))
    except ValueError:
        wins, draws, losses = 0, 0, 0

    total = wins + draws + losses
    win_rate = wins / total if total > 0 else 0

    return pd.Series([wins, draws, losses, win_rate])


def parse_streak(streak):
    """연승/연패 문자열을 방향과 개수로 분리한다."""
    streak = str(streak)

    count = re.search(r"(\d+)", streak)
    count = int(count.group(1)) if count else 0

    if "승" in streak:
        streak_type = 1
    elif "패" in streak:
        streak_type = -1
    else:
        streak_type = 0

    return pd.Series([streak_type, count])