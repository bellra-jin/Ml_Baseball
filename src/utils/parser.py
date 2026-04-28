# src/utils/parser.py
#
# KBO 원본 CSV 데이터에서 자주 등장하는 문자열 값을
# 머신러닝에 사용할 수 있는 숫자형 데이터로 변환하는 유틸 함수들을 정의한다.
# 한글 CSV 인코딩 처리, 숫자 변환, 이닝 변환,
# 최근 10경기/홈·원정/연승·연패 기록 파싱에 사용한다.

import re
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# CSV 파일 읽기
# ─────────────────────────────────────────────

def read_csv_korean(path):
    """한글 CSV 인코딩 오류를 방지해서 CSV 파일을 읽는다."""

    try:
        # utf-8-sig: 엑셀에서 저장한 UTF-8 CSV의 BOM 문제를 방지
        return pd.read_csv(path, encoding="utf-8-sig")

    except UnicodeDecodeError:
        # cp949: 윈도우/엑셀 기반 한글 CSV에서 자주 사용되는 인코딩
        return pd.read_csv(path, encoding="cp949")


# ─────────────────────────────────────────────
# 숫자형 데이터 변환
# ─────────────────────────────────────────────

def to_numeric_safe(series):
    """콤마, %, -, 공백이 섞인 값을 숫자형으로 변환한다."""

    return (
        series.astype(str)                              # 문자열로 변환
        .str.replace(",", "", regex=False)             # 1,234 → 1234
        .str.replace("%", "", regex=False)             # 56.7% → 56.7
        .str.strip()                                   # 앞뒤 공백 제거
        .replace({"": np.nan, "-": np.nan})            # 빈 값/하이픈은 결측치 처리
        .pipe(pd.to_numeric, errors="coerce")          # 숫자 변환 실패 시 NaN 처리
    )


# ─────────────────────────────────────────────
# 야구 이닝 표기 변환
# ─────────────────────────────────────────────

def ip_to_float(value):
    """야구 이닝 표기값을 소수로 변환한다."""

    # 결측치면 그대로 NaN 반환
    if pd.isna(value):
        return np.nan

    value = str(value).strip()                         # 문자열 변환 후 공백 제거

    # 빈 문자열 또는 하이픈은 결측치 처리
    if value in ["", "-"]:
        return np.nan

    # 예: "123 1/3" → 123.333...
    if " " in value:
        whole, frac = value.split(" ")
        num, den = frac.split("/")
        return float(whole) + float(num) / float(den)

    # 예: "1/3" → 0.333...
    if "/" in value:
        num, den = value.split("/")
        return float(num) / float(den)

    # 일반 숫자 문자열은 숫자형으로 변환
    return pd.to_numeric(value, errors="coerce")


# ─────────────────────────────────────────────
# 최근 10경기 기록 파싱
# ─────────────────────────────────────────────

def parse_recent10(record):
    """최근10경기 문자열을 승/무/패/승률로 분리한다."""

    record = str(record)                               # 문자열로 변환

    # 예: "6승 1무 3패"에서 각각의 숫자 추출
    win = re.search(r"(\d+)승", record)
    draw = re.search(r"(\d+)무", record)
    loss = re.search(r"(\d+)패", record)

    # 값이 없으면 0으로 처리
    wins = int(win.group(1)) if win else 0
    draws = int(draw.group(1)) if draw else 0
    losses = int(loss.group(1)) if loss else 0

    total = wins + draws + losses                      # 최근 경기 수 합계
    win_rate = wins / total if total > 0 else 0        # 승률 계산

    # DataFrame에 여러 컬럼으로 붙이기 쉽도록 Series 반환
    return pd.Series([wins, draws, losses, win_rate])


# ─────────────────────────────────────────────
# 홈/원정 성적 파싱
# ─────────────────────────────────────────────

def parse_home_away(record):
    """홈/방문 성적 문자열을 승/무/패/승률로 분리한다."""

    record = str(record).strip()                       # 문자열 변환 후 공백 제거

    try:
        # 예: "10-2-5" → 10승 2무 5패
        wins, draws, losses = map(int, record.split("-"))

    except ValueError:
        # 형식이 맞지 않으면 0승 0무 0패로 처리
        wins, draws, losses = 0, 0, 0

    total = wins + draws + losses                      # 전체 경기 수
    win_rate = wins / total if total > 0 else 0        # 승률 계산

    # DataFrame에 여러 컬럼으로 붙이기 쉽도록 Series 반환
    return pd.Series([wins, draws, losses, win_rate])


# ─────────────────────────────────────────────
# 연승/연패 기록 파싱
# ─────────────────────────────────────────────

def parse_streak(streak):
    """연승/연패 문자열을 방향과 개수로 분리한다."""

    streak = str(streak)                               # 문자열로 변환

    # 예: "3연승", "2연패"에서 숫자만 추출
    count = re.search(r"(\d+)", streak)
    count = int(count.group(1)) if count else 0

    # 연승은 1, 연패는 -1, 그 외는 0으로 처리
    if "승" in streak:
        streak_type = 1
    elif "패" in streak:
        streak_type = -1
    else:
        streak_type = 0

    # streak_type: 흐름 방향, count: 연속 경기 수
    return pd.Series([streak_type, count])