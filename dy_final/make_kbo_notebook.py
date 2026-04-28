# -*- coding: utf-8 -*-
"""Generate the KBO preprocessing and insight notebook without nbformat."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "KBO_가을야구_예측_전처리_심층분석.ipynb"


def md(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md(
        """
        # KBO 2022~2026 데이터 전처리와 2026 가을야구 예측

        목적은 KBO 정규시즌 데이터만 사용해 2026년 현재 시점에서 가을야구 진출 가능성이 높은 팀을 예측하는 것입니다.

        - 공식 데이터 출처: https://www.koreabaseball.com
        - 분석 데이터 경로: `C:/Users/Admin/Documents/GitHub/Ml_Baseball/data`
        - 예측 기준: 2026년 로컬 `팀_일자별순위.csv`의 최신 날짜
        - 타깃 라벨: 최종 순위 5위 이내 여부

        중요한 점은 누수 방지입니다. 훈련에는 “당일 일자별 순위 스냅샷”과 “전년도 팀/선수 전력 지표”만 넣고, 해당 시즌 최종 팀기록은 타깃 라벨 생성 외에는 쓰지 않습니다.
        """
    ),
    md(
        """
        ## 데이터로 가능한 인사이트와 추가 크롤링 필요성

        현재 있는 2022~2026 CSV만으로도 가을야구 예측을 위한 기본 분석은 가능합니다.

        가능한 분석:
        - 4월 순위와 최종 5강의 관계
        - 현재 승률, 게임차, 최근 10경기 흐름 기반의 실시간 예측
        - 전년도 팀 타격/투수/수비/주루 지표가 다음 시즌 성적에 주는 신호
        - 전년도 핵심 타자/투수 상위권 집계와 다음 시즌 5강 여부의 관계

        추가 크롤링이 필요한 분석:
        - 선수 프로필/통산/일자별/경기별/상황별 기록: 선수별 컨디션, 출전 지속성, 특정 상황 강점 분석
        - 선수 이동 현황: 2026년 전력 변화 보정
        - 팀 세부기록: OPS, 장타/출루, 투수 세부지표를 통한 설명력 강화
        """
    ),
    code(
        r"""
        from pathlib import Path
        import sys
        import pandas as pd

        PROJECT_DIR = Path(r"C:\Users\Admin\Documents\Codex\2026-04-28\files-mentioned-by-the-user-data")
        if str(PROJECT_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECT_DIR))

        import kbo_postseason_pipeline as pipe

        DATA_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\data")
        OUT_DIR = Path(r"C:\Users\Admin\Documents\Codex\2026-04-28\files-mentioned-by-the-user-data\kbo_outputs")

        artifacts = pipe.run_pipeline(DATA_DIR, OUT_DIR)
        artifacts["summary"]
        """
    ),
    md(
        """
        ## 전처리 과정

        파이프라인에서 수행하는 핵심 전처리입니다.

        - `YYYYMMDD` 형태 날짜를 datetime으로 변환
        - `최근10경기` 문자열에서 승/무/패와 최근 승률 추출
        - `홈`, `방문` 문자열에서 홈/원정 승률 추출
        - `IP`의 `5 1/3`, `5.1` 같은 이닝 표기를 소수 이닝으로 변환
        - 팀별 최종 순위 5위 이내를 `postseason` 라벨로 생성
        - 2023~2026 각 시즌에 대해 직전 시즌 팀/선수 피처를 붙임
        - 2026은 라벨 없이 최신 스냅샷만 예측 대상으로 사용
        """
    ),
    code(
        """
        model_df = artifacts["model_dataset"]
        model_df[[
            "season", "date", "team", "rank", "games", "win_rate", "games_behind",
            "recent10_win_rate", "home_win_rate", "away_win_rate",
            "prev_final_rank", "prev_avg", "prev_era", "prev_whip", "postseason"
        ]].head(12)
        """
    ),
    md(
        """
        ## 2026 가을야구 예측 결과

        `모델확률`은 과거 일자별 순위와 전년도 전력 지표로 학습한 로지스틱 모델의 출력입니다.
        `스탠딩확률`은 현재 순위, 승률, 게임차, 최근 10경기를 반영한 안정화 점수입니다.
        최종 `가을야구확률`은 두 값을 섞은 앙상블이며, 현재처럼 4월 데이터만 있는 상황에서 과적합을 줄이기 위한 장치입니다.
        """
    ),
    code(
        """
        pred = pd.read_csv(OUT_DIR / "2026_postseason_predictions.csv", encoding="utf-8-sig")
        pred[[
            "team", "rank", "games", "wins", "losses", "win_rate", "games_behind",
            "model_probability_pct", "standing_probability_pct",
            "postseason_probability_pct", "prediction_label"
        ]]
        """
    ),
    md(
        """
        ## 4월 순위만으로 어디까지 맞나

        2023~2025년의 4월 마지막 스냅샷을 기준으로 보면, 당시 5위권 중 최종 5강에 남은 팀은 매년 3팀이었습니다.
        즉 4월 순위는 의미 있는 신호지만 그대로 확정값처럼 쓰기에는 이릅니다.
        """
    ),
    code(
        """
        pd.read_csv(OUT_DIR / "april_rank_insight.csv", encoding="utf-8-sig")
        """
    ),
    md(
        """
        ## 시즌 홀드아웃 검증

        한 시즌을 테스트로 빼고 나머지 시즌으로 학습했습니다. 표본 시즌이 2023~2025 세 시즌뿐이므로, 이 수치는 모델의 방향성을 보는 용도입니다.
        """
    ),
    code(
        """
        pd.read_csv(OUT_DIR / "validation_leave_one_season.csv", encoding="utf-8-sig")
        """
    ),
    md(
        """
        ## 모델이 민감하게 본 피처

        표본 시즌이 적어서 계수의 부호를 인과로 해석하면 안 됩니다. 그래도 어떤 신호에 모델이 민감했는지 점검하는 용도로 사용합니다.
        """
    ),
    code(
        """
        pd.read_csv(OUT_DIR / "feature_importance_coefficients.csv", encoding="utf-8-sig").head(15)
        """
    ),
    md(
        """
        ## 2026 팀 스냅샷

        아래 테이블은 최신 2026 팀 순위와 현재까지의 팀 타자/투수/수비/주루 기본기록을 붙인 분석용 테이블입니다.
        예측 모델에는 누수 문제 때문에 같은 시즌 최종 팀기록을 훈련 피처로 쓰지 않았지만, 현재 시즌 상태 설명에는 유용합니다.
        """
    ),
    code(
        """
        snapshot = pd.read_csv(OUT_DIR / "2026_team_snapshot.csv", encoding="utf-8-sig")
        snapshot[[
            "team", "rank", "games", "win_rate", "games_behind",
            "current_runs_per_game", "current_hr_per_game",
            "current_era", "current_whip", "current_errors_per_game", "current_sb_rate"
        ]].sort_values("rank")
        """
    ),
    md(
        """
        ## 실시간 크롤링 코드 실행

        같은 폴더에 `kbo_realtime_crawler.py`를 생성했습니다. KBO 공식 사이트의 ASP.NET PostBack 구조를 따라 연도 선택, 페이지 이동, 팀 일자별 순위 이동을 처리합니다.

        기본 실행:
        ```powershell
        python kbo_realtime_crawler.py --year 2026 --data-dir "C:\\Users\\Admin\\Documents\\GitHub\\Ml_Baseball\\data"
        ```

        터미널을 계속 켜두고 매일 00:00 KST에 실행:
        ```powershell
        python kbo_realtime_crawler.py --year 2026 --watch-midnight
        ```

        Windows 작업 스케줄러 등록:
        ```powershell
        .\\register_kbo_midnight_task.ps1 -PythonExe "python"
        ```

        선수 프로필 전체 수집은 시간이 오래 걸릴 수 있으므로 필요할 때만:
        ```powershell
        python kbo_realtime_crawler.py --year 2026 --profiles
        ```
        """
    ),
    code(
        """
        from pathlib import Path

        for path in [
            Path("kbo_postseason_pipeline.py"),
            Path("kbo_realtime_crawler.py"),
            Path("register_kbo_midnight_task.ps1"),
            OUT_DIR / "analysis_report.md",
            OUT_DIR / "2026_postseason_predictions.csv",
        ]:
            print(path.resolve())
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(NOTEBOOK_PATH)
