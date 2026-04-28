# src/dataset/build_predict_dataset.py
# 
# 전처리된 processed 데이터를 이용해 예측용 데이터셋을 생성한다.
# 일자별 순위 데이터, 전년도 팀 기록(prev_), 다년도 집계 변수(avg3yr_, trend_)를 병합하고
# 시즌 진행률을 반영한 dyn_ 변수를 추가한 뒤 modeling 폴더에 저장한다.

import pandas as pd

from src.utils.paths import PROCESSED_DIR, MODELING_DIR, make_dirs
from src.utils.config import PREDICT_SEASON, MULTI_YEAR_KEYS


# ─────────────────────────────────────────────
# 예측용 데이터셋 생성
# ─────────────────────────────────────────────

def build_predict_dataset(season=PREDICT_SEASON):
    """
    예측용 데이터셋을 조립한다. 기본값은 2026 시즌.

    build_preprocessed.py 실행 후 생성된 processed 파일들을 읽어서 합친다.

    아래 파일을 머지한다:
      - team_daily_rank_clean.csv        : 일자별 순위·성적
      - prev_features_from_{season-1}.csv: 전년도 팀 기록 (prev_ 변수)
      - multi_year_features_{season}.csv : 3년 평균 (avg3yr_) · 추세 (trend_) 변수

    학습 데이터와 달리 최종 순위 라벨(postseason)은 포함하지 않는다.
    머지 후 dyn_ 변수를 계산해서 추가한다 (build_train_dataset.py와 동일한 로직).
    """

    # 프로젝트에서 사용하는 기본 폴더 생성
    make_dirs()

    # 예측 대상 시즌의 processed 폴더 경로
    current_dir = PROCESSED_DIR / str(season)

    # 예측 시즌의 일자별 순위·성적 파일 경로
    daily_path = current_dir / "team_daily_rank_clean.csv"

    # 직전 시즌 기록을 prev_ 변수로 변환한 파일 경로
    prev_path = current_dir / f"prev_features_from_{season - 1}.csv"

    # 예측 데이터셋 생성에 필요한 필수 파일이 없으면 중단
    if not daily_path.exists() or not prev_path.exists():
        print(f"[SKIP] {season} 파일 부족 — build_preprocessed.py를 먼저 실행하세요.")
        return None

    # 예측 시즌의 일자별 순위·성적 데이터 읽기
    daily_rank = pd.read_csv(daily_path)

    # 직전 시즌 기반 전년도 기록(prev_) 데이터 읽기
    prev_features = pd.read_csv(prev_path)

    # 예측 시즌의 다년도 집계 변수 파일 경로
    multi_path = current_dir / f"multi_year_features_{season}.csv"

    # 다년도 집계 변수 파일이 있으면 읽고, 없으면 None 처리
    multi_year = pd.read_csv(multi_path) if multi_path.exists() else None

    # 일자별 순위 데이터에 전년도 기록(prev_) 병합
    predict_dataset = daily_rank.merge(prev_features, on=["season", "team"], how="left")

    # 다년도 집계 변수(avg3yr_, trend_)가 있으면 추가 병합
    if multi_year is not None:
        predict_dataset = predict_dataset.merge(multi_year, on=["season", "team"], how="left")

    # dyn_ 변수 계산: 시즌 진행도에 따라 3년 평균 비중을 선형으로 줄인다
    dyn_cols = {
        f"dyn_{k}": (1 - predict_dataset["games_played_ratio"]) * predict_dataset[f"avg3yr_{k}"]
        for k in MULTI_YEAR_KEYS
        if f"avg3yr_{k}" in predict_dataset.columns
    }

    # 생성된 dyn_ 변수가 있으면 예측 데이터셋에 한 번에 추가
    if dyn_cols:
        predict_dataset = pd.concat([predict_dataset, pd.DataFrame(dyn_cols, index=predict_dataset.index)], axis=1)

    # 최종 예측용 데이터셋을 modeling 폴더에 CSV로 저장
    predict_dataset.to_csv(
        MODELING_DIR / f"predict_dataset_{season}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 저장된 데이터셋의 파일명과 크기 출력
    print(f"predict_dataset_{season}.csv 저장:", predict_dataset.shape)

    # 이후 예측 코드에서 바로 사용할 수 있도록 DataFrame 반환
    return predict_dataset


# 이 파일을 직접 실행할 때만 예측용 데이터셋 생성
if __name__ == "__main__":
    build_predict_dataset()