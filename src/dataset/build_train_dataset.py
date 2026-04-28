# src/dataset/build_train_dataset.py
#
# 전처리된 processed 데이터를 이용해 학습용 데이터셋을 생성한다.
# 2017~2025 시즌의 일자별 순위 데이터, 전년도 기록(prev_),
# 다년도 집계 변수(avg3yr_, trend_), 최종 순위 라벨(postseason)을 병합하고
# 시즌 진행률을 반영한 dyn_ 변수를 추가한 뒤 modeling 폴더에 저장한다.

import pandas as pd

from src.utils.paths import PROCESSED_DIR, MODELING_DIR, make_dirs
from src.utils.config import TRAIN_SEASONS, MULTI_YEAR_KEYS


# ─────────────────────────────────────────────
# 학습용 데이터셋 생성
# ─────────────────────────────────────────────

def build_train_dataset():
    """
    2017~2025 학습 데이터셋을 조립한다.

    build_preprocessed.py 실행 후 생성된 processed 파일들을 읽어서 합친다.

    각 시즌별로 아래 파일을 머지한다:
      - team_daily_rank_clean.csv   : 일자별 순위·성적 (행 단위 기준)
      - prev_features_from_{y-1}.csv: 전년도 팀 기록 (prev_ 변수)
      - multi_year_features_{y}.csv : 3년 평균 (avg3yr_) · 추세 (trend_) 변수
      - team_final_rank_clean.csv   : 최종 순위 + postseason 라벨 (0/1)

    머지 후 dyn_ 변수를 계산해서 추가한다:
      dyn_{k} = (1 - games_played_ratio) × avg3yr_{k}
      → 시즌 초반엔 3년 평균 전력이 반영되고, 시즌 후반으로 갈수록 0에 수렴
    """

    # 프로젝트에서 사용하는 기본 폴더 생성
    make_dirs()

    # 시즌별 학습 데이터셋을 담을 리스트
    datasets = []

    # 설정된 학습 대상 시즌을 순서대로 처리
    for year in TRAIN_SEASONS:

        # 해당 시즌의 processed 폴더 경로
        current_dir = PROCESSED_DIR / str(year)

        # 일자별 순위·성적 파일 경로
        daily_path = current_dir / "team_daily_rank_clean.csv"

        # 최종 순위 및 postseason 라벨 파일 경로
        label_path = current_dir / "team_final_rank_clean.csv"

        # 직전 시즌 기록을 prev_ 변수로 변환한 파일 경로
        prev_path = current_dir / f"prev_features_from_{year - 1}.csv"

        # 학습 데이터셋 생성에 필요한 필수 파일이 없으면 해당 시즌 건너뜀
        if not daily_path.exists() or not label_path.exists() or not prev_path.exists():
            print(f"[SKIP] {year} 파일 부족 — build_preprocessed.py를 먼저 실행하세요.")
            continue

        # 해당 시즌의 일자별 순위·성적 데이터 읽기
        daily_rank = pd.read_csv(daily_path)

        # 해당 시즌의 최종 순위 및 postseason 라벨 데이터 읽기
        final_rank = pd.read_csv(label_path)

        # 직전 시즌 기반 전년도 기록(prev_) 데이터 읽기
        prev_features = pd.read_csv(prev_path)

        # multi_year_features는 t-1~t-3 데이터가 모두 있어야 생성되므로
        # 초기 시즌(2017~2019)은 파일이 없을 수 있다 → 없으면 건너뜀
        multi_path = current_dir / f"multi_year_features_{year}.csv"
        multi_year = pd.read_csv(multi_path) if multi_path.exists() else None

        # 일자별 순위 데이터에 전년도 기록(prev_) 병합
        train_dataset = daily_rank.merge(prev_features, on=["season", "team"], how="left")

        # 다년도 집계 변수(avg3yr_, trend_)가 있으면 추가 병합
        if multi_year is not None:
            train_dataset = train_dataset.merge(multi_year, on=["season", "team"], how="left")

        # dyn_ 변수 계산: 시즌 진행도에 따라 3년 평균 비중을 선형으로 줄인다
        # avg3yr_ 컬럼이 없는 시즌(데이터 부족)은 자동으로 건너뜀
        dyn_cols = {
            f"dyn_{k}": (1 - train_dataset["games_played_ratio"]) * train_dataset[f"avg3yr_{k}"]
            for k in MULTI_YEAR_KEYS
            if f"avg3yr_{k}" in train_dataset.columns
        }

        # 생성된 dyn_ 변수가 있으면 학습 데이터셋에 한 번에 추가
        if dyn_cols:
            train_dataset = pd.concat([train_dataset, pd.DataFrame(dyn_cols, index=train_dataset.index)], axis=1)

        # 최종 순위와 postseason 정답 라벨 병합
        train_dataset = train_dataset.merge(
            final_rank[["season", "team", "final_rank", "postseason"]],
            on=["season", "team"],
            how="left",
        )

        # 시즌별 학습 데이터셋을 해당 시즌 processed 폴더에 저장
        train_dataset.to_csv(
            current_dir / f"train_dataset_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 전체 학습 데이터셋 생성을 위해 리스트에 추가
        datasets.append(train_dataset)

        # 저장된 시즌별 데이터셋의 파일명과 크기 출력
        print(f"train_dataset_{year} 저장:", train_dataset.shape)

    # 생성된 학습 데이터셋이 하나도 없으면 None 반환
    if not datasets:
        return None

    # 시즌별 학습 데이터셋을 하나의 DataFrame으로 결합
    train_df = pd.concat(datasets, ignore_index=True)

    # 전체 학습용 데이터셋을 modeling 폴더에 저장
    train_df.to_csv(
        MODELING_DIR / "train_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 저장된 전체 학습 데이터셋의 파일명과 크기 출력
    print("train_dataset.csv 저장:", train_df.shape)

    # 이후 모델 학습 코드에서 바로 사용할 수 있도록 DataFrame 반환
    return train_df


# 이 파일을 직접 실행할 때만 학습용 데이터셋 생성
if __name__ == "__main__":
    build_train_dataset()