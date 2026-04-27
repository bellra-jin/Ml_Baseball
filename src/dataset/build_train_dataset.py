import pandas as pd

from src.utils.paths import PROCESSED_DIR, MODELING_DIR, make_dirs
from src.utils.config import TRAIN_SEASONS


def build_train_dataset():
    """
    2023~2025 학습 데이터셋을 조립한다.

    build_preprocessed.py 실행 후 생성된 processed 파일들을 읽어서 합친다.
    각 시즌: 일자별 순위 + 전년도 prev_features + 최종 postseason 라벨
    """
    make_dirs()

    datasets = []

    for year in TRAIN_SEASONS:
        current_dir = PROCESSED_DIR / str(year)
        prev_dir = PROCESSED_DIR / str(year - 1)

        daily_path = current_dir / "team_daily_rank_clean.csv"
        label_path = current_dir / "team_final_rank_clean.csv"
        prev_path = prev_dir / f"prev_features_for_{year}.csv"

        if not daily_path.exists() or not label_path.exists() or not prev_path.exists():
            print(f"[SKIP] {year} 파일 부족 — build_preprocessed.py를 먼저 실행하세요.")
            continue

        daily_rank = pd.read_csv(daily_path)
        final_rank = pd.read_csv(label_path)
        prev_features = pd.read_csv(prev_path)

        train_dataset = (
            daily_rank
            .merge(prev_features, on=["season", "team"], how="left")
            .merge(
                final_rank[["season", "team", "final_rank", "postseason"]],
                on=["season", "team"],
                how="left",
            )
        )

        train_dataset.to_csv(
            current_dir / f"train_dataset_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        datasets.append(train_dataset)
        print(f"train_dataset_{year} 저장:", train_dataset.shape)

    if not datasets:
        return None

    train_df = pd.concat(datasets, ignore_index=True)

    train_df.to_csv(
        MODELING_DIR / "train_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("train_dataset.csv 저장:", train_df.shape)

    return train_df


if __name__ == "__main__":
    build_train_dataset()
