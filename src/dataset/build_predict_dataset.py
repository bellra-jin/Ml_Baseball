import pandas as pd

from src.utils.paths import PROCESSED_DIR, MODELING_DIR, make_dirs
from src.utils.config import PREDICT_SEASON


def build_predict_dataset(season=PREDICT_SEASON):
    """
    2026 예측용 데이터셋을 조립한다.

    build_preprocessed.py 실행 후 생성된 processed 파일들을 읽어서 합친다.
    2026 일자별 순위 + 2025 prev_features
    """
    make_dirs()

    current_dir = PROCESSED_DIR / str(season)

    daily_path = current_dir / "team_daily_rank_clean.csv"
    prev_path = current_dir / f"prev_features_from_{season - 1}.csv"

    if not daily_path.exists() or not prev_path.exists():
        print(f"[SKIP] {season} 파일 부족 — build_preprocessed.py를 먼저 실행하세요.")
        return None

    daily_rank = pd.read_csv(daily_path)
    prev_features = pd.read_csv(prev_path)

    predict_dataset = daily_rank.merge(
        prev_features,
        on=["season", "team"],
        how="left",
    )

    predict_dataset.to_csv(
        MODELING_DIR / f"predict_dataset_{season}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"predict_dataset_{season}.csv 저장:", predict_dataset.shape)

    return predict_dataset


if __name__ == "__main__":
    build_predict_dataset()
