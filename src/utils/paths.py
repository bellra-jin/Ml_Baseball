from pathlib import Path


# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parents[2]

# 데이터 경로
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELING_DIR = DATA_DIR / "modeling"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# 결과물 경로
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"


# 폴더 자동 생성
def make_dirs():
    """프로젝트에서 사용하는 주요 폴더를 자동 생성한다."""
    dirs = [
        RAW_DIR,
        PROCESSED_DIR,
        MODELING_DIR,
        PREDICTIONS_DIR,
        MODEL_DIR,
        REPORT_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)