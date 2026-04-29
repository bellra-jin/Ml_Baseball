from pathlib import Path


# ─────────────────────────────────────────────
# 프로젝트 기준 경로 설정
# ─────────────────────────────────────────────

# 프로젝트 루트 경로
# 현재 파일 위치 기준으로 상위 2단계 폴더를 프로젝트 루트로 사용
BASE_DIR = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────
# 데이터 저장 경로 설정
# ─────────────────────────────────────────────

# 데이터 전체 폴더 경로
DATA_DIR = BASE_DIR / "data"

# 원본 CSV 데이터 저장 폴더
RAW_DIR = DATA_DIR / "raw"

# 전처리 완료 데이터 저장 폴더
PROCESSED_DIR = DATA_DIR / "processed"

# 모델 학습용 데이터 저장 폴더
MODELING_DIR = DATA_DIR / "modeling"

# 예측 결과 CSV 저장 폴더
PREDICTIONS_DIR = DATA_DIR / "predictions"


# ─────────────────────────────────────────────
# 결과물 저장 경로 설정
# ─────────────────────────────────────────────

# 모델, 리포트 등 산출물 전체 폴더
REPORT_DIR = ARTIFACTS_DIR / "reports"

# 학습된 모델 파일 저장 폴더
MODEL_DIR = ARTIFACTS_DIR / "models"

# 평가 결과, 그래프, 리포트 저장 폴더
REPORT_DIR = ARTIFACTS_DIR / "reports"


# ─────────────────────────────────────────────
# 주요 폴더 자동 생성 함수
# ─────────────────────────────────────────────

# 폴더 자동 생성
def make_dirs():
    """프로젝트에서 사용하는 주요 폴더를 자동 생성한다."""

    # 프로젝트 실행에 필요한 주요 폴더 목록
    dirs = [
        RAW_DIR,
        PROCESSED_DIR,
        MODELING_DIR,
        PREDICTIONS_DIR,
        MODEL_DIR,
        REPORT_DIR,
    ]

    # 폴더가 없으면 생성하고, 이미 있으면 그대로 둔다
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)