"""
2026 KBO 포스트시즌 예측 리포트 실행 엔트리포인트.

실행:
    uv run python "notebooks/experiments/jh/predict_2026_postseason.py"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from notebooks.experiments.jh import strategy_c_report


if __name__ == "__main__":
    strategy_c_report.main()
