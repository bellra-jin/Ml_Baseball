import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from notebooks.experiments.jh.strategy_c_postseason import (
    MODEL_CACHE_VERSION,
    load_model_and_predict as _load_model_and_predict,
    run_loso_cv_for_app,
)


@st.cache_resource(show_spinner="Strategy C 앙상블 모델 학습 중... (첫 실행만 소요됩니다)")
def load_model_and_predict(cache_version: str = MODEL_CACHE_VERSION):
    _ = cache_version
    return _load_model_and_predict()


@st.cache_data(show_spinner="LOSO-CV 검증 실행 중... (첫 실행만 소요됩니다)")
def run_loso_cv(cache_version: str = MODEL_CACHE_VERSION):
    _ = cache_version
    return run_loso_cv_for_app()
