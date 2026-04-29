import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from notebooks.experiments.jh.strategy_c_postseason import (
    MODEL_CACHE_VERSION,
    PREDICT_DATASET_PATH,
    TRAIN_DATASET_PATH,
    load_model_and_predict as _load_model_and_predict,
    run_loso_cv_for_app,
)


def _file_cache_key(path):
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _prediction_cache_key():
    return (
        MODEL_CACHE_VERSION,
        _file_cache_key(TRAIN_DATASET_PATH),
        _file_cache_key(PREDICT_DATASET_PATH),
    )


def _validation_cache_key():
    return (
        MODEL_CACHE_VERSION,
        _file_cache_key(TRAIN_DATASET_PATH),
    )


@st.cache_resource(show_spinner="Strategy C 앙상블 모델 학습 중... (첫 실행만 소요됩니다)")
def _cached_load_model_and_predict(cache_key):
    _ = cache_key
    return _load_model_and_predict()


@st.cache_data(show_spinner="LOSO-CV 검증 실행 중... (첫 실행만 소요됩니다)")
def _cached_run_loso_cv(cache_key):
    _ = cache_key
    return run_loso_cv_for_app()


def load_model_and_predict():
    return _cached_load_model_and_predict(_prediction_cache_key())


def run_loso_cv():
    return _cached_run_loso_cv(_validation_cache_key())
