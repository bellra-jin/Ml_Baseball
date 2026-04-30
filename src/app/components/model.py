import os
import sys
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from notebooks.experiments.jh.strategy_c_postseason import (
    MODEL_CACHE_VERSION,
    PREDICT_DATASET_PATH,
    TRAIN_DATASET_PATH,
    load_model_and_predict as _load_model_and_predict,
    run_loso_cv_for_app,
)


ROOT = Path(__file__).resolve().parents[3]
APP_OUTPUT_DIR = ROOT / "notebooks/experiments/jh/kbo_prediction_2026"
TIMESERIES_PATH = APP_OUTPUT_DIR / "predict_2026_timeseries.csv"
RANK_PATH = APP_OUTPUT_DIR / "predict_2026_rank.csv"
IMPORTANCE_PATH = APP_OUTPUT_DIR / "predict_2026_importance.csv"
METADATA_PATH = APP_OUTPUT_DIR / "predict_2026_metadata.json"


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_cache_key(path):
    if not path.exists():
        return str(path), None, None
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _prediction_cache_key():
    return (
        MODEL_CACHE_VERSION,
        _file_cache_key(TRAIN_DATASET_PATH),
        _file_cache_key(PREDICT_DATASET_PATH),
        _file_cache_key(TIMESERIES_PATH),
        _file_cache_key(RANK_PATH),
        _file_cache_key(IMPORTANCE_PATH),
        _file_cache_key(METADATA_PATH),
    )


def _validation_cache_key():
    return (
        MODEL_CACHE_VERSION,
        _file_cache_key(TRAIN_DATASET_PATH),
    )


def _read_artifact_metadata():
    if not METADATA_PATH.exists():
        return None
    with METADATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _artifact_files_exist():
    return all(
        path.exists()
        for path in [TIMESERIES_PATH, RANK_PATH, IMPORTANCE_PATH, METADATA_PATH]
    )


def _artifacts_match_current_data(metadata):
    return (
        metadata is not None
        and metadata.get("model_cache_version") == MODEL_CACHE_VERSION
        and metadata.get("train_sha256") == _sha256_file(TRAIN_DATASET_PATH)
        and metadata.get("predict_sha256") == _sha256_file(PREDICT_DATASET_PATH)
    )


def _load_prediction_artifacts():
    """
    로컬에서 생성해 커밋한 예측 산출물을 우선 사용한다.

    Streamlit Cloud 환경에서 모델을 다시 학습하면 패키지/OS 차이로 피처 중요도가 달라질 수 있어,
    metadata가 현재 modeling CSV와 일치할 때는 저장된 산출물을 읽어 로컬/배포 화면을 고정한다.
    """
    if not _artifact_files_exist():
        return None

    metadata = _read_artifact_metadata()
    if not _artifacts_match_current_data(metadata):
        return None

    pred_df = pd.read_csv(TIMESERIES_PATH)
    rank_df = pd.read_csv(RANK_PATH)
    importance_df = pd.read_csv(IMPORTANCE_PATH)

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    rank_df["date"] = pd.to_datetime(rank_df["date"])

    importance = pd.Series(
        importance_df["importance"].values,
        index=importance_df["feature"].values,
    )
    feature_cols = metadata.get("feature_cols", list(importance.index))
    return pred_df, rank_df, importance, feature_cols


@st.cache_resource(show_spinner="Strategy C 앙상블 모델 학습 중... (첫 실행만 소요됩니다)")
def _cached_load_model_and_predict(cache_key):
    _ = cache_key
    artifact_result = _load_prediction_artifacts()
    if artifact_result is not None:
        return artifact_result
    return _load_model_and_predict()


@st.cache_data(show_spinner="LOSO-CV 검증 실행 중... (첫 실행만 소요됩니다)")
def _cached_run_loso_cv(cache_key):
    _ = cache_key
    return run_loso_cv_for_app()


def load_model_and_predict():
    return _cached_load_model_and_predict(_prediction_cache_key())


def run_loso_cv():
    return _cached_run_loso_cv(_validation_cache_key())
