# ─────────────────────────────────────────────
# Strategy C 포스트시즌 예측 핵심 모듈
# ─────────────────────────────────────────────
"""
Strategy C 포스트시즌 예측 핵심 모듈.

원본 `predict_2026_postseason copy.py`에서 학습/검증/예측 로직만 분리했다.
차트 저장과 리포트 실행 순서는 다른 모듈에서 담당하고, 이 파일은 아래 역할에 집중한다.

- 데이터 로드: train_dataset.csv, predict_dataset_2026.csv
- 모델 구성: LR 25% + RF 25% + lightXGB 25% + lightLGBM 25%
- 검증: LOSO-CV로 시즌별 성능과 out-of-fold 예측 생성
- 예측: 2026 시즌 팀별 포스트시즌 진출 확률 산출
- 앱 연결: Streamlit 앱에서 재사용할 반환 형태 제공
"""

from dataclasses import dataclass
from pathlib import Path

import lightgbm as _lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluation.metrics import checkpoint_hits, evaluate_binary_model, print_metrics
from src.utils.config import TOP_FEATURES as CONFIG_TOP_FEATURES


# ─────────────────────────────────────────────
# 경로 및 공통 설정
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATASET_PATH = ROOT / "data/modeling/train_dataset.csv"
PREDICT_DATASET_PATH = ROOT / "data/modeling/predict_dataset_2026.csv"
MODEL_CACHE_VERSION = "strategy_c_20_features_v3"
TEST_SEASONS = list(range(2017, 2026))


# ─────────────────────────────────────────────
# Strategy C 모델 컨테이너
# ─────────────────────────────────────────────
@dataclass
class StrategyCModel:
    """Strategy C 앙상블을 하나로 묶어 다루기 위한 얇은 컨테이너."""

    lr: Pipeline
    rf: RandomForestClassifier
    xgb: XGBClassifier
    lgbm: LGBMClassifier
    feature_cols: list[str]

    # ─────────────────────────────────────────
    # 예측 확률 계산
    # ─────────────────────────────────────────
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """4개 모델의 양성 클래스 확률을 동일 가중 평균으로 앙상블한다."""
        return ensemble_proba(self.lr, self.rf, self.xgb, self.lgbm, X[self.feature_cols])

    # ─────────────────────────────────────────
    # 피처 중요도 계산
    # ─────────────────────────────────────────
    def feature_importance(self) -> pd.Series:
        """트리 계열 3개 모델의 feature_importances_를 정규화 후 평균낸다."""
        imp_xgb = pd.Series(self.xgb.feature_importances_, index=self.feature_cols)
        imp_lgbm = pd.Series(self.lgbm.feature_importances_, index=self.feature_cols)
        imp_rf = pd.Series(self.rf.feature_importances_, index=self.feature_cols)
        return (
            imp_xgb / imp_xgb.sum()
            + imp_lgbm / imp_lgbm.sum()
            + imp_rf / imp_rf.sum()
        ) / 3


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
def load_datasets(
    train_path: Path | str = TRAIN_DATASET_PATH,
    predict_path: Path | str = PREDICT_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    모델 학습용 데이터와 2026 예측용 데이터를 읽는다.

    원본 스크립트의 "데이터 로드" 구간을 함수화했다.
    downstream 코드에서 날짜 정렬/최신 시점 추출을 바로 할 수 있도록 date 컬럼을 datetime으로 변환한다.
    """
    train_df = pd.read_csv(train_path)
    pred_df = pd.read_csv(predict_path)
    train_df["date"] = pd.to_datetime(train_df["date"])
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    return train_df, pred_df


# ─────────────────────────────────────────────
# 사용 피처 확정
# ─────────────────────────────────────────────
def resolve_feature_cols(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame | None = None,
    feature_cols: list[str] | tuple[str, ...] = CONFIG_TOP_FEATURES,
) -> list[str]:
    """
    설정된 TOP_FEATURES 중 실제 데이터에 존재하는 컬럼만 사용한다.

    학습 데이터와 예측 데이터의 컬럼 차이로 실행이 깨지지 않도록,
    예측 데이터가 주어지면 양쪽에 모두 있는 피처만 남긴다.
    """
    cols = [c for c in feature_cols if c in train_df.columns]
    if pred_df is not None:
        cols = [c for c in cols if c in pred_df.columns]
    return cols


# ─────────────────────────────────────────────
# 모델 빌더
# ─────────────────────────────────────────────
def build_models(pos_w: float):
    """
    Strategy C를 구성하는 4개 모델을 생성한다.

    원본 스크립트의 "_build_models"와 같은 설정이다.
    - LR: 결측치 대체 + 표준화 후 balanced LogisticRegression
    - RF: 얕은 트리와 큰 leaf로 과적합 완화
    - XGB/LGBM: 적은 boosting round와 강한 regularization으로 light 모델 구성
    """
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1,
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=40,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=5.0,
        reg_lambda=10.0,
        min_child_weight=30,
        scale_pos_weight=pos_w,
        eval_metric="logloss",
        random_state=42,
    )
    lgbm = LGBMClassifier(
        n_estimators=40,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=5.0,
        reg_lambda=10.0,
        min_child_samples=50,
        scale_pos_weight=pos_w,
        random_state=42,
        verbose=-1,
    )
    return lr, rf, xgb, lgbm


# ─────────────────────────────────────────────
# 앙상블 확률 계산
# ─────────────────────────────────────────────
def ensemble_proba(lr, rf, xgb, lgbm, X: pd.DataFrame) -> np.ndarray:
    """LR, RF, XGB, LGBM의 postseason=1 확률을 25%씩 평균한다."""
    return (
        lr.predict_proba(X)[:, 1]
        + rf.predict_proba(X)[:, 1]
        + xgb.predict_proba(X)[:, 1]
        + lgbm.predict_proba(X)[:, 1]
    ) / 4


# ─────────────────────────────────────────────
# 시즌 가중치 계산
# ─────────────────────────────────────────────
def season_sample_weight(seasons: pd.Series) -> np.ndarray:
    """
    최근 시즌에 더 큰 가중치를 주는 선형 sample weight를 만든다.

    가장 오래된 시즌은 0.3, 가장 최신 시즌은 1.0에 가깝게 반영한다.
    시즌별 KBO 환경 변화가 모델에 반영되도록 원본 스크립트에서 사용한 방식이다.
    """
    s_min, s_max = seasons.min(), seasons.max()
    return (0.3 + 0.7 * (seasons - s_min) / max(s_max - s_min, 1)).values


# ─────────────────────────────────────────────
# Strategy C 모델 학습
# ─────────────────────────────────────────────
def fit_strategy_c(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    with_eval_set: bool = False,
) -> StrategyCModel:
    """
    주어진 학습 데이터로 Strategy C 앙상블을 학습한다.

    with_eval_set=True이면 LOSO-CV 검증 차트용 loss curve를 남기기 위해
    가장 최신 학습 시즌을 eval_set으로 넣는다. 최종 예측 학습에서는 loss curve가 필요 없으므로 False로 둔다.
    """
    X_train = train_df[feature_cols]
    y_train = train_df["postseason"]
    sample_weight = season_sample_weight(train_df["season"])
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    lr, rf, xgb, lgbm = build_models(pos_w)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train, sample_weight=sample_weight)

    if with_eval_set:
        # loss curve 저장용 eval_set. 학습에 포함된 최신 시즌을 validation 이름으로 추적한다.
        last_season = train_df["season"].max()
        X_es = train_df.loc[train_df["season"] == last_season, feature_cols]
        y_es = train_df.loc[train_df["season"] == last_season, "postseason"]
        xgb.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_es, y_es)],
            verbose=False,
        )
        lgbm.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_es, y_es)],
            eval_names=["train", "valid"],
            eval_metric="binary_logloss",
            callbacks=[_lgb.log_evaluation(period=0)],
        )
    else:
        xgb.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        lgbm.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            callbacks=[_lgb.log_evaluation(period=0)],
        )

    return StrategyCModel(lr, rf, xgb, lgbm, feature_cols)


# ─────────────────────────────────────────────
# LOSO-CV 검증
# ─────────────────────────────────────────────
def run_loso_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    test_seasons: list[int] = TEST_SEASONS,
    *,
    include_losses: bool = False,
    print_each_fold: bool = False,
) -> dict[str, object]:
    """
    Leave-One-Season-Out CV를 수행한다.

    각 시즌을 한 번씩 test fold로 빼고 나머지 시즌으로 학습한다.
    반환값은 리포트/차트 모듈에서 바로 쓸 수 있도록 성능표, 전체 OOF 예측,
    실제 postseason 진출팀 목록, 선택적으로 boosting loss 기록을 포함한다.
    """
    cv_records = []
    cv_rows = []
    xgb_losses = {}
    lgbm_losses = {}

    for test_season in test_seasons:
        # 해당 시즌만 검증 세트로 두고, 나머지 시즌 전체로 학습한다.
        tr = train_df[train_df["season"] != test_season].copy()
        te = train_df[train_df["season"] == test_season].copy()

        model = fit_strategy_c(tr, feature_cols, with_eval_set=include_losses)
        prob_te = model.predict_proba(te)
        prob_tr = model.predict_proba(tr)

        if include_losses:
            xgb_losses[test_season] = model.xgb.evals_result()
            lgbm_losses[test_season] = model.lgbm.evals_result_

        # threshold 기반 지표와 ranking/확률 지표를 함께 저장한다.
        m_te = evaluate_binary_model(te["postseason"], prob_te)
        m_tr = evaluate_binary_model(tr["postseason"], prob_tr)
        cv_records.append({
            "season": test_season,
            "test_auc": m_te["roc_auc"],
            "train_auc": m_tr["roc_auc"],
            "gap": m_tr["roc_auc"] - m_te["roc_auc"],
            "brier": m_te["brier"],
            "f1": m_te["f1"],
            "precision": m_te["precision"],
            "recall": m_te["recall"],
            "accuracy": m_te["accuracy"],
        })

        row_pred = te[["season", "date", "team", "postseason", "games_played_ratio"]].copy()
        row_pred["prob"] = prob_te
        cv_rows.append(row_pred)

        if print_each_fold:
            print_metrics(m_te, label=str(test_season))

    all_rows = pd.concat(cv_rows, ignore_index=True)

    # 체크포인트 적중률 차트에서 쓸 시즌별 실제 포스트시즌 진출팀.
    actual_top5 = {}
    for season in test_seasons:
        labels = train_df[train_df["season"] == season].groupby("team")["postseason"].first()
        actual_top5[season] = set(labels[labels == 1].index)

    result = {
        "metrics": pd.DataFrame(cv_records),
        "all_rows": all_rows,
        "actual_top5": actual_top5,
    }
    if include_losses:
        result["xgb_losses"] = xgb_losses
        result["lgbm_losses"] = lgbm_losses
    return result


# ─────────────────────────────────────────────
# 2026 포스트시즌 확률 예측
# ─────────────────────────────────────────────
def predict_2026(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str], StrategyCModel]:
    """
    전체 학습 데이터로 최종 모델을 학습하고 2026 예측 확률을 계산한다.

    prob_raw는 4개 모델 평균 확률이고, prob_norm은 같은 날짜의 10개 팀 확률 합이
    포스트시즌 티켓 수(5)에 맞도록 정규화한 값이다. 원본 스크립트의 예측 로직과 같다.
    """
    model = fit_strategy_c(train_df, feature_cols)
    pred_df = pred_df.copy()
    pred_df["prob_raw"] = model.predict_proba(pred_df)
    pred_df["prob_norm"] = pred_df.groupby("date")["prob_raw"].transform(
        lambda x: (x / x.sum() * 5).clip(upper=1.0)
    )
    rank_df = (
        pred_df
        .sort_values(["date", "prob_norm"], ascending=[True, False])
        .assign(pred_rank=lambda d: d.groupby("date").cumcount() + 1)
    )

    # 팀별 최신 row만 뽑아 현재 기준 예측 순위와 top5 팀 목록을 만든다.
    latest = pred_df.sort_values("date").groupby("team").last().reset_index()
    latest = latest.sort_values("prob_norm", ascending=False)
    top5_teams = set(latest.head(5)["team"])
    return pred_df, rank_df, latest, top5_teams, model


# ─────────────────────────────────────────────
# Streamlit 앱용 예측 진입점
# ─────────────────────────────────────────────
def load_model_and_predict() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """
    Streamlit 앱에서 쓰는 학습+예측 진입점.

    앱은 pred_df, 날짜별 rank_df, feature importance, 실제 사용 피처 목록만 필요하므로
    최종 모델 객체는 내부에서만 사용하고 반환하지 않는다.
    """
    train_df, pred_df = load_datasets()
    feature_cols = resolve_feature_cols(train_df, pred_df)
    pred_df, rank_df, _, _, model = predict_2026(train_df, pred_df, feature_cols)
    return pred_df, rank_df, model.feature_importance(), feature_cols


# ─────────────────────────────────────────────
# Streamlit 앱용 검증 진입점
# ─────────────────────────────────────────────
def run_loso_cv_for_app() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Streamlit 검증 리포트 페이지에서 쓰는 LOSO-CV 진입점.

    차트 컴포넌트가 요구하는 형태에 맞춰 metrics_df, OOF 확률, OOF 라벨,
    checkpoint hit DataFrame을 반환한다.
    """
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    train_df["date"] = pd.to_datetime(train_df["date"])
    feature_cols = resolve_feature_cols(train_df)
    test_seasons = sorted(s for s in train_df["season"].unique() if s >= 2018)
    cv_result = run_loso_cv(train_df, feature_cols, test_seasons)

    checkpoints = {"50%": 0.50, "75%": 0.75, "90%": 0.90, "최종": 1.01}
    cp_rows = []
    for season in test_seasons:
        # 각 체크포인트에서 예측 상위 5팀이 실제 postseason 팀과 얼마나 겹치는지 계산한다.
        season_rows = cv_result["all_rows"][cv_result["all_rows"]["season"] == season]
        cp_result = checkpoint_hits(
            season_rows,
            "prob",
            cv_result["actual_top5"][season],
            checkpoints,
        )
        for label, values in cp_result.items():
            cp_rows.append({
                "season": int(season),
                "checkpoint": label,
                "hit": values["hit"],
            })

    all_rows = cv_result["all_rows"]
    return (
        cv_result["metrics"],
        np.array(all_rows["prob"]),
        np.array(all_rows["postseason"]),
        pd.DataFrame(cp_rows),
    )
