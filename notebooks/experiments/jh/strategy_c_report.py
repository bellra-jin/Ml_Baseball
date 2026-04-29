# ─────────────────────────────────────────────
# Strategy C 리포트 실행 오케스트레이터
# ─────────────────────────────────────────────
"""
Strategy C 2026 KBO 포스트시즌 예측 리포트 오케스트레이터.

원본 `predict_2026_postseason copy.py`의 전체 실행 순서를 유지하되,
세부 구현은 아래 모듈들로 분리했다.

- `strategy_c_postseason.py`: 데이터 로드, 학습, LOSO-CV, 2026 예측
- `strategy_c_validation_charts.py`: 검증 차트 V1~V5 저장
- `strategy_c_prediction_outputs.py`: 예측 결과 CSV와 예측 차트 7종 저장
- `strategy_c_style.py`: Matplotlib 폰트/공통 스타일 설정

이 파일은 실행 순서만 조율한다.
데이터 로드 -> LOSO-CV 검증 -> 검증 차트 저장 -> 최종 학습/예측 -> 예측 산출물 저장
"""

from pathlib import Path

from notebooks.experiments.jh.strategy_c_postseason import (
    TEST_SEASONS,
    load_datasets,
    predict_2026,
    resolve_feature_cols,
    run_loso_cv,
)
from notebooks.experiments.jh.strategy_c_prediction_outputs import save_prediction_outputs
from notebooks.experiments.jh.strategy_c_style import setup_matplotlib
from notebooks.experiments.jh.strategy_c_validation_charts import save_validation_charts


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parents[2]
OUTDIR = BASE_DIR / "kbo_prediction_2026"
VALDIR = OUTDIR / "validation"


# ─────────────────────────────────────────────
# 입력 데이터 준비
# ─────────────────────────────────────────────
def load_inputs():
    """
    학습 데이터와 2026 예측 데이터를 읽고 실제 사용 가능한 피처 목록을 확정한다.

    원본 스크립트의 "데이터 로드" 구간에 해당한다.
    출력 로그는 기존 실행 결과와 비슷하게 유지해 리포트 실행 상태를 확인하기 쉽게 한다.
    """
    print("데이터 로드 중...")
    train_df, pred_df = load_datasets()
    feature_cols = resolve_feature_cols(train_df, pred_df)

    print(f"학습 데이터: {train_df.shape}  ({sorted(train_df['season'].unique())})")
    print(f"예측 데이터: {pred_df.shape}")
    print(f"예측 기간:   {pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
    print(f"시즌 진행도: {pred_df['games_played_ratio'].max():.1%}")
    print(f"사용 피처:   {len(feature_cols)}개  (Strategy C)\n")
    return train_df, pred_df, feature_cols


# ─────────────────────────────────────────────
# LOSO-CV 검증 실행
# ─────────────────────────────────────────────
def validate_model(train_df, feature_cols):
    """
    LOSO-CV 검증을 실행하고 검증 결과 묶음을 반환한다.

    include_losses=True로 실행해서 XGBoost/LightGBM loss curve 차트에 필요한
    boosting round별 logloss 기록도 함께 받는다.
    """
    print("=" * 55)
    print("LOSO-CV 검증  (Strategy C | 20 피처 | 2017~2025)")
    print("=" * 55)

    cv_result = run_loso_cv(
        train_df,
        feature_cols,
        TEST_SEASONS,
        include_losses=True,
        print_each_fold=True,
    )
    cv_df = cv_result["metrics"]
    gap_mean = cv_df["gap"].mean()
    print(f"\n  Test AUC  = {cv_df['test_auc'].mean():.4f}  |  "
          f"Train AUC = {cv_df['train_auc'].mean():.4f}  |  "
          f"갭 = {gap_mean:.4f}  |  Brier = {cv_df['brier'].mean():.4f}\n")
    return cv_result


# ─────────────────────────────────────────────
# 최종 학습 및 2026 예측
# ─────────────────────────────────────────────
def train_and_predict(train_df, pred_df, feature_cols):
    """
    전체 학습 데이터로 최종 Strategy C 모델을 학습하고 2026 확률을 예측한다.

    `predict_2026()`은 예측 row 전체, 최신 기준 순위표, top5 팀 목록, 최종 모델 객체를 반환한다.
    최종 모델 객체는 이후 피처 중요도 차트에서 다시 사용된다.
    """
    print("=" * 55)
    print("최종 모델 학습 (2017~2025 전체)")
    print("=" * 55)

    pred_df, _, latest, top5_teams, final_model = predict_2026(
        train_df,
        pred_df,
        feature_cols,
    )

    print("  LogisticRegression 완료")
    print("  RandomForest 완료")
    print("  XGBoost 완료")
    print("  LightGBM 완료\n")
    return pred_df, latest, top5_teams, final_model


# ─────────────────────────────────────────────
# 최신 예측 결과 출력
# ─────────────────────────────────────────────
def print_latest_prediction(latest):
    """터미널에서 최신 기준 예측 순위를 빠르게 확인할 수 있도록 출력한다."""
    print("=" * 50)
    print(f"[2026 포스트시즌 예측] 기준: {latest['date'].iloc[0].date()}")
    print("=" * 50)
    print(f"{'순위':<4} {'팀':<8} {'확률':>8}  {'진행도':>6}")
    print("─" * 35)
    for i, row in enumerate(latest.itertuples(), 1):
        marker = "★" if i <= 5 else "  "
        print(f"{marker}{i:>2}위  {row.team:<8} {row.prob_norm:>7.1%}  ({row.games_played_ratio:.1%})")
    print()


# ─────────────────────────────────────────────
# 리포트 전체 실행
# ─────────────────────────────────────────────
def main():
    """
    리포트 전체 실행 진입점.

    실행 산출물:
    - `kbo_prediction_2026/validation/val_*.png`
    - `kbo_prediction_2026/predict_2026_*.png`
    - `kbo_prediction_2026/predict_2026_result.csv`
    """
    setup_matplotlib()
    OUTDIR.mkdir(exist_ok=True)
    VALDIR.mkdir(exist_ok=True)

    # 1. 입력 데이터 준비
    train_df, pred_df, feature_cols = load_inputs()

    # 2. 시즌 단위 LOSO-CV 검증
    cv_result = validate_model(train_df, feature_cols)

    # 3. 검증 차트 5종 저장
    save_validation_charts(
        cv_result["metrics"],
        cv_result["all_rows"],
        cv_result["xgb_losses"],
        cv_result["lgbm_losses"],
        cv_result["actual_top5"],
        VALDIR,
    )

    # 4. 전체 학습 데이터로 최종 모델 학습 후 2026 예측
    pred_df, latest, top5_teams, final_model = train_and_predict(
        train_df,
        pred_df,
        feature_cols,
    )
    print_latest_prediction(latest)

    # 5. 예측 결과 CSV와 예측 차트 7종 저장
    save_prediction_outputs(
        train_df,
        pred_df,
        latest,
        top5_teams,
        final_model,
        feature_cols,
        ROOT,
        OUTDIR,
    )

    print("\n완료.")
    print(f"  검증 차트 (5종): {VALDIR}/")
    print(f"  예측 차트 (7종): {OUTDIR}/")


if __name__ == "__main__":
    main()
