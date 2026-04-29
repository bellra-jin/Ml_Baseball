# notebooks/experiments/lsh/tpot_pipeline.py
#
# KBO 포스트시즌 진출 예측 — TPOT AutoML 파이프라인 (Phase 1)
#
# - 2017~2024 시즌 데이터로 TPOT 학습 (TimeSeriesSplit)
# - 2025 시즌 마일스톤(36/72/108/144경기) 기준 평가
# - SHAP 기반 피처 중요도 / 설명 분석
# - 2026 시즌 포스트시즌 진출 확률 예측
#
# 실행: uv run python notebooks/experiments/lsh/tpot_pipeline.py

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from tpot import TPOTClassifier

from src.utils.config import FEATURE_COLS, TRAIN_SEASONS, PREDICT_SEASON, MULTI_YEAR_KEYS
from src.utils.paths import MODELING_DIR, PREDICTIONS_DIR, make_dirs

warnings.filterwarnings("ignore")

# ─────────────────────
# 상수
# ─────────────────────
RANDOM_STATE = 42
MILESTONES = {"M1": 36, "M2": 72, "M3": 108, "M4": 144}

# 년도별 전처리된 prev_ 특성의 리그 평균 계산에 필요한 prev_ 접두사
PREV_COLS = [c for c in FEATURE_COLS if c.startswith("prev_")]
# dyn_ 접두사
DYN_COLS = [c for c in FEATURE_COLS if c.startswith("dyn_")]


# ─────────────────────
# 데이터 로드
# ─────────────────────
def load_data():
    train_path = MODELING_DIR / "train_dataset.csv"
    predict_path = MODELING_DIR / f"predict_dataset_{PREDICT_SEASON}.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train_dataset.csv not found at {train_path}. Run build_train_dataset first."
        )
    if not predict_path.exists():
        raise FileNotFoundError(
            f"predict_dataset_{PREDICT_SEASON}.csv not found at {predict_path}. Run build_predict_dataset first."
        )

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    predict_df = pd.read_csv(predict_path, encoding="utf-8-sig")

    print(f"[LOAD] train: {train_df.shape}, predict: {predict_df.shape}")
    return train_df, predict_df


# ─────────────────────
# 도메인 기반 NaN 전처리
# ─────────────────────
def preprocess_nan(df, is_train=True, train_prev_means=None):
    """
    도메인 지식 기반 NaN 처리.
    - dyn_* → 0 (시즌 진행도에 따라 자연스럽게 0으로 수렴)
    - recent20/30_win_rate → win_rate (현재 누적 승률) → 없으면 0.500
    - win_rate_delta_30d, rank_delta_30d → 0 (변화 없음)
    - games_behind_5th, wins_to_5th → 0 (초반엔 의미 없는 승차)
    - prev_* → 시즌별 리그 평균 (is_train) or train_prev_means 참조 (is_predict)
    - 잔여 NaN → 0

    Returns:
        df_clean: NaN이 제거된 DataFrame
        prev_means: 시즌별 prev_ 특성의 리그 평균 (predict용 참조)
    """
    df = df.copy()

    # 1) dyn_* → 0
    for c in DYN_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 2) recent20_win_rate, recent30_win_rate → win_rate → 0.500
    for c in ["recent20_win_rate", "recent30_win_rate"]:
        if c in df.columns:
            df[c] = df[c].fillna(df["win_rate"] if "win_rate" in df.columns else 0.500)
            df[c] = df[c].fillna(0.500)

    # 3) win_rate_delta_30d, rank_delta_30d → 0
    for c in ["win_rate_delta_30d", "rank_delta_30d"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 4) games_behind_5th, wins_to_5th → 0
    for c in ["games_behind_5th", "wins_to_5th"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 5) prev_* → 시즌별 리그 평균
    if is_train:
        prev_means = {}
        for c in PREV_COLS:
            if c in df.columns:
                season_mean = df.groupby("season")[c].mean()
                prev_means[c] = season_mean.to_dict()
                df[c] = df.apply(
                    lambda row, col=c, sm=season_mean: sm.get(row["season"], sm.mean())
                    if pd.isna(row[col])
                    else row[col],
                    axis=1,
                )
    else:
        if train_prev_means is not None:
            for c in PREV_COLS:
                if c in df.columns and c in train_prev_means:
                    sm = pd.Series(train_prev_means[c])
                    df[c] = df.apply(
                        lambda row, col=c, s=sm: s.get(row["season"], s.mean())
                        if pd.isna(row[col])
                        else row[col],
                        axis=1,
                    )
                elif c in df.columns:
                    df[c] = df[c].fillna(df[c].mean() if not df[c].isna().all() else 0)

    # 6) 잔여 NaN → 0
    remaining = df.columns[df.isna().any()]
    if len(remaining) > 0:
        print(f"[NaN] 잔여 NaN {len(remaining)}개 컬럼 → 0으로 채움: {list(remaining)}")
        df = df.fillna(0)

    total_nan = df.isna().sum().sum()
    print(f"[NaN] 전처리 후 남은 NaN: {total_nan}")

    if is_train:
        return df, prev_means
    return df


# ─────────────────────
# train / validation 분할 (시계열)
# ─────────────────────
def split_train_val(df):
    train_df = df[df["season"].isin(TRAIN_SEASONS[:-1])].copy()  # 2017~2024
    val_df = df[df["season"] == TRAIN_SEASONS[-1]].copy()  # 2025

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["postseason"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["postseason"]

    print(f"[SPLIT] train: {X_train.shape}, val: {X_val.shape}")
    print(f"[SPLIT] train postseason rate: {y_train.mean():.3f}")
    print(f"[SPLIT] val postseason rate: {y_val.mean():.3f}")

    return X_train, y_train, X_val, y_val, train_df, val_df


# ─────────────────────
# 마일스톤 데이터 추출
# ─────────────────────
def get_milestone_data(df, prefix=""):
    """
    특정 시즌의 각 팀별로 milestone 게임 수에 도달한 첫 번째 행을 추출한다.
    Returns: dict { "M1": DataFrame, "M2": ... }
    """
    results = {}
    for milestone_name, min_games in MILESTONES.items():
        idx = df.groupby("team")["games"].transform(
            lambda g: (g >= min_games).idxmax()
            if (g >= min_games).any()
            else g.idxmax()
        )
        milestone_df = df.loc[idx.unique()]
        results[milestone_name] = milestone_df
        gmin = milestone_df["games"].min()
        gmax = milestone_df["games"].max()
        n_teams = len(milestone_df)
        print(
            "  %s%s (games>=%d): %d teams, games range %d-%d"
            % (prefix, milestone_name, min_games, n_teams, gmin, gmax)
        )
    return results


# ─────────────────────
# TPOT 학습 (TimeSeriesSplit)
# ─────────────────────
def train_tpot(X_train, y_train):
    print("\n[TRAIN] TPOT 학습 시작...")
    start = datetime.now()

    tscv = TimeSeriesSplit(n_splits=5)

    tpot = TPOTClassifier(
        generations=3,
        population_size=20,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_time_mins=120,
        scorers=["roc_auc_ovr"],
        cv=tscv,
        early_stop=3,
        memory="auto",
    )

    tpot.fit(X_train, y_train)

    elapsed = datetime.now() - start
    print(f"[TRAIN] 완료. 소요 시간: {elapsed}")

    if hasattr(tpot, "evaluated_individuals") and tpot.evaluated_individuals is not None:
        best_scores = tpot.evaluated_individuals[
            tpot.evaluated_individuals[tpot.objective_names_for_selection]
            .isna().all(1)
            .ne(True)
        ][tpot.objective_names_for_selection]
        if len(best_scores) > 0:
            print(f"[TRAIN] Best CV Score: {best_scores.iloc[0].to_dict()}")

    return tpot


# ─────────────────────
# 저장용 경로 확보
# ─────────────────────
def ensure_output_dirs():
    make_dirs()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────
# Pipeline export
# ─────────────────────
def export_pipeline(tpot, pipeline_path):
    pipeline = tpot.fitted_pipeline_

    with open(pipeline_path, "w", encoding="utf-8") as f:
        f.write("# TPOT Best Pipeline (auto-generated)\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Pipeline string:\n")
        f.write(f"# {pipeline}\n\n")
        f.write("import numpy as np\n")
        f.write("import pandas as pd\n\n")
        f.write("# Exported pipeline\n")
        f.write(f"exported_pipeline = {repr(pipeline)}\n")

    steps_str = "\n  ".join([f"{i}. {s}" for i, s in enumerate(pipeline.steps)])
    print(f"[EXPORT] Pipeline saved to {pipeline_path}")
    print(f"[EXPORT] Steps:\n  {steps_str}")


# ─────────────────────
# 2025 마일스톤 평가
# ─────────────────────
def evaluate_milestones(pipeline, val_df):
    print("\n[EVAL] 2025 마일스톤 평가")
    milestones = get_milestone_data(val_df, prefix="  ")

    results = {}
    for name, mdf in milestones.items():
        X_m = mdf[FEATURE_COLS]
        y_m = mdf["postseason"]
        y_prob = pipeline.predict_proba(X_m)[:, 1]
        y_pred = pipeline.predict(X_m)

        metrics = {
            "accuracy": round(accuracy_score(y_m, y_pred), 4),
            "precision": round(precision_score(y_m, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_m, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_m, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_m, y_prob), 4),
            "confusion_matrix": confusion_matrix(y_m, y_pred).tolist(),
        }
        results[name] = metrics
        print(
            f"  {name}: acc={metrics['accuracy']:.3f}, "
            f"auc={metrics['roc_auc']:.3f}, "
            f"f1={metrics['f1']:.3f}"
        )
    return results


# ─────────────────────
# SHAP 분석
# ─────────────────────
def shap_analysis(pipeline, X_train_sample, X_val):
    print("\n[SHAP] 설명 가능성 분석 시작...")

    try:
        import shap
    except ImportError:
        print("[SHAP] shap library not installed. Skipping.")
        return {}

    # Get the final estimator from the fitted pipeline
    final_estimator = pipeline
    if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
        final_estimator = pipeline.steps[-1][1]

    est_name = str(type(final_estimator).__name__)

    # Use KernelExplainer with the full pipeline's predict_proba
    # This avoids needing to manually split and fit preprocessors
    shap_values = None
    explainer = None
    background_n = min(100, len(X_train_sample))
    X_bg = X_train_sample.iloc[:background_n]

    # LinearExplainer for logistic regression
    if hasattr(final_estimator, "coef_") and "Logistic" in est_name:
        try:
            from sklearn.pipeline import Pipeline as SKPipeline
            preproc_steps = pipeline.steps[:-1]
            preproc = SKPipeline(preproc_steps)
            preproc.fit(X_train_sample)  # Need to fit to get shapes right
            raise RuntimeError("skip to Permutation")
        except Exception:
            pass

    # Use PermutationExplainer with full pipeline for model-agnostic SHAP
    try:
        explainer = shap.PermutationExplainer(
            pipeline.predict_proba,
            X_bg,
            seed=RANDOM_STATE,
        )
        # Use a subset for speed
        X_val_subset = X_val.iloc[:min(500, len(X_val))]
        shap_values_raw = explainer(X_val_subset, max_evals=500)
        shap_values = shap_values_raw.values[:, :, 1]
        # PermutationExplainer gives SHAP values w.r.t. raw features
        shap_feature_names = list(X_val.columns)
        X_val_transformed = X_val_subset.values
        print(f"[SHAP] PermutationExplainer on {est_name}")
    except Exception as e:
        print(f"[SHAP] PermutationExplainer failed ({e}), skipping SHAP")
        return {}

    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    mean_shap = np.abs(shap_values).mean(axis=0)
    n_shap_feats = shap_values.shape[1]

    if n_shap_feats == len(shap_feature_names):
        shap_features = shap_feature_names
    else:
        shap_features = [f"feat_{i}" for i in range(n_shap_feats)]

    shap_importance = pd.DataFrame(
        {"feature": shap_features, "mean_shap": mean_shap}
    ).sort_values("mean_shap", ascending=False)

    # ── Summary Plot ──
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            X_val_transformed,
            feature_names=shap_features,
            max_display=min(20, n_shap_feats),
            show=False,
        )
        summary_path = PREDICTIONS_DIR / "shap_summary.png"
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SHAP] Summary plot saved: {summary_path}")
    except Exception as e:
        print(f"[SHAP] Summary plot error: {e}")

    # ── Importance CSV ──
    imp_path = PREDICTIONS_DIR / "tpot_feature_importance.csv"
    shap_importance.to_csv(imp_path, index=False, encoding="utf-8-sig")
    print(f"[SHAP] Feature importance: {imp_path}")
    print(f"[SHAP] Top 5 features: {shap_importance.head(5).to_dict('records')}")

    return {
        "top_features": shap_importance.head(10).to_dict("records"),
        "summary_plot": str(summary_path),
        "importance_csv": str(imp_path),
    }


# ─────────────────────
# 2026 예측
# ─────────────────────
def predict_2026(pipeline, predict_df):
    print("\n[PREDICT] 2026 시즌 예측")
    X_pred = predict_df[FEATURE_COLS]

    y_prob = pipeline.predict_proba(X_pred)[:, 1]
    y_pred = pipeline.predict(X_pred)

    predict_df = predict_df.copy()
    predict_df["postseason_prob"] = y_prob
    predict_df["postseason_pred"] = y_pred

    # 마지막 경기일 기준 각 팀 예측 정리
    latest_idx = predict_df.groupby("team")["games"].idxmax()
    latest_pred = predict_df.loc[latest_idx].copy()
    latest_pred = latest_pred.sort_values("postseason_prob", ascending=False)

    print("\n[PREDICT] 2026 팀별 Postseason 진출 확률 (최신 경기일 기준):")
    print(f"{'팀':<6s} {'확률':>8s} {'예측':>6s}")
    print("-" * 25)
    for _, row in latest_pred.iterrows():
        marker = " ★" if row["postseason_pred"] == 1 else ""
        print(
            f"{row['team']:<6s} {row['postseason_prob']:>7.1%} "
            f"{'진출' if row['postseason_pred']==1 else '탈락':>3s}{marker}"
        )

    # 저장
    pred_path = PREDICTIONS_DIR / f"tpot_{PREDICT_SEASON}_predictions.csv"
    latest_pred[["team", "games", "win_rate", "postseason_prob", "postseason_pred"]].to_csv(
        pred_path, index=False, encoding="utf-8-sig",
    )
    print(f"[PREDICT] Full predictions saved: {pred_path}")

    return latest_pred


# ─────────────────────
# 메인
# ─────────────────────
def main():
    ensure_output_dirs()
    print("=" * 60)
    print("KBO Postseason Prediction - TPOT AutoML Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"FEATURE_COLS: {len(FEATURE_COLS)}")
    print("=" * 60)

    # 1. 데이터 로드
    train_df, predict_df = load_data()

    # 2. NaN 전처리
    print("\n--- NaN Preprocessing ---")
    train_df, prev_means = preprocess_nan(train_df, is_train=True)
    predict_df = preprocess_nan(predict_df, is_train=False, train_prev_means=prev_means)

    # 3. Train/Validation 분할
    print("\n--- Train/Validation Split ---")
    X_train, y_train, X_val, y_val, train_tdf, val_df = split_train_val(train_df)

    # 4. TPOT 학습
    print("\n--- TPOT Training ---")
    tpot = train_tpot(X_train, y_train)

    # 5. Pipeline export
    print("\n--- Pipeline Export ---")
    pipeline_path = PREDICTIONS_DIR / "tpot_best_pipeline.py"
    export_pipeline(tpot, pipeline_path)

    # 6. 2025 마일스톤 평가
    print("\n--- 2025 Milestone Evaluation ---")
    milestone_metrics = evaluate_milestones(tpot.fitted_pipeline_, val_df)

    # 평가 결과 저장
    eval_path = PREDICTIONS_DIR / "tpot_2025_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(milestone_metrics, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] Results saved: {eval_path}")

    # 7. SHAP 분석
    print("\n--- SHAP Analysis ---")
    sample_n = min(2000, len(X_train))
    X_train_sample = X_train.sample(sample_n, random_state=RANDOM_STATE)
    shap_result = shap_analysis(tpot.fitted_pipeline_, X_train_sample, X_val)

    # 8. 2026 예측
    print("\n--- 2026 Prediction ---")
    latest_pred = predict_2026(tpot.fitted_pipeline_, predict_df)

    # 9. Pipeline score
    print("\n--- Summary ---")
    print(f"Best TPOT pipeline: {tpot.fitted_pipeline_}")
    y_train_pred = tpot.predict(X_train)
    y_train_prob = tpot.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_prob)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Train AUC: {train_auc:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Output directory: {PREDICTIONS_DIR}")
    print(f"Completed: {datetime.now().isoformat()}")
    print("=" * 60)

    return tpot, milestone_metrics, shap_result, latest_pred


if __name__ == "__main__":
    main()
