# src/evaluation/metrics.py
#
# 모델 평가 공통 함수 모음.
# Validation 파일들이 이 모듈을 import해서 사용한다.
# 2026 예측처럼 정답이 없는 경우엔 사용 불가 — 시즌 종료 후 실제 결과와 비교할 때 활용한다.

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)


def evaluate_binary_model(y_true, y_proba, threshold=0.5):
    """
    이진 분류 성능 지표를 계산한다.

    Args:
        y_true   : 실제 레이블 (0/1 시리즈)
        y_proba  : 예측 확률 (소프트 보팅 결과)
        threshold: 확률 → 클래스 변환 기준 (기본 0.5)

    Returns:
        dict — accuracy, precision, recall, f1, roc_auc, brier
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba),
        "brier":     brier_score_loss(y_true, y_proba),
    }


def top5_accuracy(final_df, prob_col, actual_top5):
    """
    최종 시점 예측 상위 5팀 중 실제 포스트시즌 진출 팀 수를 계산한다.

    Args:
        final_df   : 팀별 최종 시점 1행짜리 DataFrame (team 컬럼 필수)
        prob_col   : 확률 컬럼명
        actual_top5: 실제 포스트시즌 진출 팀 집합 (set)

    Returns:
        (hit_count, hit_ratio) — 적중 팀 수, 비율(0~1)
    """
    predicted_top5 = set(final_df.nlargest(5, prob_col)["team"])
    hit = len(predicted_top5 & actual_top5)
    return hit, hit / 5


def checkpoint_hits(pred_df, prob_col, actual_top5, checkpoints):
    """
    시점별(50% / 75% / 90% / 최종) 상위 5팀 예측 적중 수를 계산한다.

    Args:
        pred_df    : 일자별 전체 예측 DataFrame (games_played_ratio, date, team 컬럼 필수)
        prob_col   : 확률 컬럼명
        actual_top5: 실제 포스트시즌 진출 팀 집합 (set)
        checkpoints: dict {label: games_played_ratio 기준값}

    Returns:
        dict {label: {"top5": list, "hit": int}}
    """
    results = {}
    for label, ratio in checkpoints.items():
        snap   = pred_df[pred_df["games_played_ratio"] <= ratio]
        latest = snap.sort_values("date").groupby("team").last().reset_index()
        top5   = set(latest.nlargest(5, prob_col)["team"])
        hit    = len(top5 & actual_top5)
        results[label] = {"top5": sorted(top5), "hit": hit}
    return results


def print_checkpoint_report(results):
    """checkpoint_hits 결과를 표 형식으로 출력한다."""
    print(f"{'시점':<16} {'예측 상위 5팀':<38} {'적중':>6}")
    print("─" * 65)
    for label, v in results.items():
        print(f"{label:<16} {str(v['top5']):<38} {v['hit']}/5")


def print_metrics(metrics_dict, label=""):
    """evaluate_binary_model 결과를 한 줄로 출력한다."""
    prefix = f"[{label}] " if label else ""
    m = metrics_dict
    brier_str = f"  Brier={m['brier']:.4f}" if "brier" in m else ""
    print(
        f"{prefix}"
        f"ROC-AUC={m['roc_auc']:.4f}  "
        f"F1={m['f1']:.4f}  "
        f"Precision={m['precision']:.4f}  "
        f"Recall={m['recall']:.4f}  "
        f"Accuracy={m['accuracy']:.4f}"
        f"{brier_str}"
    )
