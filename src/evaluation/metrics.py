# 모델 성능 평가용

# import pandas as pd

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
# )


# def top5_accuracy(result_df):
#     """예측 확률 상위 5개 팀 중 실제 진출 팀 비율을 계산한다."""
#     top5 = result_df.sort_values(
#         "postseason_probability",
#         ascending=False,
#     ).head(5)

#     return top5["postseason"].sum() / 5


# def evaluate_binary_model(y_true, y_pred, y_proba):
#     """기본 이진 분류 성능 지표를 계산한다."""
#     result = {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred),
#         "recall": recall_score(y_true, y_pred),
#         "f1": f1_score(y_true, y_pred),
#         "roc_auc": roc_auc_score(y_true, y_proba),
#     }

#     return result