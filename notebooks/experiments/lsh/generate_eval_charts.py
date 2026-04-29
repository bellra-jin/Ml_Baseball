"""Generate model evaluation charts for the report."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss, accuracy_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

from src.utils.config import FEATURE_COLS, TRAIN_SEASONS
from src.utils.paths import MODELING_DIR

OUT_DIR = Path("data/predictions/charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False
for f in ["New Gulim", "Malgun Gothic", "Gulim"]:
    found = [x for x in fm.fontManager.ttflist if f in x.name]
    if found:
        plt.rcParams["font.family"] = found[0].name
        break

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Load data and preprocess
def load_and_preprocess():
    train_path = MODELING_DIR / "train_dataset.csv"
    df = pd.read_csv(train_path, encoding="utf-8-sig")

    # Domain NaN fill
    for c in [c for c in FEATURE_COLS if c.startswith("dyn_")]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    df["recent20_win_rate"] = df["recent20_win_rate"].fillna(df["win_rate"]).fillna(0.5)
    df["recent30_win_rate"] = df["recent30_win_rate"].fillna(df["win_rate"]).fillna(0.5)
    for c in ["win_rate_delta_30d", "rank_delta_30d", "games_behind_5th", "wins_to_5th"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in [c for c in FEATURE_COLS if c.startswith("prev_")]:
        if c in df.columns:
            season_mean = df.groupby("season")[c].mean()
            df[c] = df.apply(lambda row, col=c, sm=season_mean: sm.get(row["season"], sm.mean()) if pd.isna(row[col]) else row[col], axis=1)
    df = df.fillna(0)

    train_df = df[df["season"].isin(TRAIN_SEASONS[:-1])].copy()
    val_df = df[df["season"] == TRAIN_SEASONS[-1]].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["postseason"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["postseason"]
    return X_train, y_train, X_val, y_val


# Load fitted pipeline
def get_fitted_pipeline():
    import json, pickle, importlib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.linear_model import LogisticRegression

    # Manually reconstruct the pipeline (TPOT's export has issues with complex objects)
    from tpot.builtin_modules import ZeroCount, SkipTransformer, Passthrough
    from sklearn.feature_selection import SelectPercentile
    from sklearn.pipeline import FeatureUnion

    pipeline = Pipeline(steps=[
        ("normalizer", Normalizer(norm="l2")),
        ("selectpercentile", SelectPercentile(percentile=87.4306730968204)),
        ("featureunion-1", FeatureUnion(transformer_list=[
            ("featureunion", FeatureUnion(transformer_list=[("zerocount", ZeroCount())])),
            ("passthrough", Passthrough()),
        ])),
        ("featureunion-2", FeatureUnion(transformer_list=[
            ("skiptransformer", SkipTransformer()),
            ("passthrough", Passthrough()),
        ])),
        ("logisticregression", LogisticRegression(
            C=1330.4329413670953, class_weight="balanced",
            max_iter=1000, n_jobs=1, penalty="l1",
            random_state=42, solver="saga",
        )),
    ])
    return pipeline


# ─────────────────────
# Chart 1: ROC Curve
# ─────────────────────
def roc_chart(pipeline, X_train, y_train, X_val, y_val):
    pipeline.fit(X_train, y_train)

    # Train ROC
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    auc_train = auc(fpr_train, tpr_train)

    # Val ROC
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob)
    auc_val = auc(fpr_val, tpr_val)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_train, tpr_train, color=COLORS[0], linewidth=2.2, label=f'Train ROC (AUC = {auc_train:.3f})')
    ax.plot(fpr_val, tpr_val, color=COLORS[3], linewidth=2.5, label=f'2025 Validation ROC (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random (AUC = 0.500)")

    # Mark operating point (0.5 threshold)
    from sklearn.metrics import confusion_matrix
    y_pred = pipeline.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    tpr_op = tp / (tp + fn)
    fpr_op = fp / (fp + tn)
    ax.scatter([fpr_op], [tpr_op], s=120, color="#2ca02c", zorder=5, edgecolors="black", linewidths=1.2)
    ax.annotate(f"Op. Point (thresh=0.5)\nFPR={fpr_op:.2f}, TPR={tpr_op:.2f}",
                (fpr_op, tpr_op), textcoords="offset points", xytext=(20, -25),
                ha="left", fontsize=9, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve - Postseason Classification", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/roc_curve.png", y_train_prob, y_val_prob, pipeline


# ─────────────────────
# Chart 2: Learning Curve (Log Loss vs Training Size)
# ─────────────────────
def learning_curve_chart(pipeline, X_train, y_train, X_val, y_val):
    """Generate learning curve showing log loss convergence on train/validation sets."""
    sample_sizes = np.linspace(0.1, 1.0, 15)
    train_ll = []
    val_ll = []

    for frac in sample_sizes:
        n = max(int(len(X_train) * frac), 200)
        X_sub = X_train.iloc[:n]
        y_sub = y_train.iloc[:n]

        pipeline.fit(X_sub, y_sub)

        y_pred_prob_train = pipeline.predict_proba(X_sub)[:, 1]
        y_pred_prob_val = pipeline.predict_proba(X_val)[:, 1]
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        y_pred_prob_train = np.clip(y_pred_prob_train, eps, 1 - eps)
        y_pred_prob_val = np.clip(y_pred_prob_val, eps, 1 - eps)

        train_ll.append(log_loss(y_sub, y_pred_prob_train))
        val_ll.append(log_loss(y_val, y_pred_prob_val))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sample_sizes * 100, train_ll, "o-", color=COLORS[0], linewidth=2.2, markersize=6, label="Train Log Loss")
    ax.plot(sample_sizes * 100, val_ll, "s-", color=COLORS[3], linewidth=2.5, markersize=6, label="2025 Validation Log Loss")
    ax.fill_between(sample_sizes * 100, train_ll, val_ll, alpha=0.1, color="gray")

    # Mark gap
    gap = [val_ll[i] - train_ll[i] for i in range(len(train_ll))]
    ax.annotate(f"Generalization Gap (final): {gap[-1]:.4f}",
                (95, (train_ll[-1] + val_ll[-1]) / 2),
                textcoords="offset points", xytext=(10, 0),
                fontsize=9, color="#636e72")

    ax.set_xlabel("Training Set Size (%)", fontsize=12)
    ax.set_ylabel("Log Loss (lower is better)", fontsize=12)
    ax.set_title("Learning Curve - Log Loss by Training Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "learning_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/learning_curve.png", train_ll, val_ll


# ─────────────────────
# Chart 3: Precision-Recall Curve
# ─────────────────────
def pr_curve_chart(y_train, y_train_prob, y_val, y_val_prob):
    # Train
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_prob)
    ap_train = average_precision_score(y_train, y_train_prob)

    # Val
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_prob)
    ap_val = average_precision_score(y_val, y_val_prob)

    # Baseline: random classifier
    baseline = y_val.mean()
    n_pos = y_val.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_train, precision_train, color=COLORS[0], linewidth=2.2, label=f'Train PR (AP = {ap_train:.3f})')
    ax.plot(recall_val, precision_val, color=COLORS[3], linewidth=2.5, label=f'2025 Validation PR (AP = {ap_val:.3f})')
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.6, linewidth=1.2, label=f'Baseline (prev = {baseline:.2f})')

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/pr_curve.png", ap_train, ap_val


# ─────────────────────
# Chart 4: Calibration Curve (Reliability Diagram)
# ─────────────────────
def calibration_chart(y_train, y_train_prob, y_val, y_val_prob):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, y_true, y_prob, color in [
        ("Train (2017-2024)", y_train, y_train_prob, COLORS[0]),
        ("2025 Validation", y_val, y_val_prob, COLORS[3]),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, "s-", color=color, linewidth=2, markersize=7, label=f"{name}")
        # Annotate Brier score
        brier = brier_score_loss(y_true, y_prob)
        print(f"  {name} Brier Score: {brier:.4f}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Perfect Calibration")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "calibration_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/calibration_curve.png"


# ── Main ──
if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_and_preprocess()
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"Train PS rate: {y_train.mean():.3f}, Val PS rate: {y_val.mean():.3f}")

    print("\nBuilding pipeline...")
    pipeline = get_fitted_pipeline()

    print("\n1. ROC Curve...")
    roc_path, y_train_prob, y_val_prob, pipeline = roc_chart(pipeline, X_train, y_train, X_val, y_val)

    print("\n2. Learning Curve...")
    lc_path, train_ll, val_ll = learning_curve_chart(pipeline, X_train, y_train, X_val, y_val)

    print("\n3. Precision-Recall Curve...")
    pr_path, ap_train, ap_val = pr_curve_chart(y_train, y_train_prob, y_val, y_val_prob)

    print("\n4. Calibration Curve...")
    calib_path = calibration_chart(y_train, y_train_prob, y_val, y_val_prob)

    print("\n=== Summary ===")
    print(f"Train Log Loss (full): {train_ll[-1]:.4f}")
    print(f"Val Log Loss: {val_ll[-1]:.4f}")
    print(f"Train AP: {ap_train:.4f}")
    print(f"Val AP: {ap_val:.4f}")
    print(f"Val Brier: {brier_score_loss(y_val, y_val_prob):.4f}")
    print("\nAll evaluation charts generated!")
