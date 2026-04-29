"""Generate all charts for the final report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/predictions/charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Korean font ──
font_candidates = [
    "New Gulim", "Malgun Gothic", "Gulim", "Batang", "NanumGothic", "NanumBarunGothic"
]
for f in font_candidates:
    found = [x for x in fm.fontManager.ttflist if f in x.name]
    if found:
        plt.rcParams["font.family"] = found[0].name
        print(f"Font: {found[0].name}")
        break
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# ───────────────────
# Chart 1: Milestone Evaluation (line chart)
# ───────────────────
def chart_milestone():
    with open("data/predictions/tpot_2025_evaluation.json", encoding="utf-8") as f:
        data = json.load(f)

    milestones = ["36", "72", "108", "144"]
    accuracy = [data[f"M{i+1}"]["accuracy"] for i in range(4)]
    auc = [data[f"M{i+1}"]["roc_auc"] for i in range(4)]
    f1 = [data[f"M{i+1}"]["f1"] for i in range(4)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(milestones))
    ax.plot(x, accuracy, "o-", color=COLORS[0], linewidth=2.5, markersize=8, label="Accuracy")
    ax.plot(x, auc, "s-", color=COLORS[1], linewidth=2.5, markersize=8, label="ROC AUC")
    ax.plot(x, f1, "D-", color=COLORS[2], linewidth=2.5, markersize=8, label="F1 Score")

    for i in range(4):
        ax.annotate(f"{accuracy[i]:.3f}", (i, accuracy[i]), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
        ax.annotate(f"{auc[i]:.3f}", (i, auc[i]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9)
        ax.annotate(f"{f1[i]:.3f}", (i, f1[i]), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["M1\n36", "M2\n72", "M3\n108", "M4\n144"], fontsize=11)
    ax.set_ylim(0.35, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("2025 Milestone (20425 Model)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "milestone_evaluation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/milestone_evaluation.png"

# ───────────────────
# Chart 2: 2026 Prediction Bar Chart
# ───────────────────
def chart_prediction():
    df = pd.read_csv("data/predictions/tpot_2026_predictions.csv", encoding="utf-8-sig")
    df = df.sort_values("postseason_prob", ascending=True)

    teams = df["team"].tolist()
    probs = df["postseason_prob"].tolist()
    preds = df["postseason_pred"].tolist()

    colors_bar = [COLORS[2] if p == 1 else COLORS[3] for p in preds]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(teams, probs, color=colors_bar, edgecolor="white", height=0.6)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=10, fontweight="bold")

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Postseason Probability", fontsize=12)
    ax.set_title("2026 KBO Postseason Prediction (TPOT + Logistic Regression)", fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[2], label="Predicted: Postseason"),
        Patch(facecolor=COLORS[3], label="Predicted: Elimination"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "2026_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/2026_predictions.png"

# ───────────────────
# Chart 3: SHAP Top 15 Feature Importance
# ───────────────────
def chart_shap_importance():
    df = pd.read_csv("data/predictions/tpot_feature_importance.csv", encoding="utf-8-sig")
    df = df.head(15).sort_values("mean_shap", ascending=True)

    features = df["feature"].tolist()
    importances = df["mean_shap"].tolist()

    # Human-readable names
    name_map = {
        "games_behind_5th": "5 (games_behind_5th)",
        "dyn_run_differential": "  (dyn_run_differential)",
        "prev_run_differential": "   (prev_run_differential)",
        "games_behind": "1 (games_behind)",
        "wins_to_5th": "5 (wins_to_5th)",
        "rank": "",
        "games": "",
        "remaining_games": "",
        "prev_ace_era": " (prev_ace_era)",
        "rank_delta_30d": " 30  (rank_delta_30d)",
        "dyn_team_era": " ERA (dyn_team_era)",
        "streak_count": " (streak_count)",
        "prev_team_era": " ERA (prev_team_era)",
        "streak_type": " (streak_type)",
        "dyn_k_bb_ratio": "/ (dyn_k_bb_ratio)",
    }
    labels = [f"{name_map.get(f, f)}" for f in features]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_shap = [COLORS[0]] * len(features)
    # Mark dyn_ features differently
    for i, f in enumerate(features):
        if f.startswith("dyn_"):
            colors_shap[i] = COLORS[1]
        elif f.startswith("prev_"):
            colors_shap[i] = COLORS[4]

    ax.barh(labels, importances, color=colors_shap, edgecolor="white", height=0.6)
    ax.set_xlabel("Mean |SHAP| Value", fontsize=12)
    ax.set_title("Top 15 Feature Importance (Permutation SHAP)", fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements2 = [
        Patch(facecolor=COLORS[0], label="Current Season"),
        Patch(facecolor=COLORS[4], label="Previous Season (prev_)"),
        Patch(facecolor=COLORS[1], label="3-Year Dynamic (dyn_)"),
    ]
    ax.legend(handles=legend_elements2, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "shap_top15.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/shap_top15.png"

# ───────────────────
# Chart 4: Team Season Win Rate Trends (2025)
# ───────────────────
def chart_team_trends():
    train = pd.read_csv("data/modeling/train_dataset.csv", encoding="utf-8-sig")
    val = train[train["season"] == 2025].copy()
    val["team_clean"] = val["team"].apply(lambda x: str(x).encode("cp949", errors="replace").decode("cp949"))

    fig, ax = plt.subplots(figsize=(11, 6))
    postseason_teams = ["LG", "NC", "SSG", "", ""]
    # Find actual 2025 postseason teams from final_rank
    final = val[val["games"] == 144]
    ps_teams = set(final[final["final_rank"] <= 5]["team"].tolist())

    for i, team in enumerate(sorted(val["team"].unique())):
        tdf = val[val["team"] == team].sort_values("games")
        team_str = str(team)
        is_ps = team in ps_teams
        alpha = 0.9 if is_ps else 0.4
        lw = 2.0 if is_ps else 1.0
        ls = "-" if is_ps else "--"
        ax.plot(tdf["games"], tdf["win_rate"], color=COLORS[i % 10], alpha=alpha,
                linewidth=lw, linestyle=ls, label=f"{team_str} {'[PS]' if is_ps else ''}")

    ax.axhline(y=0.500, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Games Played", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("2025 Season Win Rate Trajectory (Bold = Postseason)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "team_trends_2025.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return "charts/team_trends_2025.png"

# ── Run all ──
if __name__ == "__main__":
    chart_milestone()
    chart_prediction()
    chart_shap_importance()
    chart_team_trends()
    print("\nAll charts generated!")
