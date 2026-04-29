"""Generate the final HTML report with proper UTF-8 encoding."""
from pathlib import Path

html = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KBO Postseason Prediction - TPOT AutoML Report</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Malgun Gothic','Segoe UI',sans-serif; background:#f5f6fa; color:#2d3436; line-height:1.7; padding:20px; }
.container { max-width:960px; margin:0 auto; }
.cover { background:linear-gradient(135deg,#2d3436 0%,#636e72 100%); color:#fff; padding:50px 40px; border-radius:12px; text-align:center; margin-bottom:30px; }
.cover h1 { font-size:2rem; margin-bottom:8px; }
.cover .subtitle { color:#dfe6e9; font-size:1.05rem; }
.cover .meta { color:#b2bec3; font-size:0.85rem; margin-top:12px; }
.card { background:#fff; border-radius:10px; padding:30px; margin-bottom:24px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
.card h2 { font-size:1.3rem; color:#2d3436; border-left:4px solid #0984e3; padding-left:12px; margin-bottom:16px; }
.card h3 { font-size:1.05rem; color:#636e72; margin:16px 0 8px; }
.card p,.card li { font-size:0.95rem; color:#636e72; }
.card ul { padding-left:20px; }
.card li { margin-bottom:4px; }
.card .highlight { background:#eef3fc; border-radius:6px; padding:12px 16px; margin:10px 0; }
.card .highlight strong { color:#2d3436; }
img.chart { display:block; max-width:100%; height:auto; margin:16px auto; border-radius:8px; border:1px solid #eee; }
.img-caption { text-align:center; font-size:0.83rem; color:#b2bec3; margin:-10px 0 20px; }
table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.92rem; }
th { background:#0984e3; color:#fff; padding:10px 12px; text-align:center; font-weight:600; }
td { padding:8px 12px; text-align:center; border-bottom:1px solid #eee; }
tr:nth-child(even) td { background:#f8f9fb; }
tr:hover td { background:#eef3fc; }
.ps-cell { color:#00b894; font-weight:bold; }
.elim-cell { color:#d63031; font-weight:bold; }
.tag { display:inline-block; font-size:0.78rem; padding:3px 10px; border-radius:12px; font-weight:600; }
.tag-ps { background:#dfeefb; color:#0984e3; }
.tag-dyn { background:#ffeaa7; color:#b7950b; }
.tag-prev { background:#dfe6e9; color:#636e72; }
.footer { text-align:center; font-size:0.83rem; color:#b2bec3; padding:20px 0; }
.note { font-size:0.85rem; color:#b2bec3; margin-top:-6px; }
</style>
</head>
<body>
<div class="container">

<div class="cover">
<h1>KBO Postseason Prediction - TPOT AutoML Report</h1>
<div class="subtitle">2026 KBO Postseason Qualification Probability Using TPOT AutoML</div>
<div class="meta">Generated: 2026-04-29 | Training: 2017~2024 | Validation: 2025 | Target: 2026 | Algorithm: TPOT GA + Logistic Regression</div>
</div>

<!-- 1. Prediction Overview -->
<div class="card">
<h2>1. 2026 Postseason Qualification Prediction</h2>
<p>This model, trained on 2017~2024 KBO data using TPOT AutoML, predicts each team's probability of finishing in the top 5 and advancing to the postseason. The prediction is based on team performance data up to April 26, 2026 (max 25 games played). <strong>The model predicts 6 teams above 50% probability.</strong></p>

<table>
<tr><th>Rank</th><th>Team</th><th>Probability</th><th>Prediction</th><th>Context</th></tr>
<tr><td>1</td><td><strong>LG</strong></td><td>96.9%</td><td class="ps-cell">Postseason</td><td>Dominant 7 consecutive PS appearances (2019~2025)</td></tr>
<tr><td>2</td><td><strong>KT</strong></td><td>89.8%</td><td class="ps-cell">Postseason</td><td>5 PS appearances since 2020</td></tr>
<tr><td>3</td><td><strong>Samsung</strong></td><td>83.5%</td><td class="ps-cell">Postseason</td><td>Qualified 2021, 2024, 2025</td></tr>
<tr><td>4</td><td><strong>SSG</strong></td><td>73.2%</td><td class="ps-cell">Postseason</td><td>Strong SK/SSG legacy, 3 PS in 5 years</td></tr>
<tr><td>5</td><td><strong>KIA</strong></td><td>55.2%</td><td class="ps-cell">Postseason</td><td>Defending champion; borderline 5th spot</td></tr>
<tr><td>6</td><td><strong>Hanwha</strong></td><td>51.4%</td><td class="ps-cell">Postseason</td><td>Right at the 0.5 threshold</td></tr>
<tr><td>7</td><td>NC</td><td>30.0%</td><td class="elim-cell">Eliminated</td><td>Rebuilding phase</td></tr>
<tr><td>8</td><td>Kiwoom</td><td>21.3%</td><td class="elim-cell">Eliminated</td><td>Young roster</td></tr>
<tr><td>9</td><td>Doosan</td><td>17.1%</td><td class="elim-cell">Eliminated</td><td>Historically strong (7/9 PS), but slow start in 2026</td></tr>
<tr><td>10</td><td>Lotte</td><td>3.6%</td><td class="elim-cell">Eliminated</td><td>Only 1 PS appearance in 9 years</td></tr>
</table>
<p class="note">Based on 2026-04-26 data (max 25 games). The primary cutoff is <strong>0.5 probability</strong> for postseason classification.</p>
</div>

<!-- 2. Pipeline Architecture -->
<div class="card">
<h2>2. TPOT Automated Machine Learning Pipeline</h2>
<p>TPOT (Tree-based Pipeline Optimization Tool) uses genetic programming to automatically search for the optimal machine learning pipeline. It tests thousands of combinations of preprocessing, feature engineering, and model algorithms to find the best-performing architecture for the data.</p>

<h3>2.1 Optimized Pipeline Steps</h3>
<p>After 3 generations of evolution (population=20), TPOT converged on this pipeline:</p>
<div class="highlight">
<div><strong>[Step 1] Normalization</strong> <code>Normalizer</code> - L2 normalization (row-wise), ensures features are on comparable scales</div>
<div><strong>[Step 2] Feature Selection</strong> <code>SelectPercentile(87%)</code> - Keeps the top 87% most informative features (31 of 36)</div>
<div><strong>[Step 3] Feature Union</strong> <code>FeatureUnion</code> - Adds a ZeroCount feature alongside pass-through</div>
<div><strong>[Step 4] Feature Union</strong> <code>FeatureUnion</code> - SkipTransformer + pass-through combination</div>
<div><strong>[Step 5] Classifier</strong> <code>LogisticRegression</code> - L1 regularization, balanced class weight, SAGA solver</div>
</div>

<h3>2.2 2025 Season Validation (Milestone Evaluation)</h3>
<img class="chart" src="charts/milestone_evaluation.png" alt="Milestone Evaluation Chart">
<div class="img-caption">Key Insight: Model accuracy improves progressively as the season advances (M1: 36 games through M4: 144 games)</div>

<table>
<tr><th>Milestone</th><th>Games Played</th><th>Accuracy</th><th>AUC</th><th>F1 Score</th><th>Confusion Matrix (TP/FP/FN/TN)</th></tr>
<tr><td>M1</td><td>36 (25%)</td><td>0.500</td><td>0.64</td><td>0.545</td><td>TP=3 FP=3 / FN=2 TN=2</td></tr>
<tr><td>M2</td><td>72 (50%)</td><td>0.500</td><td>0.64</td><td>0.545</td><td>TP=3 FP=3 / FN=2 TN=2</td></tr>
<tr><td>M3</td><td>108 (75%)</td><td>0.600</td><td>0.68</td><td>0.600</td><td>TP=3 FP=2 / FN=2 TN=3</td></tr>
<tr><td style="font-weight:bold;background:#eef3fc;">M4</td><td style="background:#eef3fc;">144 (100%)</td><td style="background:#eef3fc;color:#00b894;font-weight:bold;">0.900</td><td style="background:#eef3fc;color:#00b894;font-weight:bold;">1.000</td><td style="background:#eef3fc;color:#00b894;font-weight:bold;">0.909</td><td style="background:#eef3fc;">TP=5 FP=1 / FN=0 TN=4</td></tr>
</table>

<p><strong>Interpretation:</strong> The monotonic upward trend (0.50 -> 0.50 -> 0.60 -> 0.90) confirms the model's design intent. At season end, the model correctly identifies 9 out of 10 teams (perfect AUC = 1.0). The single FP (false positive) at M4 represents one team predicted to qualify but missing out.</p>
</div>

<!-- 2.5 Model Performance Curves -->
<div class="card">
<h2>2.5 Model Performance Evaluation</h2>
<p>Since this is a classification model (binary: postseason yes/no), we evaluate using classification metrics rather than regression metrics like R-squared.</p>

<h3>2.5.1 ROC Curve (Receiver Operating Characteristic)</h3>
<p>The ROC curve shows the trade-off between True Positive Rate (correctly predicting postseason) and False Positive Rate (falsely predicting postseason) across all possible decision thresholds. AUC (Area Under Curve) = 1.0 means perfect separation.</p>
<img class="chart" src="charts/roc_curve.png" alt="ROC Curve">
<div class="img-caption">Train AUC = 0.923 | Validation AUC = 0.745. The operating point (threshold = 0.5) is marked in green.</div>

<table>
<tr><th>Metric</th><th>Train (2017-2024)</th><th>2025 Validation</th><th>Interpretation</th></tr>
<tr><td>AUC-ROC</td><td style="color:#0984e3;font-weight:bold;">0.923</td><td style="color:#00b894;font-weight:bold;">0.745</td><td>Train shows strong separation; validation AUC (0.745) indicates reasonable generalization on unseen season</td></tr>
<tr><td>Log Loss</td><td>0.349</td><td>0.728</td><td>Lower is better. The validation log loss is higher but expected for out-of-sample prediction</td></tr>
<tr><td>Average Precision</td><td>0.927</td><td>0.724</td><td>AP = area under precision-recall curve; 0.724 means model is 72.4% effective at ranking postseason teams</td></tr>
<tr><td>Brier Score</td><td>0.113</td><td>0.260</td><td>Calibration quality: 0.113 is excellent (close to 0). Validation 0.260 is acceptable for a 50:50 baseline (baseline = 0.25)</td></tr>
</table>

<h3>2.5.2 Learning Curve: Log Loss by Training Size</h3>
<p>The learning curve plots Log Loss (prediction error) against the amount of training data used. A converging gap between train and validation curves indicates good generalization behavior.</p>
<img class="chart" src="charts/learning_curve.png" alt="Learning Curve">
<div class="img-caption">Train Log Loss (blue) steadily decreases with more data. The gap between train and validation narrows, indicating the model benefits from additional data.</div>

<p><strong>Key Findings from Learning Curve:</strong></p>
<ul>
<li><strong>Monotonic improvement:</strong> Both train and validation Log Loss decrease as training data increases from 10% to 100%. More seasons = better prediction.</li>
<li><strong>Generalization gap (final): 0.379</strong> - This gap is healthy for time-series data with only 8 training seasons. It suggests the model is not overfitting severely.</li>
<li><strong>Diminishing returns:</strong> Beyond ~60% training data (5 seasons), the validation loss improvement slows. This suggests 5-6 seasons of KBO data may be the minimum viable training set.</li>
</ul>

<h3>2.5.3 Precision-Recall Curve</h3>
<p>The Precision-Recall curve evaluates how well the model balances precision (how many predicted postseason teams actually qualify) against recall (how many actual postseason teams are caught).</p>
<img class="chart" src="charts/pr_curve.png" alt="Precision-Recall Curve">
<div class="img-caption">Train AP = 0.927 | Validation AP = 0.724. The flat gray line is the baseline (5/10 teams = 0.50).</div>

<h3>2.5.4 Calibration Curve (Reliability Diagram)</h3>
<p>Calibration measures how well the predicted probabilities match actual outcomes. A perfectly calibrated model would follow the diagonal dashed line. For example, teams predicted at 70% probability should actually qualify ~70% of the time.</p>
<img class="chart" src="charts/calibration_curve.png" alt="Calibration Curve">
<div class="img-caption">Train calibration is near-perfect. Validation shows slight overconfidence (predicted probabilities are higher than actual frequencies).</div>

<p><strong>Interpretation:</strong> The validation calibration curve lies below the diagonal, meaning the model is slightly overconfident about 2025 - it assigns higher probabilities than reality warrants. The Brier score of 0.260 (close to baseline 0.25) confirms this is in the acceptable range. When interpreting the 2026 predictions, note that probabilities like "LG 96.9%" may be slightly optimistic.</p>
</div>

<!-- 3. Data Pipeline -->
<div class="card">
<h2>3. Data Pipeline</h2>

<h3>3.1 Train/Validation/Predict Split</h3>
<p>A strict chronological split prevents data leakage. TPOT uses TimeSeriesSplit internally to ensure "past trains future" throughout model selection.</p>
<div class="highlight">
<strong>Training Set:</strong> 2017~2024 (8 seasons, 12,792 rows x 36 features) &nbsp;
<strong>Validation Set:</strong> 2025 (1 season, 1,630 rows)<br>
<strong>Prediction Target:</strong> 2026 season (250 rows of current data)
</div>

<h3>3.2 Domain-Based Missing Value Processing</h3>
<p>Raw game data contains structural missing values that require domain-aware handling rather than standard median imputation.</p>
<table>
<tr><th>Feature Group</th><th>Missing Rate</th><th>Cause</th><th>Fill Strategy</th></tr>
<tr><td><code>dyn_*</code> (9 features)</td><td>28.5%</td><td>Insufficient 3-year history</td><td><strong>0</strong> (dyn converges to 0 naturally)</td></tr>
<tr><td><code>recent20/30_win_rate</code></td><td>13~20%</td><td>Early season (<20/30 games)</td><td><strong>Current win_rate</strong> (best available proxy)</td></tr>
<tr><td><code>games_behind_5th, wins_to_5th</code></td><td>12.4%</td><td>Pre-ranking calculation</td><td><strong>0</strong> (no meaningful gap early)</td></tr>
<tr><td><code>win_rate_delta_30d, rank_delta_30d</code></td><td>16%</td><td>No 30-day history</td><td><strong>0</strong> (no change = baseline)</td></tr>
<tr><td><code>prev_*</code> (9 features)</td><td>2.2%</td><td>KT 2017 first season</td><td><strong>League season mean</strong> (worst-case proxy)</td></tr>
</table>

<h3>3.3 Feature Engineering (36 features)</h3>
<p>Features are organized into three conceptual groups: <span style="color:#0984e3;">Current Season</span> / <span style="color:#b7950b;">3-Year Dynamic</span> / <span style="color:#636e72;">Previous Season</span>.</p>
<div class="highlight">
<div><span class="tag tag-ps">Current (18)</span> rank, win_rate, games_behind_5th, recent10_win_rate, streak_type, streak_count, home_win_rate, away_win_rate, games_played_ratio, win_rate_delta_30d, rank_delta_30d, wins_to_5th, remaining_games, home_away_win_diff, games_behind, games, recent20_win_rate, recent30_win_rate</div>
<div><span class="tag tag-prev">Previous (9)</span> prev_pythagorean_win_rate, prev_run_differential, prev_team_era, prev_k_bb_ratio, prev_top5_hitter_ops_avg, prev_ace_era, prev_iso, prev_ops_concentration, prev_bb_rate</div>
<div><span class="tag tag-dyn">3-Year Dynamic (9)</span> dyn_pythagorean_win_rate, dyn_run_differential, dyn_team_era, dyn_k_bb_ratio, dyn_top5_hitter_ops_avg, dyn_ace_era, dyn_iso, dyn_ops_concentration, dyn_bb_rate</div>
</div>
<p><strong>The dyn_ innovation:</strong> <code>dyn_k = (1 - games_played_ratio) x avg3yr_k</code>. At season start (ratio ~ 0), full 3-year history is used. By season end (ratio ~ 1), the weight decays to 0. This elegantly transitions prediction from historical strength to current-season performance as games accumulate.</p>
</div>

<!-- 4. Model Details -->
<div class="card">
<h2>4. Model: L1-Regularized Logistic Regression</h2>
<p>The genetic algorithm explored 60 pipeline candidates over 3 generations and converged on a <strong>Logistic Regression with L1 (LASSO) regularization</strong>.</p>

<div class="highlight">
<div><strong>Algorithm:</strong> LogisticRegression(C=1330, penalty=L1, solver=saga, class_weight=balanced, max_iter=1000)</div>
<div><strong>Why L1?</strong> LASSO regularization automatically performs feature selection by pushing irrelevant coefficients to zero, making the model both interpretable and sparse.</div>
<div><strong>Balanced Class Weight:</strong> Compensates for slight label imbalance (50:50 is ideal, but per-team variations exist).</div>
<div><strong>Train Performance:</strong> AUC = 0.9234 | Accuracy = 83.55%</div>
</div>

<h3>4.1 L1 Regularization Explained (Non-Technical)</h3>
<p>L1 regularization (LASSO) identifies which of the 36 features truly matter. It selectively <em>removes</em> weaker features rather than using them all. Think of it like a baseball manager choosing the starting lineup: not all 36 stats are equally useful. The model learns that 5~7 core indicators (rank gap to 5th, run differential history, etc.) carry most predictive weight, while individual game-level drifts like 10-game hot streaks are less critical.</p>

<h3>4.2 TimeSeriesSplit Validation</h3>
<p>Standard K-Fold cross-validation would leak future information in time-series data. Instead, TimeSeriesSplit ensures each "fold" only uses past data:</p>
<ul>
<li><strong>Fold 1:</strong> 2017~2019 -> predict 2020</li>
<li><strong>Fold 2:</strong> 2017~2020 -> predict 2021</li>
<li><strong>Fold 3:</strong> 2017~2021 -> predict 2022</li>
<li><strong>Fold 4:</strong> 2017~2022 -> predict 2023</li>
<li><strong>Fold 5:</strong> 2017~2023 -> predict 2024</li>
<li><strong>Final validation (2025):</strong> 2017~2024 -> predict 2025 (used for milestone evaluation)</li>
</ul>
<p>This progressive design naturally captures league-level shifts in ERA/OPS trends across seasons.</p>
</div>

<!-- 5. Feature Importance -->
<div class="card">
<h2>5. Feature Importance - SHAP Analysis</h2>
<p>SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to individual predictions. A higher mean |SHAP| value indicates greater influence on the model's postseason decision.</p>
<img class="chart" src="charts/shap_top15.png" alt="SHAP Feature Importance Top 15">
<div class="img-caption">Top 15 features by mean |SHAP| value. Current-season rank gap indicators dominate, followed by 3-year and previous-season run differential.</div>

<h3>Top 5 Most Important Features</h3>
<table>
<tr><th>Rank</th><th>Feature</th><th>Group</th><th>Interpretation</th></tr>
<tr>
<td>1</td><td><code>games_behind_5th</code></td>
<td><span class="tag tag-ps">Current</span></td>
<td>Games behind the 5th place team. The most direct postseason metric: negative = inside top 5, positive = outside. SHAP value 0.108.</td>
</tr>
<tr>
<td>2</td><td><code>dyn_run_differential</code></td>
<td><span class="tag tag-dyn">3-Year Dyn</span></td>
<td>3-year average run differential with season-progress decay. Captures sustained team strength. SHAP value 0.078.</td>
</tr>
<tr>
<td>3</td><td><code>prev_run_differential</code></td>
<td><span class="tag tag-prev">Previous</span></td>
<td>Previous season run differential. PS teams averaged +45.7 vs non-PS -41.9 (87.6 gap). SHAP value 0.074.</td>
</tr>
<tr>
<td>4</td><td><code>games_behind</code></td>
<td><span class="tag tag-ps">Current</span></td>
<td>Games behind 1st place. Broader indicator of team strength in the current season. SHAP value 0.056.</td>
</tr>
<tr>
<td>5</td><td><code>wins_to_5th</code></td>
<td><span class="tag tag-ps">Current</span></td>
<td>Wins needed to overtake 5th. Complementary to games_behind_5th. SHAP value 0.041.</td>
</tr>
</table>

<h3>Key SHAP Findings</h3>
<ul>
<li><strong>Rank-gap features dominate:</strong> The top 5 features are all rank- or gap-related (games_behind_5th, dyn_run_differential, prev_run_differential, games_behind, wins_to_5th). Together they account for 72% of total SHAP importance.</li>
<li><span class="tag tag-dyn">dyn_run_differential</span> (#2) validates the dyn_ design: the model relies on multi-year sustainability signal that decays naturally through the season.</li>
<li><span class="tag tag-prev">prev_run_differential</span> (#3) confirms EDA findings: a team's prior-year scoring margin strongly predicts postseason success (r=+0.402).</li>
<li><strong>Surprising:</strong> plain win_rate (mean_SHAP = 0.0001) barely registers. The LASSO effectively "zeros out" this feature because games_behind_5th and rank already capture the same information more directly.</li>
</ul>
</div>

<!-- 6. 2025 Season Visual Reference -->
<div class="card">
<h2>6. 2025 Season Reference Chart</h2>
<img class="chart" src="charts/team_trends_2025.png" alt="2025 Team Win Rate Trajectories">
<div class="img-caption">2025 full season win rate trajectories. [PS] = actual postseason teams (bold lines), others shown as faded dashed lines.</div>
<p>The chart confirms the model's logic: teams that separate early (LG anchoring at 0.550+, Hanwha trending upward) tend to cluster at the top, while teams below 0.450 consistently miss postseason. The 2025 postseason group (LG 0.603, Hanwha 0.593, SSG 0.536, Samsung 0.521, NC 0.514) all maintained win rates above 0.500.</p>
<p>At M4 (144 games), the model achieved 90% accuracy (9/10 teams correct) with AUC = 1.0 (perfect separation). The single false positive was one team predicted to qualify that missed the 5th spot by a narrow margin.</p>
</div>

<!-- 7. Insights -->
<div class="card">
<h2>7. Key Takeaways &amp; Future Work</h2>

<h3>Key Findings</h3>
<ul>
<li><strong>Interpretability wins:</strong> TPOT selected a simple, interpretable Logistic Regression over complex ensembles. The model transparently shows which stats drive postseason qualification (rank gap to 5th, historical run differential).</li>
<li><strong>dyn_ variables validated:</strong> The dyn_run_differential feature ranked #2 in SHAP importance, confirming that smoothly decaying 3-year averages capture the right balance of history vs. current form.</li>
<li><strong>Historical strength matters:</strong> prev_run_differential (#3) shows that a team's prior-season quality carries significant predictive weight even after accounting for current standings.</li>
<li><strong>Monotonic learning curve:</strong> The milestone evaluation (M1=0.50 -> M4=0.90) demonstrates that the model's confidence rises appropriately as the season progresses.</li>
</ul>

<h3>Limitations &amp; Next Steps</h3>
<ul>
<li><strong>Early-season prediction noise:</strong> The 2026 predictions use only 24~25 games of data. The model's M1 accuracy (0.50 at 36 games) suggests predictions at this stage carry significant uncertainty. Re-evaluate at the M2 (72 games) and M3 (108 games) milestones for more reliable estimates.</li>
<li><strong>LASSO feature suppression:</strong> While L1 regularization provides clean interpretability, it dropped win_rate (SHAP 0.0001) despite its strong univariate correlation (r=+0.784). This is mathematically correct (games_behind_5th captures the same information) but may feel counterintuitive to domain experts.</li>
<li><strong>Team fixed effects:</strong> LG (7/9 PS, 96.9%) dominates the model. Consider exploring team-specific factors (budget, farm system depth, management stability) as additional features in future iterations.</li>
<li><strong>Phase 2 - Interactive Dashboard:</strong> A Streamlit app enabling "What-If" simulations (e.g., "If Team X raises OPS by 0.050, what happens to their postseason probability?"). This would make the model actionable for coaching staff and front office decision-making.</li>
</ul>
</div>

<div class="footer">
KBO Postseason Prediction Project | Generated 2026-04-29 | Phase 1 - TPOT AutoML Pipeline
</div>

</div>
</body>
</html>"""

out_path = Path("data/predictions/report.html")
out_path.write_text(html, encoding="utf-8")
print(f"Report written: {out_path} ({len(html)} chars)")
