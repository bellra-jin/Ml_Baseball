import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd
import streamlit as st

from components.model import load_model_and_predict, run_loso_cv
from components.charts import (
    TEAM_COLORS, bar_chart, importance_chart,
    val_scorecard_chart, val_overfit_gap_chart, val_calibration_chart,
)
from components.style import COMMON_CSS

ROOT = os.path.join(os.path.dirname(__file__), "../../..")

st.set_page_config(
    page_title="분석 보고서 | 2026 KBO",
    page_icon="📝",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.report-card {
    margin-bottom: 24px;
}
.report-card h2 {
    font-size: 1.0rem;
    font-weight: 700;
    color: #1E293B;
    padding: 7px 14px;
    margin: 0 0 18px 0;
    background: white;
    border-left: 4px solid #2563EB;
    border-radius: 0 10px 10px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 8px;
}
.report-card h3 {
    font-size: 0.96rem;
    font-weight: 800;
    color: #334155;
    margin: 18px 0 8px 0;
}
.report-card p, .report-card li {
    font-size: 0.86rem;
    color: #475569;
    line-height: 1.85;
}
.report-card ul {
    margin: 8px 0 0 0;
    padding-left: 1.2rem;
}
.report-highlight {
    background: linear-gradient(90deg, #EFF6FF 0%, #F8FAFC 100%);
    border-left: 4px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 13px 16px;
    margin: 12px 0;
    font-size: 0.84rem;
    color: #1E40AF;
    line-height: 1.75;
}
.tag {
    display: inline-block;
    font-size: 0.74rem;
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 800;
    margin-right: 4px;
}
.tag-now { background:#DBEAFE; color:#1D4ED8; }
.tag-prev { background:#E2E8F0; color:#475569; }
.tag-dyn { background:#DCFCE7; color:#166534; }
.caption {
    text-align:center;
    font-size:0.78rem;
    color:#94A3B8;
    margin-top: -4px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 12px 0 8px;">
            <div style="font-size:2.4rem;">⚾</div>
            <div style="font-size:1.1rem; font-weight:900; letter-spacing:1px;">KBO 2026</div>
            <div style="font-size:0.78rem; opacity:0.65; margin-top:2px;">포스트시즌 예측 대시보드</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
**분석 보고서**
- 2026 예측 결과
- Strategy C 모델 구조
- LOSO-CV 검증
- 데이터/피처 설계
- 중요도 및 한계점
        """)


def _open_card(title):
    st.markdown(f'<div class="report-card"><h2>{title}</h2>', unsafe_allow_html=True)


def _close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def _metric_card(label, value, sub, color):
    return f"""
    <div style="background:white;border-radius:16px;padding:20px;
                box-shadow:0 2px 12px rgba(0,0,0,0.07);
                border-top:5px solid {color};height:100%;">
        <div style="font-size:0.72rem;font-weight:700;color:#94A3B8;
                    letter-spacing:0.5px;">{label}</div>
        <div style="font-size:1.65rem;font-weight:900;color:{color};
                    letter-spacing:-0.8px;margin:6px 0 3px;">{value}</div>
        <div style="font-size:0.78rem;color:#64748B;line-height:1.5;">{sub}</div>
    </div>
    """




_sidebar()
pred_df, rank_df, importance, feature_cols = load_model_and_predict()
metrics_df, oof_probs, oof_labels, _ = run_loso_cv()

train_path = os.path.join(ROOT, "data/modeling/train_dataset.csv")
predict_path = os.path.join(ROOT, "data/modeling/predict_dataset_2026.csv")
train_df = pd.read_csv(train_path)
predict_df = pd.read_csv(predict_path)

latest = (
    pred_df
    .sort_values("date")
    .groupby("team")
    .last()
    .reset_index()
    .sort_values("prob_norm", ascending=False)
    .reset_index(drop=True)
)
top5 = set(latest.head(5)["team"])
ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y-%m-%d")
ref_date_kr = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y년 %m월 %d일")
ref_ratio = latest["games_played_ratio"].mean()
ref_games = int(latest["games"].mean())
over_50 = int((latest["prob_norm"] >= 0.5).sum())
cutline_gap = latest.iloc[4]["prob_norm"] - latest.iloc[5]["prob_norm"]
top1 = latest.iloc[0]

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
    border-radius: 18px; padding: 34px 38px 30px; color: white; margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(30,58,138,0.28);
">
    <div style="font-size:1.9rem; font-weight:900; letter-spacing:-0.5px;">
        2026 KBO 포스트시즌 예측 분석 보고서
    </div>
    <div style="font-size:0.95rem; opacity:0.82; margin-top:8px;">
        Strategy C 앙상블을 활용한 2026 시즌 포스트시즌 진출 확률 분석
    </div>
    <div style="font-size:0.78rem; opacity:0.68; margin-top:12px;">
        기준일: {ref_date_kr} | 학습 데이터: 2017~2025 | 검증: LOSO-CV 2018~2025 | 타겟 시즌: 2026 | 모델: LR + RF + lightXGB + lightLGBM
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
summary_cards = [
    (c1, "예측 1위", top1["team"], f"정규화 확률 {top1['prob_norm']:.1%}", TEAM_COLORS.get(top1["team"], "#1D4ED8")),
    (c2, "50% 이상 팀", f"{over_50}팀", "모델 기준 진출 가능권", "#0891B2"),
    (c3, "5·6위 격차", f"{cutline_gap:.1%}", f"{latest.iloc[4]['team']} vs {latest.iloc[5]['team']}", "#D97706"),
    (c4, "시즌 진행도", f"{ref_ratio:.1%}", f"평균 {ref_games}경기 소화", "#7C3AED"),
]
for col, label, value, sub, color in summary_cards:
    with col:
        st.markdown(_metric_card(label, value, sub, color), unsafe_allow_html=True)


_open_card("1. 2026 포스트시즌 진출 예측 결과")
st.markdown(f"""
<p>
본 보고서는 <strong>Strategy C 앙상블</strong>(LR·RF·lightXGB·lightLGBM 소프트 보팅)을 기준으로 작성되었습니다.
모델은 2017~2025년 10개 팀의 시즌별 날짜 단위 스냅샷을 학습해, {ref_date_kr} 기준 2026 시즌
각 팀의 포스트시즌 진출 확률을 산출합니다. 현재 평균 {ref_games}경기를 소화한 시점이므로,
확률 절대값보다 <strong>팀 간 상대 순위와 컷라인(5위·6위) 격차</strong>를 중심으로 해석하는 것이 적절합니다.
</p>
<div class="report-highlight">
핵심 요약: 현재 모델은 <strong>{top1['team']}</strong>을 진출 가능성이 가장 높은 팀으로 평가합니다.
정규화 확률 50% 이상 팀은 <strong>{over_50}팀</strong>이며, 5위 {latest.iloc[4]['team']}와
6위 {latest.iloc[5]['team']}의 확률 격차는 <strong>{cutline_gap:.1%}</strong>로 컷라인 구도는 유동적입니다.
</div>
""", unsafe_allow_html=True)

pred_table = latest[["team", "games", "wins", "losses", "win_rate", "prob_raw", "prob_norm"]].copy()
pred_table.index = range(1, len(pred_table) + 1)
pred_table.columns = ["팀", "경기", "승", "패", "승률", "원시확률", "정규화확률"]
pred_table["예측 결과"] = ["포스트시즌" if i <= 5 else "탈락 예상" for i in pred_table.index]
pred_table["승률"] = pred_table["승률"].map("{:.3f}".format)
pred_table["원시확률"] = pred_table["원시확률"].map("{:.3f}".format)
pred_table["정규화확률"] = pred_table["정규화확률"].map("{:.1%}".format)
st.dataframe(pred_table, use_container_width=True, height=385)
_close_card()


_open_card("2. Strategy C 앙상블 파이프라인")
st.markdown("""
<p>
Strategy C는 서로 다른 편향을 가진 네 모델을 소프트 보팅으로 결합해 단일 알고리즘의 약점을 보완합니다.
KBO처럼 샘플 수가 제한적인 환경에서 과적합 갭을 줄이기 위해 입력 피처를 20개로 축소하고,
XGBoost·LightGBM의 트리 깊이와 라운드 수를 의도적으로 제한했습니다.
</p>
<h3>2.1 모델 구성</h3>
<div class="report-highlight">
<strong>[1] LogisticRegression</strong> — 중앙값 대체·표준화 전처리, C=0.1 L2 규제, class_weight='balanced' 적용한 선형 기준 모델<br>
<strong>[2] RandomForest</strong> — max_depth=4, min_samples_leaf=20, class_weight='balanced'로 제한한 안정형 배깅 모델<br>
<strong>[3] lightXGB</strong> — n_estimators=40, max_depth=2, scale_pos_weight 적용의 저복잡도 XGBoost<br>
<strong>[4] lightLGBM</strong> — n_estimators=40, max_depth=2, scale_pos_weight 적용의 저복잡도 LightGBM<br>
<strong>[최종 확률]</strong> — 네 모델의 predict_proba를 단순 평균한 소프트 보팅
</div>
<h3>2.2 설계 의도</h3>
<ul>
<li><strong>LR(C=0.1)</strong>은 과도한 비선형 학습을 막는 보수적인 기준선 역할을 합니다.</li>
<li><strong>RF(max_depth=4)</strong>는 예측 분산을 낮추고, 부스팅이 놓칠 수 있는 안정적인 분기 신호를 보완합니다.</li>
<li><strong>lightXGB/lightLGBM(max_depth=2)</strong>은 비선형 패턴을 포착하되, 짧은 라운드와 얕은 깊이로 과적합 위험을 낮춥니다.</li>
<li>최종 확률은 날짜별 10팀의 원시 확률 합계를 5(진출 팀 수)로 맞추도록 정규화합니다. 즉 <code>prob_norm = prob_raw / 날짜별 합계 × 5</code>이며, 상한은 1.0입니다.</li>
</ul>
""", unsafe_allow_html=True)
_close_card()


_open_card("3. 모델 성능 및 과적합 검증")
st.markdown("""
<p>
검증은 2018~2025 시즌을 대상으로 <strong>LOSO-CV(Leave-One-Season-Out)</strong>를 적용했습니다.
매 이터레이션마다 한 시즌을 테스트셋으로 제외하고 나머지로 학습해, 특정 연도에 과하게 맞춘 모델인지를 시즌 단위로 확인합니다.
</p>
""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["성능 스코어카드", "과적합 갭", "확률 보정"])
with tab1:
    st.plotly_chart(val_scorecard_chart(metrics_df), use_container_width=True)
    st.markdown("""
    <p><strong>해석:</strong> 시즌별 지표를 나란히 비교해 특정 연도에서만 성능이 급락하거나 급등하는지 확인합니다.
    Strategy C는 단순히 Train AUC를 높이기보다 시즌이 달라져도 일관된 Test AUC를 유지하는 것을 목표로 합니다.</p>
    """, unsafe_allow_html=True)
with tab2:
    st.plotly_chart(val_overfit_gap_chart(metrics_df), use_container_width=True)
    st.markdown("""
    <p><strong>해석:</strong> Train-Test AUC 갭이 0.15 이상이라면 학습 데이터에 과하게 맞았을 가능성이 큽니다.
    피처 20개 축소·max_depth=2~4 제한·L2 강화로 이전 설정 대비 갭을 줄이는 방향으로 조정했습니다.</p>
    """, unsafe_allow_html=True)
with tab3:
    st.plotly_chart(val_calibration_chart(oof_probs, oof_labels), use_container_width=True)
    st.markdown("""
    <p><strong>해석:</strong> 캘리브레이션 곡선이 대각선에서 멀어질수록 확률 출력을 액면 그대로 신뢰하기 어렵습니다.
    시즌 초반에는 데이터가 부족해 확률이 과장될 수 있으므로 절대 수치보다 컷라인 순위 차이를 기준으로 해석하는 것이 안전합니다.</p>
    """, unsafe_allow_html=True)
_close_card()


_open_card("4. 데이터 파이프라인")
st.markdown(f"""
<h3>4.1 학습/예측 데이터</h3>
<div class="report-highlight">
<strong>학습 세트:</strong> data/modeling/train_dataset.csv — {train_df.shape[0]:,}행 × {train_df.shape[1]:,}열,
시즌 {int(train_df['season'].min())}~{int(train_df['season'].max())}<br>
<strong>예측 세트:</strong> data/modeling/predict_dataset_2026.csv — {predict_df.shape[0]:,}행 × {predict_df.shape[1]:,}열,
기준 시즌 2026<br>
<strong>타깃:</strong> 최종 순위 5위 이내면 postseason=1, 나머지는 0
</div>
<h3>4.2 피처셋 축소</h3>
<p>
전체 피처 후보에서 LOSO-CV 중요도 결과를 바탕으로 상위 20개만 선별해 사용합니다.
샘플 수가 제한적인 KBO 데이터에서 불필요한 피처를 제거해 모델 복잡도를 낮추고, 해석 가능한 핵심 신호에 집중하기 위한 결정입니다.
</p>
""", unsafe_allow_html=True)

feat_groups = pd.DataFrame([
    ["현재 시즌", 7, "rank, win_rate, games_behind_5th, wins_to_5th 등 현재 순위/승률/컷라인 지표"],
    ["전년도 prev_", 9, "prev_pythagorean_win_rate, prev_run_differential, prev_team_era 등 전년도 전력 지표"],
    ["3년 역가중 dyn_", 4, "dyn_run_differential, dyn_bb_rate 등 시즌 초반 과거 전력 보정 지표"],
], columns=["그룹", "개수", "역할"])
st.dataframe(feat_groups, use_container_width=True, hide_index=True)

st.markdown("""
<h3>4.3 dyn_ 피처 설계</h3>
<div class="report-highlight">
<code>dyn_k = (1 - games_played_ratio) × avg3yr_k</code><br>
시즌 초반에는 과거 3년 평균 전력을 강하게 반영하고, 시즌이 진행될수록 현재 시즌 성적 중심으로 이동합니다.
</div>
""", unsafe_allow_html=True)
_close_card()


_open_card("5. 모델 상세 및 피처 중요도")
st.markdown("""
<p>
피처 중요도는 트리 기반 3개 모델(RandomForest, XGBoost, LightGBM)의 feature_importances_를 각각 [0,1]로 정규화한 뒤 평균해 산출합니다.
LogisticRegression은 예측 앙상블에 포함되지만, 중요도 속성이 다른 방식으로 계산되어 집계에서 제외합니다.
</p>
""", unsafe_allow_html=True)

col_chart, col_table = st.columns([1.8, 1])
with col_chart:
    st.plotly_chart(importance_chart(importance), use_container_width=True)
with col_table:
    top_imp = importance.sort_values(ascending=False).head(20).reset_index()
    top_imp.columns = ["피처", "중요도"]
    top_imp["중요도"] = top_imp["중요도"].map("{:.4f}".format)
    st.dataframe(top_imp, use_container_width=True, hide_index=True, height=530)

top5_features = importance.sort_values(ascending=False).head(5)
feature_items = "".join(
    f"<li><code>{feat}</code>: 중요도 {val:.4f}</li>"
    for feat, val in top5_features.items()
)
st.markdown(f"""
<h3>중요 피처 해석</h3>
<ul>{feature_items}</ul>
<p>
상위 피처에 현재 순위·승률 신호와 전년도 전력 지표(prev_)가 함께 포함되어 있습니다.
이는 시즌 초반처럼 현재 성적 표본이 작은 시점에서 모델이 과거 전력 신호로 불확실성을 보정하고 있음을 보여줍니다.
</p>
""", unsafe_allow_html=True)
_close_card()


_open_card("6. 2026 시즌 예측 시각화")
st.markdown("""
<p>
아래 막대 차트는 현재 기준 10개 팀의 포스트시즌 진출 정규화 확률(prob_norm)을 비교합니다.
색이 강조된 상위 5팀이 현재 예측 포스트시즌 진출권이며, 50% 기준선은 모델이 '진출 가능성이 미진출보다 높다'고 판단하는 직관적 임계값입니다.
</p>
""", unsafe_allow_html=True)
st.plotly_chart(bar_chart(latest, top5), use_container_width=True)
_close_card()


_open_card("7. 주요 시사점 및 한계")
st.markdown(f"""
<h3>주요 발견</h3>
<ul>
<li><strong>상위권 구분:</strong> 현재 {top1['team']}가 정규화 확률 {top1['prob_norm']:.1%}로 1위이며, 상위 5팀 평균 확률은 {latest.head(5)['prob_norm'].mean():.1%}입니다.</li>
<li><strong>컷라인 경쟁:</strong> 5위 {latest.iloc[4]['team']}와 6위 {latest.iloc[5]['team']}의 격차는 {cutline_gap:.1%}로, 후반부 성적 변동에 따라 순위 역전이 가능한 범위입니다.</li>
<li><strong>피처 축소 효과:</strong> 후보 피처 중 중요도 상위 20개만 사용해 모델 복잡도를 낮추고, LOSO-CV 기준 과적합 갭 감소를 확인했습니다.</li>
<li><strong>dyn_ 신호 전환:</strong> 시즌 초반(ratio≈0)에는 최근 3년 평균 전력을 강하게 반영하고, 시즌 후반(ratio≈1)에는 현재 성적이 예측을 주도하도록 자연스럽게 전환됩니다.</li>
</ul>
<h3>한계점</h3>
<ul>
<li><strong>시즌 초반 불확실성:</strong> 현재 평균 {ref_games}경기(시즌 {ref_ratio:.1%}) 시점으로, 표본이 작아 날짜별 확률 변동이 큽니다. 경기 수가 쌓일수록 예측 안정성은 높아집니다.</li>
<li><strong>확률의 절대 해석 주의:</strong> prob_norm은 날짜별 10팀의 원시 확률 합계를 5로 정규화한 값으로, '70%'가 실제 7할 진출 가능성을 보장하지 않습니다. 팀 간 순위와 컷라인 격차를 중심으로 해석해야 합니다.</li>
<li><strong>외부 변수 부재:</strong> 부상, 트레이드, 외국인 선수 교체, 일정 강도 같은 이벤트성 정보는 현재 피처에 포함되지 않아 급격한 전력 변화를 반영하는 데 한계가 있습니다.</li>
<li><strong>운영 제안:</strong> 36경기(약 25%)·72경기(50%)·108경기(75%) 체크포인트에서 재학습을 진행하면 dyn_ 신호 전환과 함께 예측 신뢰도를 단계적으로 높일 수 있습니다.</li>
</ul>
""", unsafe_allow_html=True)
_close_card()

st.markdown(f"""
<div style="text-align:center;font-size:0.78rem;color:#94A3B8;padding:18px 0 8px;">
KBO 포스트시즌 예측 프로젝트 | 생성일: 2026-04-29 | 기준 데이터: {ref_date} | Strategy C 앙상블 리포트
</div>
""", unsafe_allow_html=True)
