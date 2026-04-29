import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import streamlit as st
from components.model import run_loso_cv
from components.charts import (
    val_scorecard_chart,
    val_overfit_gap_chart,
    val_calibration_chart,
    val_checkpoint_chart,
)
from components.style import COMMON_CSS

st.set_page_config(
    page_title="검증 리포트 | 2026 KBO",
    page_icon="🧪",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)


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
**검증 리포트**
- 시즌 단위 교차검증
- 과적합 갭 점검
- 확률 보정 상태 확인
- 주요 체크포인트 적중률
        """)


def _metric_card(icon, label, value, sub, color):
    return f"""
    <div style="background:white;border-radius:16px;padding:20px;
                box-shadow:0 2px 12px rgba(0,0,0,0.07);
                border-left:5px solid {color};height:100%;">
        <div style="font-size:1.45rem;margin-bottom:6px;">{icon}</div>
        <div style="font-size:0.72rem;font-weight:700;color:#94A3B8;
                    letter-spacing:0.5px;">{label}</div>
        <div style="font-size:1.12rem;font-weight:900;color:{color};
                    margin:5px 0 3px;">{value}</div>
        <div style="font-size:0.78rem;color:#64748B;line-height:1.55;">{sub}</div>
    </div>
    """


def _txt_card(title, bullets):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""
    <div style="background:white;border-radius:16px;padding:22px 24px;
                box-shadow:0 2px 12px rgba(0,0,0,0.07);
                border-top:5px solid #2563EB;min-height:260px;
                display:flex;flex-direction:column;justify-content:center;">
        <div style="font-size:1rem;font-weight:900;color:#1E293B;margin-bottom:12px;">
            {title}
        </div>
        <ul style="margin:0;padding-left:1.15rem;color:#475569;
                   font-size:0.84rem;line-height:1.85;">
            {items}
        </ul>
    </div>
    """


_sidebar()

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
    border-radius: 16px; padding: 22px 30px; color: white; margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(30,58,138,0.22);
">
    <div style="font-size:1.6rem; font-weight:900; letter-spacing:-0.3px;">🧪 검증 리포트</div>
    <div style="font-size:0.85rem; opacity:0.75; margin-top:4px;">
        Strategy C 모델의 시즌 단위 검증 결과와 과적합 위험을 확인합니다.
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "📅", "검증 방식", "LOSO-CV", "2018~2025 각 시즌을 순서대로 테스트셋으로 분리해 8회 반복", "#1D4ED8"),
    (c2, "🧠", "모델", "Strategy C", "LR · RF · lightXGB · lightLGBM 소프트 보팅 앙상블", "#0891B2"),
    (c3, "📐", "피처셋", "Top 20", "LOSO-CV 중요도 기반으로 선별한 저복잡도 피처 구성", "#0EA5E9"),
    (c4, "🎯", "검증 초점", "일반화", "Train/Test AUC 갭과 확률 캘리브레이션을 함께 점검", "#D97706"),
]
for col, icon, label, value, sub, color in cards:
    with col:
        st.markdown(_metric_card(icon, label, value, sub, color), unsafe_allow_html=True)

metrics_df, oof_probs, oof_labels, checkpoint_df = run_loso_cv()

st.markdown('<div class="section-title">📌 검증 차트 해석</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
2018~2025 시즌을 한 번씩 테스트셋으로 분리하는 <b>LOSO-CV 결과를 직접 계산</b>한 차트입니다.
앱 실행 시 모델을 재학습하므로 현재 Strategy C 설정을 그대로 반영합니다.
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["성능 스코어카드", "과적합 갭", "캘리브레이션", "체크포인트"])

with tabs[0]:
    ch_col, txt_col = st.columns([2.25, 1])
    with ch_col:
        st.plotly_chart(val_scorecard_chart(metrics_df), use_container_width=True)
    with txt_col:
        st.markdown(_txt_card("성능 스코어카드", [
            "2018~2025 각 시즌을 테스트셋으로 분리했을 때의 AUC, F1, Precision, Recall, Brier Score를 한 화면에 비교합니다.",
            "특정 연도에서만 성능이 급락하거나 급등하는지 확인해 모델이 특정 시즌에 편향되어 있는지 점검합니다.",
            "칸 색상이 시즌 전반에 걸쳐 고르게 유지될수록 연도가 달라져도 예측 품질이 일관된다는 의미입니다.",
        ]), unsafe_allow_html=True)

with tabs[1]:
    ch_col, txt_col = st.columns([2.25, 1])
    with ch_col:
        st.plotly_chart(val_overfit_gap_chart(metrics_df), use_container_width=True)
    with txt_col:
        st.markdown(_txt_card("과적합 갭", [
            "각 시즌의 Train AUC와 Test AUC 차이(갭)를 나란히 시각화합니다.",
            "갭이 0.1 이하로 유지되면 모델이 학습 데이터에 과도하게 맞지 않았다는 신호입니다.",
            "Strategy C는 피처 20개 축소와 max_depth=2~4 제한으로 이전 설정 대비 갭을 줄이는 방향으로 설계했습니다.",
        ]), unsafe_allow_html=True)

with tabs[2]:
    ch_col, txt_col = st.columns([2.25, 1])
    with ch_col:
        st.plotly_chart(val_calibration_chart(oof_probs, oof_labels), use_container_width=True)
    with txt_col:
        st.markdown(_txt_card("캘리브레이션", [
            "모델 출력 확률(x축)과 실제 포스트시즌 진출 비율(y축)을 10구간으로 나눠 일치 정도를 확인합니다.",
            "대각선에 가까울수록 신뢰도 높은 확률 출력으로, '70%라고 했을 때 실제로도 70% 근방에서 진출'한다는 의미입니다.",
            "예측 확률이 0 또는 1 극단에 몰린다면 Platt Scaling 등 캘리브레이션 후처리가 필요한 신호입니다.",
        ]), unsafe_allow_html=True)

with tabs[3]:
    ch_col, txt_col = st.columns([2.25, 1])
    with ch_col:
        st.plotly_chart(val_checkpoint_chart(checkpoint_df), use_container_width=True)
    with txt_col:
        st.markdown(_txt_card("체크포인트 적중", [
            "시즌 50%·75%·90%·최종 시점별로 예측 상위 5팀과 실제 포스트시즌 5팀의 교집합(적중 수)을 시즌별로 집계합니다.",
            "후반 체크포인트로 갈수록 적중 수가 증가하면, 현재 성적이 쌓이면서 예측이 실제에 수렴하고 있다는 신호입니다.",
            "5팀 중 4팀 이상을 맞추는 시즌 비율이 체크포인트별로 어떻게 달라지는지 비교해 모델의 후반 신뢰도를 확인합니다.",
        ]), unsafe_allow_html=True)
