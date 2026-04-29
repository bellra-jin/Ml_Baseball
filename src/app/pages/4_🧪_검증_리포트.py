import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import streamlit as st
from components.style import COMMON_CSS

ROOT = os.path.join(os.path.dirname(__file__), "../../..")
VALDIR = os.path.join(ROOT, "notebooks/experiments/jh/kbo_prediction_2026/validation")

st.set_page_config(
    page_title="검증 리포트 | 2026 KBO",
    page_icon="🧪",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)


def _asset_path(filename):
    return os.path.join(VALDIR, filename)


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


def _show_validation_block(filename, title, bullets):
    path = _asset_path(filename)
    img_col, txt_col = st.columns([2.25, 1])

    with img_col:
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.warning(f"검증 이미지가 없습니다: `{path}`")

    with txt_col:
        items = "".join(f"<li>{b}</li>" for b in bullets)
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:22px 24px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.07);
                    border-top:5px solid #2563EB;height:390px;
                    display:flex;flex-direction:column;justify-content:center;">
            <div style="font-size:1rem;font-weight:900;color:#1E293B;margin-bottom:12px;">
                {title}
            </div>
            <ul style="margin:0;padding-left:1.15rem;color:#475569;
                       font-size:0.84rem;line-height:1.85;">
                {items}
            </ul>
        </div>
        """, unsafe_allow_html=True)


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
    (c1, "📅", "검증 방식", "LOSO-CV", "2018~2025 시즌을 한 시즌씩 테스트셋으로 분리", "#1D4ED8"),
    (c2, "🧠", "모델", "Strategy C", "LR · RF · lightXGB · lightLGBM 소프트 보팅", "#0891B2"),
    (c3, "📐", "피처셋", "Top 20", "중요도 기반으로 축소한 저과적합 피처 구성", "#2E8B57"),
    (c4, "🎯", "검증 초점", "일반화", "성능보다 과적합 갭과 확률 신뢰도를 함께 확인", "#D97706"),
]
for col, icon, label, value, sub, color in cards:
    with col:
        st.markdown(_metric_card(icon, label, value, sub, color), unsafe_allow_html=True)

st.markdown('<div class="section-title">📌 검증 차트 해석</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
아래 차트는 <b>notebooks/experiments/jh/kbo_prediction_2026/validation</b>에 저장된 검증 결과입니다.
화면의 예측 모델과 같은 Strategy C 설정으로 생성된 자료를 사용합니다.
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["성능 스코어카드", "과적합 갭", "로스 커브", "캘리브레이션", "체크포인트"])

with tabs[0]:
    _show_validation_block(
        "val_scorecard.png",
        "성능 스코어카드",
        [
            "각 시즌을 테스트셋으로 뺐을 때의 AUC, F1, Precision, Recall, Brier 등을 한 번에 비교합니다.",
            "특정 시즌에서만 성능이 크게 흔들리는지 확인해 모델이 특정 연도에 과하게 맞춰졌는지 점검합니다.",
            "밝고 안정적인 칸이 많을수록 시즌이 바뀌어도 예측 품질이 유지된다는 의미입니다.",
        ],
    )

with tabs[1]:
    _show_validation_block(
        "val_overfit_gap.png",
        "과적합 갭",
        [
            "Train AUC와 Test AUC의 차이를 시즌별로 비교합니다.",
            "갭이 작을수록 학습 데이터에만 과하게 맞춘 모델이 아니라는 신호입니다.",
            "이번 Strategy C는 피처 축소와 얕은 모델 설정으로 기존보다 과적합을 줄이는 방향입니다.",
        ],
    )

with tabs[2]:
    _show_validation_block(
        "val_loss_curves.png",
        "로스 커브",
        [
            "XGBoost와 LightGBM의 부스팅 라운드별 logloss 흐름을 확인합니다.",
            "검증 손실이 빠르게 악화되면 모델이 불필요하게 복잡해졌다는 신호로 볼 수 있습니다.",
            "lightXGB와 lightLGBM은 라운드와 깊이를 낮춰 손실 곡선이 과하게 벌어지는 것을 억제합니다.",
        ],
    )

with tabs[3]:
    _show_validation_block(
        "val_calibration.png",
        "캘리브레이션",
        [
            "모델이 말한 확률과 실제 포스트시즌 진출 비율이 얼마나 가까운지 확인합니다.",
            "대각선에 가까울수록 70%라고 예측한 경우 실제로도 비슷한 빈도로 맞는다는 뜻입니다.",
            "확률 분포까지 함께 보면서 모델이 지나치게 0 또는 1에 몰리는지도 확인합니다.",
        ],
    )

with tabs[4]:
    _show_validation_block(
        "val_checkpoint.png",
        "체크포인트 적중",
        [
            "시즌 50%, 75%, 90%, 최종 시점에서 상위 5팀 예측이 실제 포스트시즌 팀과 얼마나 겹쳤는지 봅니다.",
            "초반보다 후반 체크포인트에서 적중이 올라가면 현재 시즌 정보가 점점 잘 반영된다는 의미입니다.",
            "실제 서비스 화면에서는 이 체크포인트 관점이 가장 직관적인 신뢰도 설명이 됩니다.",
        ],
    )
