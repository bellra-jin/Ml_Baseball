import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import streamlit as st
import pandas as pd

from components.model import load_model_and_predict
from components.charts import importance_chart, scatter_chart, heatmap_chart
from components.style import COMMON_CSS

st.set_page_config(
    page_title="피처 분석 | 2026 KBO",
    page_icon="🔍",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)

# ── 데이터 ───────────────────────────────────────
pred_df, _, importance, feature_cols = load_model_and_predict()

latest   = pred_df.sort_values("date").groupby("team").last().reset_index()
latest   = latest.sort_values("prob_norm", ascending=False).reset_index(drop=True)
top5     = set(latest.head(5)["team"])
ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y년 %m월 %d일")

# ── 사이드바 ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 8px;">
        <div style="font-size:2.4rem;">⚾</div>
        <div style="font-size:1.1rem; font-weight:900; letter-spacing:1px;">KBO 2026</div>
        <div style="font-size:0.78rem; opacity:0.65; margin-top:2px;">포스트시즌 예측 대시보드</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    total = importance.sum()
    grp_pct = {
        "현재 시즌": importance[~importance.index.str.startswith(("dyn_", "prev_"))].sum() / total,
        "전년도 (prev_)": importance[importance.index.str.startswith("prev_")].sum() / total,
        "3년 역가중 (dyn_)": importance[importance.index.str.startswith("dyn_")].sum() / total,
    }
    st.markdown("**피처 그룹 비중**")
    colors_grp = {"현재 시즌": "#94A3B8", "전년도 (prev_)": "#2563A8", "3년 역가중 (dyn_)": "#2E8B57"}
    for label, pct in grp_pct.items():
        c = colors_grp[label]
        st.markdown(
            f'<div style="margin:6px 0;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:3px;">'
            f'<span style="font-weight:600;">{label}</span>'
            f'<span style="font-weight:800;">{pct:.1%}</span></div>'
            f'<div style="background:rgba(255,255,255,0.15);border-radius:6px;height:7px;overflow:hidden;">'
            f'<div style="background:{c};width:{pct*100:.1f}%;height:100%;border-radius:6px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown("**범례**")
    st.markdown("""
🟢 3년 평균 역가중 피처
🔵 전년도 기록 피처
⚪ 현재 시즌 피처
    """)

# ── 헤더 ─────────────────────────────────────────
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
    border-radius: 16px; padding: 22px 30px; color: white; margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(30,58,138,0.22);
">
    <div style="font-size:1.6rem; font-weight:900; letter-spacing:-0.3px;">🔍 피처 분석</div>
    <div style="font-size:0.85rem; opacity:0.75; margin-top:4px;">
        기준일: {ref_date} &nbsp;·&nbsp; 모델이 중요하게 본 피처와 팀별 성적-예측 비교
    </div>
</div>
""", unsafe_allow_html=True)

# ── 피처 중요도 ───────────────────────────────────
st.markdown('<div class="section-title">📊 피처 중요도 Top 20</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
Strategy C의 트리 계열 3개 모델(XGBoost · LightGBM · RandomForest) 중요도 평균값입니다. &nbsp;|&nbsp;
<span style="color:#2E8B57;font-weight:700;">■ dyn_</span>: 3년 평균 역가중 &nbsp;
<span style="color:#2563A8;font-weight:700;">■ prev_</span>: 전년도 기록 &nbsp;
<span style="color:#94A3B8;font-weight:700;">■</span>: 현재 시즌
</div>
""", unsafe_allow_html=True)

col_imp, col_list = st.columns([3, 1])

with col_imp:
    st.plotly_chart(importance_chart(importance), use_container_width=True)

with col_list:
    top10 = importance.sort_values(ascending=False).head(10)
    cards = ""
    for i, (feat, val) in enumerate(top10.items(), 1):
        if feat.startswith("dyn_"):
            badge, color = "🟢", "#2E8B57"
        elif feat.startswith("prev_"):
            badge, color = "🔵", "#2563A8"
        else:
            badge, color = "⚪", "#64748B"
        cards += (
            f'<div style="padding:8px 10px;margin:0 0 6px;border-radius:8px;'
            f'background:white;box-shadow:0 1px 4px rgba(0,0,0,0.06);flex-shrink:0;">'
            f'<div style="font-size:0.7rem;color:#94A3B8;font-weight:600;">{badge} #{i}</div>'
            f'<div style="font-size:0.82rem;font-weight:700;color:{color};'
            f'word-break:break-all;">{feat}</div>'
            f'<div style="font-size:0.74rem;color:#64748B;font-weight:600;">{val:.4f}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="font-weight:700;font-size:0.95rem;color:#1E293B;'
        f'margin-bottom:8px;">Top 10 피처</div>'
        f'<div style="height:496px;overflow-y:auto; padding: 0px 0px 0px 0px;">'
        f'{cards}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── 산점도 ────────────────────────────────────────
st.markdown('<div class="section-title">🎯 현재 승률 vs 모델 예측 확률</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
대각선 <b>위</b>: 모델이 현재 성적보다 낙관적 평가 → 숨겨진 강팀 가능성 &nbsp;|&nbsp;
대각선 <b>아래</b>: 모델이 현재 성적보다 비관적 평가 → 성적 대비 전력 불안
</div>
""", unsafe_allow_html=True)
st.plotly_chart(scatter_chart(latest, top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 히트맵 ───────────────────────────────────────
st.markdown('<div class="section-title">🗓️ 팀 × 날짜 포스트시즌 확률 히트맵</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
날짜별 각 팀의 정규화 확률을 색상으로 표현합니다. &nbsp;|&nbsp;
<b>진할수록 진출 확률 높음</b> &nbsp;|&nbsp;
<b>★</b> 현재 예측 상위 5팀 &nbsp;|&nbsp;
빨간 점선: 포스트시즌 컷라인
</div>
""", unsafe_allow_html=True)
st.plotly_chart(heatmap_chart(pred_df, top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 피처 그룹 비중 카드 ───────────────────────────
st.markdown('<div class="section-title">📐 피처 그룹별 중요도 비중</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
group_data = [
    ("⚪", "현재 시즌 피처",      "#94A3B8", grp_pct["현재 시즌"]),
    ("🔵", "전년도 기록 (prev_)", "#2563A8", grp_pct["전년도 (prev_)"]),
    ("🟢", "3년 역가중 (dyn_)",   "#2E8B57", grp_pct["3년 역가중 (dyn_)"]),
]
for col, (icon, label, color, pct) in zip([c1, c2, c3], group_data):
    with col:
        st.markdown(f"""
        <div style="background:white; border-radius:16px; padding:24px 20px;
                    text-align:center; box-shadow:0 2px 12px rgba(0,0,0,0.07);
                    border-top:5px solid {color};">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-size:0.78rem;color:#94A3B8;font-weight:700;
                        margin:8px 0 4px;letter-spacing:0.5px;">{label}</div>
            <div style="font-size:2.2rem;font-weight:900;color:{color};
                        letter-spacing:-1px;">{pct:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
