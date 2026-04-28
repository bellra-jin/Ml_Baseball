import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import streamlit as st
import pandas as pd

from components.model import load_model_and_predict
from components.charts import importance_chart, scatter_chart, heatmap_chart

st.set_page_config(
    page_title="피처 분석 | 2026 KBO",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background-color: #1B3F7A; }
    [data-testid="stSidebar"] * { color: white !important; }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1e293b;
        border-left: 4px solid #1B3F7A;
        padding-left: 10px; margin: 1.5rem 0 0.8rem;
    }
    .info-box {
        background: white; border-radius: 10px;
        padding: 14px 18px; font-size: 0.88rem; color: #475569;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .legend-dot {
        display: inline-block; width: 12px; height: 12px;
        border-radius: 50%; margin-right: 6px; vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

pred_df, rank_df, importance, feature_cols = load_model_and_predict()

latest   = pred_df.sort_values("date").groupby("team").last().reset_index()
latest   = latest.sort_values("prob_norm", ascending=False).reset_index(drop=True)
top5     = set(latest.head(5)["team"])
ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y년 %m월 %d일")

st.markdown("## 🔍 피처 분석")
st.markdown(f"**기준일:** {ref_date}")
st.markdown("<hr style='border-color:#E2E8F0; margin:0.5rem 0 1.5rem;'>", unsafe_allow_html=True)

# ── 피처 중요도 ───────────────────────────────────
st.markdown('<div class="section-title">피처 중요도 Top 20</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
XGBoost · LightGBM · RandomForest 3개 모델의 정규화된 중요도 평균값입니다. &nbsp;|&nbsp;
<span style="color:#2E8B57; font-weight:600;">■ dyn_</span>: 3년 평균 역가중 피처 &nbsp;
<span style="color:#2563A8; font-weight:600;">■ prev_</span>: 전년도 기록 피처 &nbsp;
<span style="color:#888; font-weight:600;">■</span>: 현재 시즌 피처
</div>
""", unsafe_allow_html=True)

col_imp, col_top = st.columns([3, 1])
with col_imp:
    st.plotly_chart(importance_chart(importance), use_container_width=True)

with col_top:
    st.markdown("**Top 10 피처**")
    top10 = importance.sort_values(ascending=False).head(10)
    for i, (feat, val) in enumerate(top10.items(), 1):
        if feat.startswith("dyn_"):
            badge = "🟢"
        elif feat.startswith("prev_"):
            badge = "🔵"
        else:
            badge = "⚪"
        st.markdown(f"{i}. {badge} `{feat}`  \n&nbsp;&nbsp;&nbsp;&nbsp;**{val:.4f}**")

st.markdown("<br>", unsafe_allow_html=True)

# ── 산점도 ────────────────────────────────────────
st.markdown('<div class="section-title">현재 승률 vs 모델 예측 확률</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
대각선 <b>위</b>: 모델이 현재 성적보다 낙관적으로 평가 (숨겨진 강팀 가능성) &nbsp;|&nbsp;
대각선 <b>아래</b>: 모델이 현재 성적보다 비관적으로 평가 (성적 대비 전력 불안)
</div>
""", unsafe_allow_html=True)
st.plotly_chart(scatter_chart(latest, top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 히트맵 ───────────────────────────────────────
st.markdown('<div class="section-title">팀 × 날짜 포스트시즌 확률 히트맵</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
날짜별 각 팀의 정규화 확률을 색상으로 표현합니다. &nbsp;|&nbsp;
진할수록 진출 확률 높음 &nbsp;|&nbsp;
★ 현재 예측 상위 5팀 &nbsp;|&nbsp;
빨간 점선: 포스트시즌 컷라인
</div>
""", unsafe_allow_html=True)
st.plotly_chart(heatmap_chart(pred_df, top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 피처 상관 요약 ────────────────────────────────
st.markdown('<div class="section-title">피처 그룹별 중요도 비중</div>', unsafe_allow_html=True)

total = importance.sum()
grp = {
    "현재 시즌 피처": importance[~importance.index.str.startswith(("dyn_", "prev_"))].sum() / total,
    "전년도 기록 (prev_)": importance[importance.index.str.startswith("prev_")].sum() / total,
    "3년 평균 역가중 (dyn_)": importance[importance.index.str.startswith("dyn_")].sum() / total,
}

c1, c2, c3 = st.columns(3)
icons = ["📊", "📋", "🔄"]
colors = ["#888888", "#2563A8", "#2E8B57"]
for col, (label, pct), icon, color in zip([c1, c2, c3], grp.items(), icons, colors):
    with col:
        st.markdown(f"""
        <div style="background:white; border-radius:12px; padding:20px;
                    text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.07);
                    border-top: 4px solid {color};">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="font-size:0.8rem; color:#94A3B8; font-weight:600; margin:6px 0 2px;">{label}</div>
            <div style="font-size:2rem; font-weight:900; color:{color};">{pct:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
