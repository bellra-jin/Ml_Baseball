"""
실행 코드 cd src/app
uv run streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import pandas as pd

from components.model import load_model_and_predict
from components.charts import TEAM_COLORS, bar_chart

st.set_page_config(
    page_title="2026 KBO 포스트시즌 예측",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* 전체 배경 */
    .stApp { background-color: #F8F9FA; }

    /* 사이드바 */
    [data-testid="stSidebar"] { background-color: #1B3F7A; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #CBD5E1 !important; }

    /* 메인 헤더 */
    .hero-title {
        font-size: 2.2rem; font-weight: 800;
        color: #1B3F7A; margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1rem; color: #64748B; margin-bottom: 2rem;
    }

    /* 팀 카드 */
    .team-card {
        background: white;
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 5px solid;
        transition: transform 0.2s;
    }
    .team-card:hover { transform: translateY(-3px); }
    .team-card .rank  { font-size: 0.8rem; font-weight: 600; color: #94A3B8; letter-spacing: 1px; }
    .team-card .name  { font-size: 1.5rem; font-weight: 800; margin: 4px 0; }
    .team-card .prob  { font-size: 1.9rem; font-weight: 900; }
    .team-card .games { font-size: 0.78rem; color: #94A3B8; margin-top: 4px; }

    /* 스탯 카드 */
    .stat-card {
        background: white; border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .stat-card .label { font-size: 0.78rem; color: #94A3B8; font-weight: 600; }
    .stat-card .value { font-size: 1.6rem; font-weight: 800; color: #1B3F7A; }

    /* 섹션 제목 */
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1e293b;
        border-left: 4px solid #1B3F7A;
        padding-left: 10px; margin: 1.5rem 0 0.8rem;
    }

    /* 구분선 */
    hr { border-color: #E2E8F0; margin: 1.5rem 0; }
    .stPlotlyChart { border-radius: 12px; }
    div[data-testid="metric-container"] { background: white; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ── 사이드바 ──────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ KBO 2026")
    st.markdown("### 포스트시즌 예측")
    st.markdown("---")
    st.markdown("""
    **모델 구성**
    - XGBoost
    - LightGBM
    - RandomForest
    - 소프트 보팅 앙상블

    **피처셋 (36개)**
    - 현재 시즌 성적 18개
    - 전년도 핵심 지표 9개
    - 3년 평균 역가중 9개

    **학습 기간**
    - 2017 ~ 2025 시즌
    """)
    st.markdown("---")
    st.markdown("""
    **📌 페이지 안내**

    - **홈** — 예측 요약 & 바 차트
    - **추이 분석** — 확률 추이 & 순위 변화
    - **피처 분석** — 중요도 & 산점도 & 히트맵
    """)


# ── 데이터 로드 ───────────────────────────────────
pred_df, rank_df, importance, feature_cols = load_model_and_predict()

latest    = pred_df.sort_values("date").groupby("team").last().reset_index()
latest    = latest.sort_values("prob_norm", ascending=False).reset_index(drop=True)
top5      = set(latest.head(5)["team"])
ref_date  = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y년 %m월 %d일")
ref_ratio = latest["games_played_ratio"].mean()
ref_games = int(latest["games"].mean())


# ── 헤더 ─────────────────────────────────────────
st.markdown(f'<div class="hero-title">⚾ 2026 KBO 포스트시즌 예측</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">기준일: {ref_date} &nbsp;|&nbsp; '
    f'시즌 진행도: {ref_ratio:.1%} ({ref_games}경기) &nbsp;|&nbsp; '
    f'XGBoost + LightGBM + RandomForest 앙상블</div>',
    unsafe_allow_html=True,
)

# ── 요약 스탯 ─────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
top1 = latest.iloc[0]
with c1:
    st.metric("1위 예측팀", top1["team"], f"{top1['prob_norm']:.1%}")
with c2:
    st.metric("시즌 진행도", f"{ref_ratio:.1%}", f"{ref_games}경기 소화")
with c3:
    avg_top5 = latest.head(5)["prob_norm"].mean()
    st.metric("상위 5팀 평균 확률", f"{avg_top5:.1%}")
with c4:
    cutline_gap = latest.iloc[4]["prob_norm"] - latest.iloc[5]["prob_norm"]
    st.metric("5·6위 확률 격차", f"{cutline_gap:.1%}")

st.markdown("<hr>", unsafe_allow_html=True)

# ── 포스트시즌 예측 Top 5 카드 ────────────────────
st.markdown('<div class="section-title">포스트시즌 예측 상위 5팀</div>', unsafe_allow_html=True)

cols = st.columns(5)
rank_labels = ["1st ★", "2nd ★", "3rd ★", "4th ★", "5th ★"]

for i, col in enumerate(cols):
    row   = latest.iloc[i]
    team  = row["team"]
    color = TEAM_COLORS.get(team, "#1B3F7A")
    with col:
        st.markdown(f"""
        <div class="team-card" style="border-top-color:{color};">
            <div class="rank">{rank_labels[i]}</div>
            <div class="name" style="color:{color};">{team}</div>
            <div class="prob" style="color:{color};">{row['prob_norm']:.1%}</div>
            <div class="games">{int(row['games'])}경기 · 승률 {row['win_rate']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 바 차트 ──────────────────────────────────────
st.markdown('<div class="section-title">팀별 포스트시즌 진출 확률</div>', unsafe_allow_html=True)
st.plotly_chart(bar_chart(latest, top5), use_container_width=True)

# ── 전체 순위 테이블 ──────────────────────────────
st.markdown('<div class="section-title">전체 예측 순위표</div>', unsafe_allow_html=True)

table = latest[["team", "games", "wins", "losses", "win_rate",
                "games_played_ratio", "prob_norm", "prob_raw"]].copy()
table.index = range(1, len(table) + 1)
table.columns = ["팀", "경기", "승", "패", "승률", "시즌진행도", "예측확률(정규화)", "예측확률(원시)"]
table["승률"]          = table["승률"].map("{:.3f}".format)
table["시즌진행도"]    = table["시즌진행도"].map("{:.1%}".format)
table["예측확률(정규화)"] = table["예측확률(정규화)"].map("{:.1%}".format)
table["예측확률(원시)"]   = table["예측확률(원시)"].map("{:.4f}".format)

def highlight_top5(row):
    color = "background-color: #EFF6FF; font-weight: bold;" if row.name <= 5 else ""
    return [color] * len(row)

st.dataframe(
    table.style.apply(highlight_top5, axis=1),
    use_container_width=True,
    height=385,
)
