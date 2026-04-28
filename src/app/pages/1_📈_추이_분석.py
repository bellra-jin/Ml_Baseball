import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import streamlit as st
import pandas as pd

from components.model import load_model_and_predict
from components.charts import trend_chart, bump_chart, TEAM_COLORS

st.set_page_config(
    page_title="추이 분석 | 2026 KBO",
    page_icon="📈",
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
</style>
""", unsafe_allow_html=True)

pred_df, rank_df, importance, _ = load_model_and_predict()

latest   = pred_df.sort_values("date").groupby("team").last().reset_index()
latest   = latest.sort_values("prob_norm", ascending=False).reset_index(drop=True)
top5     = set(latest.head(5)["team"])
ref_date = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y년 %m월 %d일")

st.markdown("## 📈 추이 분석")
st.markdown(f"**기준일:** {ref_date}")
st.markdown("<hr style='border-color:#E2E8F0; margin:0.5rem 0 1.5rem;'>", unsafe_allow_html=True)

# ── 팀 필터 ──────────────────────────────────────
all_teams = sorted(pred_df["team"].unique())
with st.expander("🔎 표시할 팀 선택 (기본: 전체)", expanded=False):
    selected = st.multiselect(
        "팀 선택",
        options=all_teams,
        default=all_teams,
        label_visibility="collapsed",
    )

if not selected:
    selected = all_teams

filtered_pred = pred_df[pred_df["team"].isin(selected)]
filtered_rank = rank_df[rank_df["team"].isin(selected)]
filtered_top5 = top5 & set(selected)

# ── 확률 추이 차트 ────────────────────────────────
st.markdown('<div class="section-title">포스트시즌 진출 확률 추이</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
진한 실선: 현재 예측 상위 5팀 &nbsp;|&nbsp;
점선: 하위 5팀 (투명도 낮춤) &nbsp;|&nbsp;
빨간 점선: 포스트시즌 기준선 (50%)
</div>
""", unsafe_allow_html=True)
st.plotly_chart(trend_chart(filtered_pred, filtered_top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 범프 차트 ─────────────────────────────────────
st.markdown('<div class="section-title">예측 순위 변화 (범프 차트)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
날짜별 예측 확률 기준 순위를 시각화합니다. &nbsp;|&nbsp;
위로 올라갈수록 순위 상승 (1위가 맨 위) &nbsp;|&nbsp;
빨간 점선: 포스트시즌 컷라인 (5위)
</div>
""", unsafe_allow_html=True)
st.plotly_chart(bump_chart(filtered_rank, filtered_top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 팀별 최신 스냅샷 ─────────────────────────────
st.markdown('<div class="section-title">팀별 최신 시점 상세</div>', unsafe_allow_html=True)

snap = (
    filtered_pred
    .sort_values("date")
    .groupby("team").last()
    .reset_index()
    .sort_values("prob_norm", ascending=False)
    .reset_index(drop=True)
)
snap.index = range(1, len(snap) + 1)

cols_show = ["team", "games", "wins", "losses", "win_rate",
             "games_behind", "recent10_win_rate", "prob_norm"]
snap_show = snap[cols_show].copy()
snap_show.columns = ["팀", "경기", "승", "패", "승률", "게임차", "최근10경기승률", "예측확률"]
snap_show["승률"]        = snap_show["승률"].map("{:.3f}".format)
snap_show["게임차"]      = snap_show["게임차"].map("{:.1f}".format)
snap_show["최근10경기승률"] = snap_show["최근10경기승률"].map("{:.3f}".format)
snap_show["예측확률"]    = snap_show["예측확률"].map("{:.1%}".format)

def row_style(row):
    s = "background-color: #EFF6FF; font-weight: bold;" if row.name <= 5 else ""
    return [s] * len(row)

st.dataframe(
    snap_show.style.apply(row_style, axis=1),
    use_container_width=True,
    height=420,
)
