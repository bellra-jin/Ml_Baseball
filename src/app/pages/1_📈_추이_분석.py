import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import streamlit as st
import pandas as pd

from components.model import load_model_and_predict
from components.charts import trend_chart, bump_chart, TEAM_COLORS
from components.style import COMMON_CSS

st.set_page_config(
    page_title="추이 분석 | 2026 KBO",
    page_icon="📈",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)

# ── 데이터 ───────────────────────────────────────
pred_df, rank_df, _, _ = load_model_and_predict()

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

    all_teams = sorted(pred_df["team"].unique())
    st.markdown("**🔎 팀 필터**")
    selected = st.multiselect(
        "표시할 팀 선택",
        options=all_teams,
        default=all_teams,
        label_visibility="collapsed",
    )
    if not selected:
        selected = all_teams

    st.markdown("---")
    st.markdown("**현재 상위 5팀**")
    for i, row in latest.head(5).iterrows():
        team  = row["team"]
        color = TEAM_COLORS.get(team, "#888")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{color};'
            f'flex-shrink:0;"></div>'
            f'<span style="font-weight:700;">{i+1}. {team}</span>'
            f'<span style="opacity:0.7;margin-left:auto;">{row["prob_norm"]:.1%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── 헤더 ─────────────────────────────────────────
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
    border-radius: 16px; padding: 22px 30px; color: white; margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(30,58,138,0.22);
">
    <div style="font-size:1.6rem; font-weight:900; letter-spacing:-0.3px;">📈 추이 분석</div>
    <div style="font-size:0.85rem; opacity:0.75; margin-top:4px;">
        기준일: {ref_date} &nbsp;·&nbsp; 시즌 경과에 따른 포스트시즌 확률 및 순위 변화
    </div>
</div>
""", unsafe_allow_html=True)

# ── 필터 적용 ─────────────────────────────────────
filtered_pred = pred_df[pred_df["team"].isin(selected)]
filtered_rank = rank_df[rank_df["team"].isin(selected)]
filtered_top5 = top5 & set(selected)

# ── 확률 추이 차트 ────────────────────────────────
st.markdown('<div class="section-title">📉 포스트시즌 진출 확률 추이</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<b>진한 실선 + 색상 채움</b>: 현재 예측 상위 5팀 &nbsp;|&nbsp;
<b>점선 (연한 색)</b>: 하위 5팀 &nbsp;|&nbsp;
<b>빨간 점선</b>: 포스트시즌 기준선 (50%)
</div>
""", unsafe_allow_html=True)
st.plotly_chart(trend_chart(filtered_pred, filtered_top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 범프 차트 ─────────────────────────────────────
st.markdown('<div class="section-title">🔀 예측 순위 변화 (범프 차트)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
날짜별 예측 확률 기준 순위 변화를 시각화합니다. &nbsp;|&nbsp;
<b>위로 올라갈수록 순위 상승</b> (1위가 맨 위) &nbsp;|&nbsp;
<b>빨간 점선</b>: 포스트시즌 컷라인 (5위)
</div>
""", unsafe_allow_html=True)
st.plotly_chart(bump_chart(filtered_rank, filtered_top5), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 팀별 최신 스냅샷 ─────────────────────────────
st.markdown('<div class="section-title">📋 팀별 최신 시점 상세</div>', unsafe_allow_html=True)

snap = (
    filtered_pred
    .sort_values("date")
    .groupby("team").last()
    .reset_index()
    .sort_values("prob_norm", ascending=False)
    .reset_index(drop=True)
)
snap.index = range(1, len(snap) + 1)

snap_show = snap[["team", "games", "wins", "losses", "win_rate",
                   "games_behind", "recent10_win_rate", "prob_norm"]].copy()
snap_show.columns = ["팀", "경기", "승", "패", "승률", "게임차", "최근10경기승률", "예측확률"]
snap_show["승률"]           = snap_show["승률"].map("{:.3f}".format)
snap_show["게임차"]         = snap_show["게임차"].map("{:.1f}".format)
snap_show["최근10경기승률"] = snap_show["최근10경기승률"].map("{:.3f}".format)
snap_show["예측확률"]       = snap_show["예측확률"].map("{:.1%}".format)

def row_style(row):
    s = "background-color:#EFF6FF; font-weight:bold;" if row.name <= 5 else ""
    return [s] * len(row)

st.dataframe(
    snap_show.style.apply(row_style, axis=1),
    use_container_width=True,
    height=420,
)
