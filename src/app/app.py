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
from components.style import COMMON_CSS

st.set_page_config(
    page_title="2026 KBO 포스트시즌 예측",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)


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
#     st.markdown("""
# **📌 페이지 안내**
# - **홈** — 예측 요약 & 순위
# - **📈 추이 분석** — 확률 추이 & 순위 변화
# - **🔍 피처 분석** — 중요도 & 산점도 & 히트맵
# - **ℹ️ 모델 소개** — 모델 구성 & 피처 정의
#     """)


# ── 데이터 로드 ───────────────────────────────────
pred_df, rank_df, importance, feature_cols = load_model_and_predict()

latest    = pred_df.sort_values("date").groupby("team").last().reset_index()
latest    = latest.sort_values("prob_norm", ascending=False).reset_index(drop=True)
top5      = set(latest.head(5)["team"])
ref_date  = pd.to_datetime(latest["date"].iloc[0]).strftime("%Y년 %m월 %d일")
ref_ratio = latest["games_played_ratio"].mean()
ref_games = int(latest["games"].mean())


# ── 히어로 배너 ───────────────────────────────────
progress_pct = ref_ratio * 100
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
    border-radius: 20px;
    padding: 28px 36px 22px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(30,58,138,0.28);
    position: relative;
    overflow: hidden;
">
    <div style="position:absolute;top:-50px;right:-30px;width:220px;height:220px;
                background:rgba(255,255,255,0.05);border-radius:50%;pointer-events:none;"></div>
    <div style="position:absolute;bottom:-70px;right:140px;width:160px;height:160px;
                background:rgba(255,255,255,0.03);border-radius:50%;pointer-events:none;"></div>
    <div style="position:relative;">
        <div style="font-size:0.72rem;font-weight:700;opacity:0.6;letter-spacing:3px;
                    text-transform:uppercase;margin-bottom:6px;">2026 KBO LEAGUE</div>
        <div style="font-size:1.95rem;font-weight:900;letter-spacing:-0.5px;line-height:1.2;
                    margin-bottom:6px;">
            ⚾ 포스트시즌 진출 예측
        </div>
        <div style="font-size:0.86rem;opacity:0.75;margin-bottom:22px;">
            기준일: {ref_date} &nbsp;·&nbsp; Strategy C: LR + RF + lightXGB + lightLGBM 앙상블
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.74rem;
                    opacity:0.72;margin-bottom:7px;">
            <span>시즌 시작</span>
            <span style="font-weight:700;opacity:1;font-size:0.82rem;">
                {ref_ratio:.1%} 진행 중 &nbsp;({ref_games}경기 소화)
            </span>
            <span>시즌 종료 (144경기)</span>
        </div>
        <div style="background:rgba(255,255,255,0.20);border-radius:10px;height:13px;
                    overflow:hidden;position:relative;">
            <div style="background:linear-gradient(90deg,#60A5FA,#BAE6FD);
                        width:{progress_pct:.2f}%;height:100%;border-radius:10px;
                        box-shadow:0 0 10px rgba(147,197,253,0.55);"></div>
            <div style="position:absolute;top:0;left:50%;width:1px;height:100%;
                        background:rgba(255,255,255,0.38);"></div>
            <div style="position:absolute;top:0;left:75%;width:1px;height:100%;
                        background:rgba(255,255,255,0.38);"></div>
        </div>
        <div style="position:relative;height:18px;font-size:0.68rem;opacity:0.52;margin-top:4px;">
            <span style="position:absolute;left:50%;transform:translateX(-50%);">50%</span>
            <span style="position:absolute;left:75%;transform:translateX(-50%);">75%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── 요약 스탯 ─────────────────────────────────────
top1 = latest.iloc[0]
avg_top5     = latest.head(5)["prob_norm"].mean()
cutline_gap  = latest.iloc[4]["prob_norm"] - latest.iloc[5]["prob_norm"]

top1_color = TEAM_COLORS.get(top1["team"], "#1B3F7A")

def _stat_card(icon, label, value, sub, accent):
    return f"""
    <div style="
        background: white;
        border-radius: 18px;
        padding: 22px 20px 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.07);
        border-top: 5px solid {accent};
        display: flex; flex-direction: column; gap: 6px;
        height: 100%;
    ">
        <div style="
            width: 36px; height: 36px; border-radius: 10px;
            background: {accent}18;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.1rem; margin-bottom: 2px;
        ">{icon}</div>
        <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8;
                    letter-spacing: 0.5px; text-transform: uppercase;">{label}</div>
        <div style="font-size: 1.85rem; font-weight: 900; color: {accent};
                    letter-spacing: -1px; line-height: 1.1;">{value}</div>
        <div style="font-size: 0.78rem; color: #64748B; font-weight: 500;
                    margin-top: 2px;">{sub}</div>
    </div>
    """

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "🏆", "1위 예측팀",        top1["team"],          f"진출 확률 {top1['prob_norm']:.1%}",  top1_color),
    (c2, "📅", "시즌 진행도",        f"{ref_ratio:.1%}",    f"{ref_games}경기 소화",               "#7C3AED"),
    (c3, "📊", "상위 5팀 평균 확률", f"{avg_top5:.1%}",     "Top 5 평균",                          "#0891B2"),
    (c4, "✂️", "5 · 6위 확률 격차", f"{cutline_gap:.1%}",  "컷라인 격차",                         "#D97706"),
]
for col, icon, label, value, sub, accent in cards:
    with col:
        st.markdown(_stat_card(icon, label, value, sub, accent), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ── 포스트시즌 예측 Top 5 카드 ────────────────────
st.markdown('<div class="section-title">⭐ 포스트시즌 예측 상위 5팀</div>', unsafe_allow_html=True)

rank_icons  = ["🥇", "🥈", "🥉", "4th", "5th"]
rank_labels = ["1위", "2위", "3위", "4위", "5위"]

cols = st.columns(5)
for i, col in enumerate(cols):
    row   = latest.iloc[i]
    team  = row["team"]
    color = TEAM_COLORS.get(team, "#1B3F7A")
    icon  = rank_icons[i]
    label = rank_labels[i]
    # 1위는 카드를 살짝 더 크게 강조
    scale = "scale(1.04)" if i == 0 else "scale(1)"
    with col:
        badge = f"{icon}&nbsp;{label}" if i < 3 else label
        st.markdown(
            f'<div style="background:linear-gradient(145deg,{color}F2 0%,{color}BB 100%);'
            f'border-radius:20px;padding:26px 14px 20px;text-align:center;'
            f'box-shadow:0 6px 24px {color}55;position:relative;overflow:hidden;'
            f'transform:{scale};transform-origin:bottom center;">'
            f'<div style="position:absolute;bottom:-14px;right:4px;font-size:5.5rem;'
            f'font-weight:900;color:rgba(255,255,255,0.10);line-height:1;pointer-events:none;'
            f'letter-spacing:-4px;">{i+1}</div>'
            f'<div style="display:inline-flex;align-items:center;gap:4px;'
            f'background:rgba(255,255,255,0.22);border-radius:20px;padding:3px 10px;'
            f'margin-bottom:14px;font-size:0.72rem;font-weight:800;color:white;'
            f'letter-spacing:0.5px;">{badge}</div>'
            f'<div style="font-size:1.45rem;font-weight:900;color:white;letter-spacing:-0.5px;'
            f'margin-bottom:6px;text-shadow:0 2px 8px rgba(0,0,0,0.18);">{team}</div>'
            f'<div style="font-size:2.1rem;font-weight:900;color:white;letter-spacing:-1.5px;'
            f'line-height:1;text-shadow:0 2px 12px rgba(0,0,0,0.20);">{row["prob_norm"]:.1%}</div>'
            f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.72);margin-top:10px;'
            f'font-weight:500;">{int(row["games"])}경기 &nbsp;·&nbsp; 승률 {row["win_rate"]:.3f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ── 바 차트 ──────────────────────────────────────
st.markdown('<div class="section-title">📊 팀별 포스트시즌 진출 확률</div>', unsafe_allow_html=True)
st.plotly_chart(bar_chart(latest, top5), use_container_width=True)


# ── 전체 순위 테이블 ──────────────────────────────
st.markdown('<div class="section-title">📋 전체 예측 순위표</div>', unsafe_allow_html=True)

table = latest[["team", "games", "wins", "losses", "win_rate",
                "games_played_ratio", "prob_norm", "prob_raw"]].copy()
table.index = range(1, len(table) + 1)
table.columns = ["팀", "경기", "승", "패", "승률", "시즌진행도", "예측확률(정규화)", "예측확률(원시)"]
table["승률"]             = table["승률"].map("{:.3f}".format)
table["시즌진행도"]       = table["시즌진행도"].map("{:.1%}".format)
table["예측확률(정규화)"] = table["예측확률(정규화)"].map("{:.1%}".format)
table["예측확률(원시)"]   = table["예측확률(원시)"].map("{:.4f}".format)

def highlight_top5(row):
    s = "background-color:#EFF6FF; font-weight:bold;" if row.name <= 5 else ""
    return [s] * len(row)

st.dataframe(
    table.style.apply(highlight_top5, axis=1),
    use_container_width=True,
    height=385,
)
