import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import streamlit as st
from components.style import COMMON_CSS

st.set_page_config(
    page_title="모델 소개 | 2026 KBO",
    page_icon="ℹ️",
    layout="wide",
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

# ── 헤더 ─────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
    border-radius: 16px; padding: 22px 30px; color: white; margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(30,58,138,0.22);
">
    <div style="font-size:1.6rem; font-weight:900; letter-spacing:-0.3px;">ℹ️ 모델 소개</div>
    <div style="font-size:0.85rem; opacity:0.75; margin-top:4px;">
        예측 모델의 구성 방식, 피처 정의, 학습 데이터를 설명합니다.
    </div>
</div>
""", unsafe_allow_html=True)


# ── 모델 구성 ─────────────────────────────────────
st.markdown('<div class="section-title">🤖 모델 구성</div>', unsafe_allow_html=True)

def _model_card(icon, name, color, desc):
    return (
        f'<div style="background:white;border-radius:16px;padding:16px 20px;'
        f'box-shadow:0 2px 12px rgba(0,0,0,0.07);border-top:5px solid {color};'
        f'height:200px;display:flex;flex-direction:column;">'
        f'<div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>'
        f'<div style="font-size:1.0rem;font-weight:900;color:{color};margin-bottom:8px;">{name}</div>'
        f'<div style="font-size:0.82rem;color:#475569;line-height:1.7;flex:1;overflow:hidden;">{desc}</div>'
        f'</div>'
    )

c1, c2, c3, c4 = st.columns(4)
model_cards = [
    (c1, "📈", "LogisticRegression", "#0891B2",
     "C=0.1 L2 규제와 class_weight='balanced'를 적용한 선형 기준 모델. 시즌 초반 표본이 작은 환경에서도 안정적인 기준 확률을 제공합니다."),
    (c2, "🌳", "RandomForest",       "#7C3AED",
     "max_depth=4, min_samples_leaf=20으로 제한한 배깅 앙상블. 예측 분산을 낮추고 날짜별 확률 추이를 안정적으로 유지합니다."),
    (c3, "🌲", "lightXGB",           "#16A34A",
     "n_estimators=40, max_depth=2의 저복잡도 XGBoost. 비선형 패턴을 포착하면서도 라운드와 깊이를 제한해 과적합을 억제합니다."),
    (c4, "💡", "lightLGBM",          "#D97706",
     "n_estimators=40, max_depth=2의 저복잡도 LightGBM. leaf-wise 분기로 XGBoost와 독립적인 관점에서 동일 신호를 교차 검증합니다."),
]
for col, icon, name, color, desc in model_cards:
    with col:
        st.markdown(_model_card(icon, name, color, desc), unsafe_allow_html=True)

st.markdown("""
<div style="background:white;border-radius:16px;padding:22px 26px;
            box-shadow:0 2px 12px rgba(0,0,0,0.07);margin-top:16px;
            border-left:5px solid #1D4ED8;">
    <div style="font-size:0.9rem;font-weight:800;color:#1D4ED8;margin-bottom:14px;">
        💬 이 모델 조합을 선택한 이유
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div>
            <div style="font-size:0.82rem;font-weight:700;color:#1E293B;margin-bottom:4px;">
                📌 KBO 데이터의 특성
            </div>
            <div style="font-size:0.82rem;color:#475569;line-height:1.8;">
                KBO 데이터는 10팀 × 9시즌 × 날짜 단위 스냅샷으로 구성되어 실제 학습
                가능한 행 수가 수천 건에 불과합니다. 대규모 데이터를 전제로 하는 딥러닝보다
                <b>소규모 테이블 데이터에서 강점을 보이는 트리 기반 앙상블</b>이 적합합니다.
            </div>
        </div>
        <div>
            <div style="font-size:0.82rem;font-weight:700;color:#1E293B;margin-bottom:4px;">
                📌 단일 모델의 한계 보완
            </div>
            <div style="font-size:0.82rem;color:#475569;line-height:1.8;">
                LR은 과도한 비선형 학습을 막는 보수적 기준선을 제공하고, XGBoost·LightGBM은
                비선형 패턴을 포착하며, RF는 예측 분산을 낮춥니다. 네 모델의 predict_proba를
                평균하는 <b>소프트 보팅</b>으로 편향-분산 균형을 맞췄습니다.
            </div>
        </div>
        <div>
            <div style="font-size:0.82rem;font-weight:700;color:#1E293B;margin-bottom:4px;">
                📌 피처 중요도 해석 가능성
            </div>
            <div style="font-size:0.82rem;color:#475569;line-height:1.8;">
                피처 중요도는 트리 기반 3개 모델(RF, XGBoost, LightGBM)의 feature_importances_를
                각각 [0,1]로 정규화한 뒤 <b>평균</b>해 산출합니다. LR은 예측 앙상블에 포함되지만
                중요도 속성이 다른 방식으로 계산되어 집계에서 제외합니다.
            </div>
        </div>
        <div>
            <div style="font-size:0.82rem;font-weight:700;color:#1E293B;margin-bottom:4px;">
                📌 클래스 불균형 대응
            </div>
            <div style="font-size:0.82rem;color:#475569;line-height:1.8;">
                포스트시즌 진출팀(5팀)은 전체(10팀)의 절반이지만, 날짜별 스냅샷 단위로는
                불균형이 더 심해집니다. LR·RF는 <b>class_weight='balanced'</b>, 부스팅 모델은
                <b>scale_pos_weight</b>로 이 불균형을 보정했습니다.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── 학습 데이터 ───────────────────────────────────
st.markdown('<div class="section-title">📂 학습 데이터</div>', unsafe_allow_html=True)

d1, d2, d3 = st.columns(3)
data_cards = [
    (d1, "📅", "학습 기간",   "#1D4ED8", "2017 ~ 2025 시즌", "총 9개 시즌 데이터 활용"),
    (d2, "🏷️", "레이블",     "#16A34A", "포스트시즌 진출 여부", "시즌 최종 순위 5위 이내 = 1"),
    (d3, "⚖️", "클래스 균형", "#D97706", "balanced + scale_pos_weight", "진출 / 미진출 불균형 보정"),
]
for col, icon, label, color, val, sub in data_cards:
    with col:
        st.markdown(
            f'<div style="background:white;border-radius:16px;padding:20px;'
            f'box-shadow:0 2px 12px rgba(0,0,0,0.07);border-left:5px solid {color};">'
            f'<div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>'
            f'<div style="font-size:0.72rem;font-weight:700;color:#94A3B8;letter-spacing:0.5px;">{label}</div>'
            f'<div style="font-size:1.05rem;font-weight:900;color:{color};margin:4px 0 2px;">{val}</div>'
            f'<div style="font-size:0.78rem;color:#64748B;">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ── 피처셋 ────────────────────────────────────────
st.markdown('<div class="section-title">📐 피처셋 (총 20개)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
피처는 <b>3개 그룹</b>으로 구성됩니다. &nbsp;|&nbsp;
<span style="color:#1D4ED8;font-weight:700;">■ 현재 시즌</span>: 7개 &nbsp;
<span style="color:#2563A8;font-weight:700;">■ 전년도 (prev_)</span>: 9개 &nbsp;
<span style="color:#0EA5E9;font-weight:700;">■ 3년 역가중 (dyn_)</span>: 4개
</div>
""", unsafe_allow_html=True)

feat_groups = [
    {
        "title": "⚪ 현재 시즌 성적 (7개)",
        "color": "#1D4ED8",
        "bg": "#EFF6FF",
        "features": [
            ("rank",                "현재 순위"),
            ("win_rate",            "누적 승률"),
            ("games_behind",        "1위와의 게임차"),
            ("games_behind_5th",    "5위와의 게임차 (음수 = 5위권 내)"),
            ("wins_to_5th",         "5위 추월에 필요한 잔여 승수"),
            ("home_win_rate",       "홈 승률"),
            ("away_win_rate",       "원정 승률"),
        ],
    },
    {
        "title": "🔵 전년도 핵심 지표 — prev_ (9개)",
        "color": "#2563A8",
        "bg": "#EFF6FF",
        "features": [
            ("prev_pythagorean_win_rate",  "운을 제거한 기대 승률 (종합 전력)"),
            ("prev_run_differential",      "득실차 (공격 + 수비 종합)"),
            ("prev_team_era",              "팀 ERA (투수진 품질)"),
            ("prev_k_bb_ratio",            "탈삼진 / 볼넷 비율 (제구력)"),
            ("prev_top5_hitter_ops_avg",   "주전 타자 OPS 평균 (타격력)"),
            ("prev_ace_era",               "에이스 ERA (1선발 품질)"),
            ("prev_iso",                   "ISO — 순수 장타력"),
            ("prev_ops_concentration",     "타선 균형도 (낮을수록 균형)"),
            ("prev_bb_rate",               "볼넷 비율 (선구안)"),
        ],
    },
    {
        "title": "🟢 3년 평균 역가중 지표 — dyn_ (4개)",
        "color": "#0EA5E9",
        "bg": "#F0F9FF",
        "features": [
            ("dyn_pythagorean_win_rate",  "피타고라스 승률 — 3년 역가중"),
            ("dyn_run_differential",      "득실차 — 3년 역가중"),
            ("dyn_k_bb_ratio",            "탈삼진 / 볼넷 비율 — 3년 역가중"),
            ("dyn_bb_rate",               "볼넷 비율 — 3년 역가중"),
        ],
    },
]

for grp in feat_groups:
    st.markdown(
        f'<div style="background:{grp["bg"]};border-radius:14px;padding:18px 20px 12px;'
        f'margin-bottom:16px;border-left:5px solid {grp["color"]};">'
        f'<div style="font-size:0.95rem;font-weight:800;color:{grp["color"]};'
        f'margin-bottom:12px;">{grp["title"]}</div>',
        unsafe_allow_html=True,
    )
    rows = ""
    for name, desc in grp["features"]:
        rows += (
            f'<div style="display:flex;gap:12px;padding:6px 0;'
            f'border-bottom:1px solid rgba(0,0,0,0.05);align-items:baseline;">'
            f'<code style="font-size:0.75rem;background:{grp["color"]}18;color:{grp["color"]};'
            f'padding:2px 7px;border-radius:5px;white-space:nowrap;font-weight:700;">{name}</code>'
            f'<span style="font-size:0.82rem;color:#475569;">{desc}</span>'
            f'</div>'
        )
    st.markdown(rows + "</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── dyn_ 피처 설명 ────────────────────────────────
st.markdown('<div class="section-title">🔄 dyn_ 피처란?</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:white;border-radius:16px;padding:24px 26px;
            box-shadow:0 2px 12px rgba(0,0,0,0.07);">
    <div style="font-size:0.9rem;color:#1E293B;line-height:1.9;">
        <b style="color:#0EA5E9;">dyn_ 피처</b>는 시즌 초반에는 최근 3년 평균 전력을 강하게 반영하고,
        경기가 쌓일수록 현재 시즌 성적이 예측을 주도하도록 설계된 <b>동적 가중 지표</b>입니다.<br><br>
        <div style="background:#F0F9FF;border-radius:10px;padding:14px 18px;
                    font-family:monospace;font-size:0.85rem;color:#166534;
                    border-left:4px solid #0EA5E9;margin:8px 0 12px;">
            dyn_X = (1 − games_played_ratio) × avg3yr_X
        </div>
        <ul style="margin:0;padding-left:1.2rem;color:#475569;font-size:0.84rem;">
            <li>시즌 초 (ratio ≈ 0): dyn_ ≈ 3년 평균 → 소규모 표본의 불확실성을 과거 전력으로 보정</li>
            <li>시즌 후반 (ratio ≈ 1): dyn_ ≈ 0 → 충분히 쌓인 현재 성적만으로 판단</li>
            <li>중반부에는 두 신호가 games_played_ratio 비율로 자연스럽게 혼합되어 안정적인 전환 제공</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)
