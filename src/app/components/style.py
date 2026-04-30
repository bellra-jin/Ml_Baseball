COMMON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;900&display=swap');

html, body {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stMarkdown, button, p, label, td, th {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── 전체 배경 ── */
.stApp { background: #EDF1F7; }
.main .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1440px; }

/* ── 사이드바 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0C1E4E 0%, #1B3F7A 55%, #1E4D93 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.90) !important; }
[data-testid="stSidebar"] strong { color: #fff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #fff !important; }
[data-testid="stSidebarContent"] { padding-top: 0.5rem; }

/* 사이드바 multiselect */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] { background: white !important; border-radius: 8px; }
[data-testid="stSidebar"] [data-baseweb="select"] input { color: #1E293B !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #fff !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] button svg { fill: rgba(255,255,255,0.8) !important; }

/* ── 섹션 제목 ── */
.section-title {
    font-size: 1.0rem; font-weight: 700; color: #1E293B;
    padding: 7px 14px 7px 14px;
    margin: 2rem 0 1rem;
    background: white;
    border-left: 4px solid #2563EB;
    border-radius: 0 10px 10px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    display: flex; align-items: center; gap: 8px;
}



/* ── Info Box ── */
.info-box {
    background: linear-gradient(90deg, #EFF6FF 0%, #F8FAFC 100%);
    border: 1px solid #BFDBFE;
    border-left: 4px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 11px 16px;
    font-size: 0.83rem;
    color: #1E40AF;
    margin-bottom: 1rem;
    line-height: 1.65;
}

/* ── 차트 래퍼 ── */
.stPlotlyChart {
    background: white;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    padding: 4px;
}

.st-emotion-cache-18kf3ut {
    margin-bottom: 2rem;
}

.st-emotion-cache-1s8qyds {
    font-size: 0.84rem;
}

.st-emotion-cache-1s8qyds h3 {
    font-size: 1.4rem;
}

/* ── 구분선 ── */
hr { border-color: #E2E8F0 !important; margin: 1.2rem 0 !important; }
</style>
"""
