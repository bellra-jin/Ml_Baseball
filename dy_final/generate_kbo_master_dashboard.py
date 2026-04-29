# -*- coding: utf-8 -*-
"""Generate a static KBO master dashboard from dy_final/data_ clean CSV files."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\dy_final")
DATA_DIR = ROOT_DIR / "data_"
OUT_HTML = Path(__file__).resolve().parent / "kbo_2022_2026_master_dashboard.html"
YEARS = [2022, 2023, 2024, 2025, 2026]
TEAM_ORDER = ["KT", "LG", "SSG", "삼성", "KIA", "NC", "한화", "두산", "롯데", "키움"]


def read_csv(name: str) -> pd.DataFrame:
    path = ROOT_DIR / name
    if not path.exists():
        path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV read failed: {path}")


def fmt_avg(v: float | int | None) -> str:
    if pd.isna(v):
        return "-"
    text = f"{float(v):.3f}"
    return text[1:] if text.startswith("0") else text


def fmt_rate(v: float | int | None) -> str:
    return fmt_avg(v)


def fmt_num(v: float | int | None, digits: int = 2) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{digits}f}"


def fmt_int(v: float | int | None) -> str:
    if pd.isna(v):
        return "-"
    return f"{int(round(float(v))):,}"


def rank_class(rank: float | int | None) -> str:
    if pd.isna(rank):
        return "r910"
    r = int(rank)
    if r == 1:
        return "r1"
    if r == 2:
        return "r2"
    if r == 3:
        return "r3"
    if r in (4, 5):
        return "r45"
    if r in (6, 7):
        return "r67"
    if r == 8:
        return "r8"
    return "r910"


def pill_class(kind: str) -> str:
    return {
        "green": "p-g",
        "blue": "p-b",
        "red": "p-r",
        "purple": "p-p",
        "yellow": "p-y",
    }.get(kind, "p-b")


def team_sentence(ranks: list[int]) -> tuple[str, str, str, str]:
    top5 = sum(1 for r in ranks if r <= 5)
    current = ranks[-1]
    previous = ranks[-2]
    best = min(ranks)
    worst = max(ranks)
    swing = worst - best

    if top5 == 5:
        return "5년 연속 상위권", "green", "리그 최강 안정형", "green"
    if current == 1:
        return "2026 선두 도약", "purple", "2026 돌풍", "purple"
    if current <= 5 and previous > 5:
        return "하위권→5강권 반등", "green", "반등 시동", "green"
    if current <= 5 and top5 >= 3:
        return "상위권 재진입", "blue", "5강권 유지", "blue"
    if current <= 5:
        return "초반 5강권 진입", "yellow", "검증 구간", "yellow"
    if current >= 9 and worst >= 9:
        return "하위권 고착", "red", "반등 필요", "red"
    if swing >= 5:
        return "등락 폭 큰 V자 흐름", "yellow", "변동성 높음", "yellow"
    return "중위권 부침", "yellow", "추격권", "yellow"


def trend(v: pd.Series, lower_is_better: bool = False) -> tuple[str, str]:
    latest = float(v.iloc[-1])
    prev = float(v.iloc[-2]) if len(v) >= 2 else latest
    delta = latest - prev
    if abs(delta) < 1e-9:
        return "eq", "→ 큰 변화 없음"
    improved = delta < 0 if lower_is_better else delta > 0
    if improved:
        return "up", "▲ 2026 초반 개선 흐름"
    return "dn", "▼ 2026 초반 하락 신호"


def series_text(values: pd.Series, formatter) -> str:
    parts = [formatter(v) for v in values]
    return "→".join(parts[:-1]) + f"→<span class=\"dn\">{parts[-1]}</span>"


def rank_badge(value: float | int | None) -> str:
    if pd.isna(value):
        return '<span class="rd r910">-</span>'
    rank = int(value)
    return f'<span class="rd {rank_class(rank)}">{rank}</span>'


def load_team_master() -> pd.DataFrame:
    if (ROOT_DIR / "team_master_2022_2026.csv").exists():
        return read_csv("team_master_2022_2026.csv")
    return read_csv("팀_종합.csv")


def load_prediction() -> pd.DataFrame:
    df = read_csv("2026_가을야구_예측결과.csv")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or str(c) == ""]
    return df.drop(columns=unnamed, errors="ignore")


def build_summary_cards(league_bat: pd.DataFrame, league_pit: pd.DataFrame) -> str:
    league_bat = league_bat.sort_values("연도")
    league_pit = league_pit.sort_values("연도")
    avg_cls, avg_note = trend(league_bat["리그_AVG"])
    era_cls, era_note = trend(league_pit["리그_ERA"], lower_is_better=True)
    whip_cls, whip_note = trend(league_pit["리그_WHIP"], lower_is_better=True)
    k_cls, k_note = trend(league_pit["리그_K9"])

    card_data = [
        ("리그 평균 타율", league_bat["리그_AVG"], fmt_avg, avg_cls, avg_note),
        ("규정이닝 평균 ERA", league_pit["리그_ERA"], lambda v: fmt_num(v, 2), era_cls, era_note),
        ("평균 WHIP", league_pit["리그_WHIP"], lambda v: fmt_num(v, 2), whip_cls, whip_note),
        ("평균 K/9", league_pit["리그_K9"], lambda v: fmt_num(v, 2), k_cls, k_note),
        (
            "규정이닝 투수 수",
            league_pit["규정투수수"],
            fmt_int,
            "eq",
            "⚡ 2026 진행중 기준",
        ),
    ]
    cards = []
    for title, values, formatter, cls, note in card_data:
        text = "→".join(formatter(v) for v in values.iloc[:-1])
        latest = formatter(values.iloc[-1])
        if title.endswith("투수 수"):
            latest = f'<span style="color:#534AB7">{latest}명</span>'
            note_html = '<div class="cs" style="color:#534AB7">⚡ 2026 진행중 기준</div>'
        else:
            latest = f'<span class="{cls}">{latest}</span>'
            note_html = f'<div class="cs {cls}">{escape(note)}</div>'
        cards.append(
            f'''
  <div class="card">
    <div class="cl">{escape(title)}</div>
    <div class="cv">{text}→{latest}</div>
    {note_html}
  </div>'''.rstrip()
        )
    return "\n".join(cards)


def build_rank_table(team: pd.DataFrame, pred: pd.DataFrame) -> str:
    team = team.copy()
    rank_pivot = team.pivot_table(index="팀명", columns="연도", values="순위", aggfunc="first")
    current = team[team["연도"] == 2026].set_index("팀명")
    pred_idx = pred.set_index("팀명") if "팀명" in pred.columns else pd.DataFrame()
    if not pred.empty and "가을야구_확률(%)" in pred.columns:
        order = pred.sort_values("가을야구_확률(%)", ascending=False)["팀명"].tolist()
    else:
        order = current.sort_values(["순위", "승률"], ascending=[True, False]).index.tolist() or TEAM_ORDER

    rows = []
    for t in order:
        ranks = [int(rank_pivot.loc[t, y]) for y in YEARS if y in rank_pivot.columns and t in rank_pivot.index]
        if len(ranks) < 5:
            continue
        c = current.loc[t]
        p = pred_idx.loc[t] if not pred_idx.empty and t in pred_idx.index else None
        trajectory, trajectory_kind, team_type, type_kind = team_sentence(ranks)
        rank_cells = "".join(f"<td>{rank_badge(rank_pivot.loc[t, y])}</td>" for y in YEARS)
        if p is not None:
            prob = fmt_num(p.get("가을야구_확률(%)"), 1)
            label = str(p.get("예측", "-")).replace("✅ ", "").replace("❌ ", "")
            label_kind = "green" if "진출" in str(p.get("예측", "")) and "미진출" not in str(p.get("예측", "")) else "red"
            era = p.get("ERA", c.get("pit_ERA"))
            win_rate = p.get("승률", c.get("승률"))
        else:
            prob = "-"
            label = team_type
            label_kind = type_kind
            era = c.get("pit_ERA")
            win_rate = c.get("승률")
        rows.append(
            f'''
  <tr>
    <td>{escape(t)}</td>
    {rank_cells}
    <td><span class="pill {pill_class(trajectory_kind)}">{escape(trajectory)}</span></td>
    <td>{fmt_num(era, 2)}</td>
    <td>{fmt_rate(win_rate)}</td>
    <td>{prob}%</td>
    <td><span class="pill {pill_class(label_kind)}">{escape(label)}</span></td>
  </tr>'''.rstrip()
        )
    return "\n".join(rows)


def insight_cards(team: pd.DataFrame, league_pit: pd.DataFrame, pred: pd.DataFrame) -> str:
    cur = team[team["연도"] == 2026].sort_values(["순위", "승률"], ascending=[True, False]).copy()
    leader = cur.iloc[0]
    best_pitching = cur.sort_values("pit_ERA").iloc[0]
    worst_pitching = cur.sort_values("pit_ERA", ascending=False).iloc[0]
    bottom = cur.sort_values(["순위", "승률"], ascending=[False, True]).iloc[0]
    era2026 = league_pit.loc[league_pit["연도"] == 2026, "리그_ERA"].iloc[0]
    pred_sorted = pred.sort_values("가을야구_확률(%)", ascending=False).copy()
    top_pred = pred_sorted.head(4)
    bubble = pred_sorted.iloc[4] if len(pred_sorted) > 4 else pred_sorted.iloc[-1]
    upset = pred_sorted[(pred_sorted["순위"] <= 5) & (pred_sorted["예측"].astype(str).str.contains("미진출"))]
    upset_row = upset.iloc[0] if not upset.empty else bubble

    data_cards = [
        (
            "#27500A",
            "예측 진출권 Top4",
            " · ".join(f"{r['팀명']} {fmt_num(r['가을야구_확률(%)'], 1)}%" for _, r in top_pred.iterrows()),
            "노트북 CELL 8의 앙상블 ML 결과를 그대로 반영했습니다.",
        ),
        (
            "#0C447C",
            f"{leader['팀명']} 선두권 페이스",
            f"{fmt_int(leader['승'])}승 {fmt_int(leader['패'])}패 · ERA {fmt_num(leader['pit_ERA'], 2)}",
            f"현재 {fmt_int(leader['순위'])}위, 승률 {fmt_rate(leader['승률'])}. 예측 확률도 최상위권입니다.",
        ),
        (
            "#993C1D",
            f"{upset_row['팀명']} 순위 대비 예측 경고",
            f"현재 {fmt_int(upset_row['순위'])}위 · 예측 {fmt_num(upset_row['가을야구_확률(%)'], 1)}%",
            "현재 순위만으로는 좋아 보여도 팀 세부 지표를 섞은 모델에서는 위험 신호가 잡혔습니다.",
        ),
        (
            "#27500A",
            "투수 강세 시즌 조짐",
            f"리그 ERA {fmt_num(era2026, 2)}",
            "2026은 진행중 시즌이라 표본 주의가 필요하지만, 현재 리그 평균 ERA는 최근 5년 중 낮은 축입니다.",
        ),
        (
            "#501313",
            f"{worst_pitching['팀명']} ERA 경고등",
            f"팀 ERA {fmt_num(worst_pitching['pit_ERA'], 2)} · 순위 {fmt_int(worst_pitching['순위'])}위",
            "실점 억제력이 가장 큰 리스크로 보입니다. 불펜/선발 안정화 여부가 순위 반등의 핵심입니다.",
        ),
        (
            "#3C3489",
            "데이터 현황",
            "노트북 실행 결과 반영",
            "예측 CSV와 team_master_2022_2026.csv를 우선 사용해 HTML을 생성했습니다.",
        ),
    ]

    return "\n".join(
        f'''
  <div class="card">
    <div class="cl" style="color:{color};font-weight:500">{escape(title)}</div>
    <div class="cv" style="font-size:13px">{escape(value)}</div>
    <div class="cs">{escape(desc)}</div>
  </div>'''.rstrip()
        for color, title, value, desc in data_cards
    )


def dataset_table() -> str:
    items = [
        ("2026_예측결과", "2026_가을야구_예측결과.csv", "노트북 CELL 8 앙상블 ML 예측 결과"),
        ("team_master_2022_2026", "team_master_2022_2026.csv", "노트북 CELL 5 이후 저장된 팀 종합 마스터"),
        ("타자_마스터", "타자_마스터.csv", "기본+세부 JOIN · OPS+, BABIP, ISO 등 파생지표 포함"),
        ("투수_마스터", "투수_마스터.csv", "기본+세부 JOIN · ERA+, FIP-, K9, BB9 포함"),
        ("팀_순위변동성", "팀_순위변동성.csv", "평균순위·표준편차·최고/최저순위·월별 평균순위"),
        ("선수이동현황", "선수이동현황_통합.csv", "2026 선수 이동/부상/등록 말소 흐름"),
        ("리그_타격/투구환경", "리그_타격환경.csv", "연도별 리그 평균 지표 · 2026은 진행중 기준"),
    ]
    rows = []
    for category, filename, note in items:
        try:
            df = read_csv(filename)
        except FileNotFoundError:
            continue
        years = "2022~2026" if "연도" in df.columns and df["연도"].nunique() > 1 else "2026" if "연도" in df.columns else "-"
        if category == "리그_타격/투구환경":
            pitch = read_csv("리그_투구환경.csv")
            row_count = f"{len(df) + len(pitch):,}행"
            col_count = f"{df.shape[1] + pitch.shape[1]}열"
        else:
            row_count = f"{len(df):,}행"
            col_count = f"{df.shape[1]}열"
        rows.append(
            f"<tr><td>{escape(category)}</td><td>{row_count}</td><td>{col_count}</td><td>{years}</td><td>{escape(note)}</td></tr>"
        )
    return "\n".join(rows)


def html_document() -> str:
    team = load_team_master()
    pred = load_prediction()
    league_bat = read_csv("리그_타격환경.csv")
    league_pit = read_csv("리그_투구환경.csv")
    latest_date = "4/24"

    css = r'''
<style>
*{box-sizing:border-box}
.w{padding:.5rem 0;font-size:13px;color:var(--color-text-primary)}
h3{font-size:14px;font-weight:500;margin:1.6rem 0 .7rem;color:var(--color-text-primary);border-bottom:.5px solid var(--color-border-tertiary);padding-bottom:5px}
.g5{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:1rem}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:1rem}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1rem}
.card{background:var(--color-background-secondary);border-radius:9px;padding:.75rem .9rem}
.cl{font-size:10px;color:var(--color-text-secondary);margin-bottom:2px;text-transform:uppercase;letter-spacing:.3px}
.cv{font-size:15px;font-weight:500}
.cs{font-size:11px;margin-top:3px;line-height:1.35}
.up{color:#D85A30}.dn{color:#185FA5}.eq{color:var(--color-text-tertiary)}
.live{display:inline-block;font-size:9px;background:#E8593C;color:#fff;padding:1px 5px;border-radius:3px;margin-left:4px;font-weight:500;vertical-align:middle}

table{width:100%;border-collapse:collapse;font-size:11.5px;margin-bottom:.5rem}
th{font-weight:500;padding:5px 7px;text-align:center;border-bottom:.5px solid var(--color-border-secondary);color:var(--color-text-secondary);font-size:10px}
th:first-child{text-align:left}
td{padding:4px 7px;border-bottom:.5px solid var(--color-border-tertiary);text-align:center}
td:first-child{text-align:left;font-weight:500}

.rd{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:50%;font-size:10px;font-weight:500}
.r1{background:#EAF3DE;color:#27500A}
.r2{background:#E1F5EE;color:#0F6E56}
.r3{background:#E6F1FB;color:#0C447C}
.r45{background:#EEEDFE;color:#3C3489}
.r67{background:#FAEEDA;color:#633806}
.r8{background:#FAECE7;color:#712B13}
.r910{background:#FCEBEB;color:#501313}

.pill{display:inline-block;font-size:9.5px;padding:1px 7px;border-radius:10px;font-weight:500}
.p-g{background:#E1F5EE;color:#0F6E56}
.p-b{background:#E6F1FB;color:#185FA5}
.p-r{background:#FAECE7;color:#993C1D}
.p-p{background:#EEEDFE;color:#3C3489}
.p-y{background:#FAEEDA;color:#633806}

.trend-bar{display:flex;align-items:center;gap:3px;margin:1px 0}
.tb{width:14px;height:14px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:600;flex-shrink:0}
</style>'''.strip()

    return f'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KBO 2022~2026 Master Dashboard</title>
{css}
</head>
<body>
<div class="w">
<h3>리그 환경 5년 트렌드</h3>
<div class="g5">
{build_summary_cards(league_bat, league_pit)}
</div>

<h3>5년 팀 순위 흐름표 <span class="live">LIVE {latest_date}</span></h3>
<table>
  <tr>
    <th>팀</th><th>2022</th><th>2023</th><th>2024</th><th>2025</th><th>2026 현재</th>
    <th>5년 궤적</th><th>2026 ERA</th><th>2026 승률</th><th>PO 확률</th><th>예측</th>
  </tr>
{build_rank_table(team, pred)}
</table>

<h3>2026 초반 주목 포인트 <span class="live">LIVE</span></h3>
<div class="g3">
{insight_cards(team, league_pit, pred)}
</div>

<h3>클린 데이터셋 현황</h3>
<table>
  <tr><th>카테고리</th><th>적재 행수</th><th>컬럼</th><th>연도</th><th>비고</th></tr>
{dataset_table()}
</table>

</div>
</body>
</html>'''


def main() -> None:
    OUT_HTML.write_text(html_document(), encoding="utf-8")
    print(OUT_HTML)


if __name__ == "__main__":
    main()
