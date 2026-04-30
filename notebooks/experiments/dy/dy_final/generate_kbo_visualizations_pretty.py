# -*- coding: utf-8 -*-
"""Presentation-style KBO visualization generator.

The previous chart script depended on matplotlib/seaborn and produced analysis
check charts. This version uses pandas + Pillow so the scheduled update can
refresh polished PNGs without an extra plotting stack.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / "data"
OUT_DIR = ROOT_DIR / "kbo_outputs"

W, H = 1400, 850
BG = "#F5F7FB"
CARD = "#FFFFFF"
INK = "#171A20"
MUTED = "#6B7280"
GRID = "#E5E7EB"
GREEN = "#159A74"
BLUE = "#2563EB"
PURPLE = "#5B4BC4"
ORANGE = "#E26A3D"
YELLOW = "#EAB308"
RED = "#DC4C3E"
NAVY = "#18243B"


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV read failed: {path}")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\malgunbd.ttf" if bold else r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(20)
F_H2 = font(25, True)
F_LABEL = font(18)
F_SMALL = font(15)
F_VALUE = font(23, True)
F_BADGE = font(16, True)


def text_wh(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), str(text), font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill=INK) -> None:
    w, h = text_wh(draw, text, fnt)
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2 - 2), text, font=fnt, fill=fill)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((46, 38, W - 46, H - 40), radius=28, fill=CARD)
    draw.text((86, 68), title, fill=INK, font=F_TITLE)
    draw.text((88, 122), subtitle, fill=MUTED, font=F_SUB)
    draw.line((86, 168, W - 86, 168), fill="#EEF0F5", width=2)
    return img, draw


def save(img: Image.Image, name: str) -> None:
    path = ROOT_DIR / name
    img.save(path, quality=95)
    print(f"[saved] {path}")


def nice_range(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if math.isclose(lo, hi):
        return lo - 1, hi + 1
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def fmt_value(v: float, kind: str = "num") -> str:
    if pd.isna(v):
        return "-"
    if kind == "avg":
        return f"{float(v):.3f}".replace("0.", ".")
    if kind == "pct":
        return f"{float(v):.1f}%"
    if kind == "int":
        return f"{int(round(float(v))):,}"
    return f"{float(v):.2f}"


def draw_grid(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], y_ticks: int = 4) -> None:
    x0, y0, x1, y1 = rect
    for i in range(y_ticks + 1):
        y = y1 - (y1 - y0) * i / y_ticks
        draw.line((x0, y, x1, y), fill=GRID, width=1)


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    labels: list[str],
    values: list[float],
    color: str,
    value_kind: str = "num",
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="#FAFBFD", outline="#EAECF0")
    draw.text((x0 + 24, y0 + 20), title, font=F_H2, fill=INK)
    plot = (x0 + 54, y0 + 88, x1 - 34, y1 - 70)
    px0, py0, px1, py1 = plot
    draw_grid(draw, plot)
    lo, hi = nice_range(values)
    step = (px1 - px0) / max(len(values) - 1, 1)

    pts = []
    for i, v in enumerate(values):
        x = px0 + step * i
        y = py1 - (float(v) - lo) / (hi - lo) * (py1 - py0)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=color, width=5, joint="curve")
    for (x, y), label, value in zip(pts, labels, values):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=CARD, width=3)
        draw_center(draw, (int(x - 42), int(y - 44), int(x + 42), int(y - 18)), fmt_value(value, value_kind), F_SMALL, color)
        draw_center(draw, (int(x - 36), y1 - 46, int(x + 36), y1 - 18), label, F_SMALL, MUTED)


def rank_color(rank: float) -> str:
    if pd.isna(rank):
        return "#EEF0F5"
    r = int(rank)
    if r <= 2:
        return "#DDF3EA"
    if r <= 5:
        return "#E7F0FF"
    if r <= 7:
        return "#FFF2CE"
    return "#FDE3DD"


def load_team_master() -> pd.DataFrame:
    return read_csv(ROOT_DIR / "team_master_2022_2026.csv")


def plot_01_league_trend() -> None:
    bat = read_csv(ROOT_DIR / "리그_타격환경.csv").sort_values("연도")
    pit = read_csv(ROOT_DIR / "리그_투구환경.csv").sort_values("연도")
    img, draw = canvas("KBO 리그 환경 5년 트렌드", "2022~2026 정규시즌 팀 기록 기반 · 2026은 진행 중 시즌")
    labels = [str(y) for y in bat["연도"]]
    draw_line_chart(draw, (86, 210, 480, 720), "리그 평균 타율", labels, bat["리그_AVG"].tolist(), GREEN, "avg")
    draw_line_chart(draw, (504, 210, 898, 720), "리그 평균 ERA", labels, pit["리그_ERA"].tolist(), ORANGE, "num")
    draw_line_chart(draw, (922, 210, 1316, 720), "팀 평균 홈런", labels, bat["리그_HR평균"].tolist(), PURPLE, "num")
    draw.text((92, 760), "포인트: 2024 타고투저 흐름 이후 2026은 아직 표본이 작아 해석 시 진행 중 시즌 보정이 필요합니다.", fill=MUTED, font=F_SMALL)
    save(img, "01_league_trend.png")


def plot_02_rank_heatmap(team: pd.DataFrame) -> None:
    img, draw = canvas("팀별 연도별 순위 변화", "순위가 낮을수록 상위권 · 2026은 현재 순위 기준")
    pivot = team.pivot_table(index="팀명", columns="연도", values="순위", aggfunc="first")
    years = [2022, 2023, 2024, 2025, 2026]
    pivot = pivot[[y for y in years if y in pivot.columns]].sort_values(2026 if 2026 in pivot.columns else pivot.columns[-1])
    x0, y0 = 260, 228
    cell_w, cell_h = 158, 52
    draw.text((92, y0 - 38), "팀", font=F_H2, fill=INK)
    for j, year in enumerate(pivot.columns):
        draw_center(draw, (x0 + j * cell_w, y0 - 46, x0 + (j + 1) * cell_w - 10, y0 - 12), str(year), F_H2, INK)
    for i, (team_name, row) in enumerate(pivot.iterrows()):
        y = y0 + i * cell_h
        draw.text((92, y + 12), str(team_name), font=F_VALUE, fill=INK)
        for j, year in enumerate(pivot.columns):
            x = x0 + j * cell_w
            rank = row[year]
            draw.rounded_rectangle((x, y + 4, x + cell_w - 12, y + cell_h - 6), radius=13, fill=rank_color(rank))
            draw_center(draw, (x, y + 4, x + cell_w - 12, y + cell_h - 6), f"{int(rank)}위", F_VALUE, INK)
    legend_y = 772
    for x, text, color in [(92, "1~2위", "#DDF3EA"), (202, "3~5위", "#E7F0FF"), (312, "6~7위", "#FFF2CE"), (422, "8~10위", "#FDE3DD")]:
        draw.rounded_rectangle((x, legend_y, x + 28, legend_y + 20), radius=6, fill=color)
        draw.text((x + 36, legend_y - 1), text, font=F_SMALL, fill=MUTED)
    save(img, "02_rank_heatmap.png")


def scale(value: float, lo: float, hi: float, a: float, b: float, reverse: bool = False) -> float:
    if math.isclose(lo, hi):
        return (a + b) / 2
    t = (float(value) - lo) / (hi - lo)
    if reverse:
        t = 1 - t
    return a + t * (b - a)


def plot_03_era_winrate(team: pd.DataFrame) -> None:
    img, draw = canvas("팀 ERA와 승률 관계", "2022~2026 팀 단위 산점도 · 2026 팀은 진하게 표시")
    df = team.dropna(subset=["pit_ERA", "승률"]).copy()
    rect = (150, 230, 1220, 705)
    x0, y0, x1, y1 = rect
    draw_grid(draw, rect)
    era_lo, era_hi = nice_range(df["pit_ERA"].tolist())
    win_lo, win_hi = nice_range(df["승률"].tolist())
    colors = {2022: "#CBD5E1", 2023: "#AEB8C8", 2024: "#8995A8", 2025: "#64748B", 2026: GREEN}
    for _, r in df.iterrows():
        x = scale(r["pit_ERA"], era_lo, era_hi, x0, x1)
        y = scale(r["승률"], win_lo, win_hi, y0, y1, reverse=True)
        year = int(r["연도"])
        radius = 9 if year < 2026 else 14
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors.get(year, BLUE), outline=CARD, width=2)
        if year == 2026:
            draw.text((x + 12, y - 12), str(r["팀명"]), font=F_SMALL, fill=INK)
    draw.text((x0, y1 + 32), "팀 ERA 낮음", font=F_LABEL, fill=MUTED)
    draw.text((x1 - 95, y1 + 32), "팀 ERA 높음", font=F_LABEL, fill=MUTED)
    draw.text((70, y0 - 6), "승률 높음", font=F_LABEL, fill=MUTED)
    draw.text((70, y1 - 18), "승률 낮음", font=F_LABEL, fill=MUTED)
    draw.text((150, 760), "해석: 좌상단에 가까울수록 실점 억제와 승률이 동시에 좋은 팀입니다.", fill=MUTED, font=F_SMALL)
    save(img, "03_era_winrate.png")


def plot_04_playoff_profile(team: pd.DataFrame) -> None:
    img, draw = canvas("가을야구 진출/미진출 팀 지표 비교", "2022~2025 완료 시즌 기준 평균 비교")
    df = team[team["연도"] < 2026].copy()
    df["진출"] = np.where(df["순위"] <= 5, "진출", "미진출")
    metrics = [("pit_ERA", "평균 ERA", "낮을수록 좋음"), ("bat_OPS", "평균 OPS", "높을수록 좋음"), ("run_differential", "평균 득실차", "높을수록 좋음")]
    for idx, (col, title, note) in enumerate(metrics):
        x = 104 + idx * 420
        y = 238
        box = (x, y, x + 365, 690)
        draw.rounded_rectangle(box, radius=20, fill="#FAFBFD", outline="#EAECF0")
        draw.text((x + 28, y + 24), title, font=F_H2, fill=INK)
        draw.text((x + 28, y + 58), note, font=F_SMALL, fill=MUTED)
        vals = df.groupby("진출")[col].mean()
        entry = [("진출", float(vals.get("진출", 0)), GREEN), ("미진출", float(vals.get("미진출", 0)), ORANGE)]
        vmin, vmax = nice_range([v for _, v, _ in entry])
        if col == "run_differential":
            vmin = min(vmin, 0)
        for k, (label, value, color) in enumerate(entry):
            by = y + 150 + k * 125
            draw.text((x + 28, by), label, font=F_VALUE, fill=INK)
            bar_x0, bar_x1 = x + 28, x + 318
            if col == "pit_ERA":
                width = scale(vmax - value, 0, vmax - vmin, 0, bar_x1 - bar_x0)
            else:
                width = scale(value, vmin, vmax, 0, bar_x1 - bar_x0)
            draw.rounded_rectangle((bar_x0, by + 42, bar_x1, by + 72), radius=12, fill="#EDF0F5")
            draw.rounded_rectangle((bar_x0, by + 42, bar_x0 + width, by + 72), radius=12, fill=color)
            kind = "num" if col != "run_differential" else "int"
            draw.text((x + 28, by + 82), fmt_value(value, kind), font=F_VALUE, fill=color)
    save(img, "04_playoff_boxplot.png")


def plot_05_transfer() -> None:
    img, draw = canvas("2026 선수 이동 현황", "KBO 공식 선수 이동 현황 API 기준 · 보조 전력 변화 데이터")
    path = DATA_DIR / "2026" / "2026_선수_이동_현황.csv"
    if not path.exists():
        draw_center(draw, (0, 300, W, 520), "선수 이동 현황 CSV가 없습니다.", F_H2, MUTED)
        save(img, "05_transfer.png")
        return
    df = read_csv(path)
    counts = Counter(df["항목"].dropna().astype(str)) if "항목" in df.columns else Counter()
    items = counts.most_common(10)
    if not items:
        draw_center(draw, (0, 300, W, 520), "선수 이동 현황 데이터가 비어 있습니다.", F_H2, MUTED)
        save(img, "05_transfer.png")
        return
    x0, y0, x1, y1 = 310, 225, 1260, 720
    max_val = max(v for _, v in items)
    row_h = 46
    for i, (name, value) in enumerate(reversed(items)):
        y = y0 + i * row_h
        color = GREEN if name in {"FA 계약", "트레이드", "FA 보상선수"} else PURPLE
        label_w, _ = text_wh(draw, name, F_LABEL)
        draw.text((x0 - label_w - 28, y + 6), name, font=F_LABEL, fill=INK)
        draw.rounded_rectangle((x0, y + 4, x1, y + 34), radius=12, fill="#EDF0F5")
        width = (x1 - x0) * value / max_val
        draw.rounded_rectangle((x0, y + 4, x0 + width, y + 34), radius=12, fill=color)
        draw.text((x0 + width + 16, y + 3), f"{value}건", font=F_VALUE, fill=INK)
    draw.text((92, 762), f"총 {len(df):,}건 · FA/트레이드/보상선수는 전력 변화 이벤트로 별도 해석 가능", font=F_SMALL, fill=MUTED)
    save(img, "05_transfer.png")


def plot_06_feature_importance() -> None:
    img, draw = canvas("가을야구 예측 피처 영향도", "로지스틱 회귀 계수 기준 · 절대값 상위 12개")
    df = read_csv(OUT_DIR / "feature_importance_coefficients.csv").head(12).copy()
    df = df.sort_values("abs_coefficient")
    x0, y0, x1 = 450, 225, 1245
    max_val = float(df["abs_coefficient"].max())
    row_h = 43
    for i, (_, r) in enumerate(df.iterrows()):
        y = y0 + i * row_h
        name = str(r["feature_ko"])
        value = float(r["abs_coefficient"])
        coef = float(r["coefficient"])
        color = GREEN if coef >= 0 else ORANGE
        label_w, _ = text_wh(draw, name, F_LABEL)
        draw.text((x0 - label_w - 28, y + 5), name, font=F_LABEL, fill=INK)
        draw.rounded_rectangle((x0, y + 6, x1, y + 34), radius=12, fill="#EDF0F5")
        width = (x1 - x0) * value / max_val
        draw.rounded_rectangle((x0, y + 6, x0 + width, y + 34), radius=12, fill=color)
        sign = "+" if coef >= 0 else "-"
        draw.text((x0 + width + 14, y + 3), f"{sign}{value:.2f}", font=F_BADGE, fill=color)
    draw.text((92, 765), "초록은 진출 확률을 높이는 방향, 주황은 낮추는 방향으로 해석합니다.", fill=MUTED, font=F_SMALL)
    save(img, "06_feature_importance.png")


def plot_07_playoff_prob() -> None:
    img, draw = canvas("2026 가을야구 진출 확률", "현재 순위 흐름 + 모델 확률 앙상블 기준")
    pred = read_csv(OUT_DIR / "2026_postseason_predictions.csv").sort_values("postseason_probability_pct", ascending=True)
    x0, y0, x1 = 315, 222, 1245
    row_h = 48
    for i, (_, r) in enumerate(pred.iterrows()):
        y = y0 + i * row_h
        team = str(r["team"])
        prob = float(r["postseason_probability_pct"])
        rank = int(float(r["rank"]))
        color = GREEN if str(r["prediction_label"]) == "진출" else ORANGE
        draw.text((95, y + 5), f"{rank}위", font=F_BADGE, fill=MUTED)
        draw.text((165, y + 4), team, font=F_VALUE, fill=INK)
        draw.rounded_rectangle((x0, y + 7, x1, y + 36), radius=12, fill="#EDF0F5")
        width = (x1 - x0) * prob / 100
        draw.rounded_rectangle((x0, y + 7, x0 + width, y + 36), radius=12, fill=color)
        draw.text((x0 + width + 12, y + 3), f"{prob:.1f}%", font=F_VALUE, fill=color)
    baseline = x0 + (x1 - x0) * 0.5
    draw.line((baseline, y0 - 22, baseline, y0 + row_h * len(pred) + 10), fill="#9CA3AF", width=2)
    draw.text((baseline + 8, y0 - 48), "50% 기준선", font=F_SMALL, fill=MUTED)
    save(img, "07_playoff_prob.png")


def plot_08_april_vs_final() -> None:
    img, draw = canvas("4월 순위의 최종 순위 예측력", "완료 시즌 기준 4월 Top5 적중률과 순위 상관")
    april = read_csv(OUT_DIR / "april_rank_insight.csv").copy()
    years = april["season"].astype(str).tolist()
    overlap = (april["current_top5_overlap_rate"] * 100).tolist()
    corr = (april["rank_final_spearman"] * 100).tolist()
    rect = (170, 235, 1220, 700)
    x0, y0, x1, y1 = rect
    draw_grid(draw, rect)
    step = (x1 - x0) / max(len(years), 1)
    for i, (year, value) in enumerate(zip(years, overlap)):
        cx = x0 + step * i + step / 2
        bar_h = (y1 - y0) * value / 100
        draw.rounded_rectangle((cx - 65, y1 - bar_h, cx - 12, y1), radius=12, fill=BLUE)
        draw_center(draw, (int(cx - 80), int(y1 - bar_h - 40), int(cx + 2), int(y1 - bar_h - 10)), f"{value:.0f}%", F_VALUE, BLUE)
        draw_center(draw, (int(cx - 78), y1 + 26, int(cx + 78), y1 + 58), year, F_LABEL, MUTED)
    pts = []
    for i, value in enumerate(corr):
        cx = x0 + step * i + step / 2 + 38
        cy = y1 - (y1 - y0) * value / 100
        pts.append((cx, cy))
    if len(pts) > 1:
        draw.line(pts, fill=ORANGE, width=5)
    for (x, y), value in zip(pts, corr):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=ORANGE, outline=CARD, width=3)
        draw_center(draw, (int(x - 38), int(y - 44), int(x + 38), int(y - 18)), f"{value:.0f}", F_SMALL, ORANGE)
    draw.rounded_rectangle((930, 205, 1270, 275), radius=14, fill="#FAFBFD", outline="#EAECF0")
    draw.rounded_rectangle((956, 226, 980, 246), radius=6, fill=BLUE)
    draw.text((990, 220), "4월 Top5 적중률", font=F_SMALL, fill=INK)
    draw.line((956, 258, 980, 258), fill=ORANGE, width=5)
    draw.text((990, 250), "순위 상관 ×100", font=F_SMALL, fill=INK)
    save(img, "08_apr_vs_final.png")


def plot_09_cluster_pca(team: pd.DataFrame) -> None:
    img, draw = canvas("팀 유형 클러스터링 PCA", "2022~2025 팀 지표를 2차원으로 축약한 유형 지도")
    features = ["team_avg", "pit_ERA", "pit_WHIP", "team_hr", "run_differential", "team_fpct"]
    df = team[team["연도"] < 2026].dropna(subset=features).copy()
    x = df[features].astype(float).to_numpy()
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:2]
    coords = x @ vecs[:, order]
    df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
    df["유형"] = np.where(df["순위"] <= 3, "상위권형", np.where(df["순위"] <= 6, "중위권형", "하위권형"))
    colors = {"상위권형": GREEN, "중위권형": PURPLE, "하위권형": ORANGE}
    rect = (170, 225, 1220, 700)
    x0, y0, x1, y1 = rect
    draw_grid(draw, rect)
    pc1_lo, pc1_hi = nice_range(df["PC1"].tolist())
    pc2_lo, pc2_hi = nice_range(df["PC2"].tolist())
    for _, r in df.iterrows():
        x_pos = scale(r["PC1"], pc1_lo, pc1_hi, x0, x1)
        y_pos = scale(r["PC2"], pc2_lo, pc2_hi, y0, y1, reverse=True)
        color = colors[str(r["유형"])]
        draw.ellipse((x_pos - 9, y_pos - 9, x_pos + 9, y_pos + 9), fill=color, outline=CARD, width=2)
        if int(r["순위"]) == 1:
            draw.text((x_pos + 11, y_pos - 10), f"{r['팀명']} {int(r['연도'])}", font=F_SMALL, fill=INK)
    legend_x = 925
    for i, (label, color) in enumerate(colors.items()):
        y = 225 + i * 34
        draw.ellipse((legend_x, y, legend_x + 18, y + 18), fill=color)
        draw.text((legend_x + 28, y - 3), label, font=F_SMALL, fill=INK)
    draw.text((170, 742), "좌표는 해석용 축약값이며, 점이 가까울수록 팀 지표 조합이 유사하다는 의미입니다.", font=F_SMALL, fill=MUTED)
    save(img, "09_cluster_pca.png")


def main() -> None:
    team = load_team_master()
    plot_01_league_trend()
    plot_02_rank_heatmap(team)
    plot_03_era_winrate(team)
    plot_04_playoff_profile(team)
    plot_05_transfer()
    plot_06_feature_importance()
    plot_07_playoff_prob()
    plot_08_april_vs_final()
    plot_09_cluster_pca(team)
    print(f"Generated presentation PNG files in {ROOT_DIR}")


if __name__ == "__main__":
    main()
