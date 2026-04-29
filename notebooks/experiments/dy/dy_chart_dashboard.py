# -*- coding: utf-8 -*-
"""Keep the richer Chart.js dashboard design while refreshing embedded data."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "kbo_outputs"
RAW_2026_DIR = BASE_DIR.parents[2] / "data" / "raw" / "2026"
PREFERRED_HTML = BASE_DIR / "kbo_2022_2026_master_dashboard (1).html"
CANONICAL_HTML = BASE_DIR / "kbo_2022_2026_master_dashboard.html"


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        text = str(value).replace(",", "").replace("%", "").strip()
        if text in {"", "-"}:
            return default
        return float(text)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    number = to_float(value)
    return default if number is None else int(round(number))


def round_or_none(value: Any, digits: int = 3) -> float | None:
    number = to_float(value)
    return None if number is None else round(number, digits)


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_for_json(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if pd.isna(value):
        return None
    return value


def load_template_data(template: str) -> dict[str, Any]:
    match = re.search(r"var D = (.*?);\s*\nvar TC", template, flags=re.S)
    if not match:
        return {}
    payload = re.sub(r"\bNaN\b", "null", match.group(1))
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def latest_date(pred: pd.DataFrame) -> str:
    if "date" not in pred.columns:
        return "2026"
    dates = pd.to_datetime(pred["date"], errors="coerce").dropna()
    return "2026" if dates.empty else dates.max().date().isoformat()


def build_pred_2026() -> tuple[list[dict[str, Any]], str]:
    pred = read_csv(OUT_DIR / "2026_postseason_predictions.csv")
    master = read_csv(BASE_DIR / "team_master_2022_2026.csv")
    stats = master[master["연도"].astype(str) == "2026"].copy()
    merged = pred.merge(stats, left_on="team", right_on="팀명", how="left", suffixes=("", "_stat"))

    rows = []
    for _, row in merged.sort_values("postseason_probability_pct", ascending=False).iterrows():
        rows.append(
            {
                "팀명": row["team"],
                "순위": to_int(row["rank"]),
                "경기": to_int(row["games"]),
                "승": to_int(row["wins"]),
                "패": to_int(row["losses"]),
                "승률": round_or_none(row["win_rate"], 3),
                "bat_OPS": round_or_none(row.get("bat_OPS"), 3),
                "pit_ERA": round_or_none(row.get("pit_ERA"), 2),
                "pit_WHIP": round_or_none(row.get("pit_WHIP"), 2),
                "득실차": round_or_none(row.get("run_differential"), 1),
                "기대승률": round_or_none(row.get("pythagorean_win_rate"), 4),
                "PO_확률": round_or_none(row["postseason_probability_pct"], 1),
            }
        )
    return rows, latest_date(pred)


def build_rank_pivot() -> list[dict[str, Any]]:
    master = read_csv(BASE_DIR / "team_master_2022_2026.csv")
    pivot = master.pivot_table(index="팀명", columns="연도", values="순위", aggfunc="first").reset_index()
    rows = []
    for _, row in pivot.sort_values("팀명").iterrows():
        item: dict[str, Any] = {"팀명": row["팀명"]}
        for year in [2022, 2023, 2024, 2025, 2026]:
            item[str(year)] = to_int(row.get(year), default=None) if year in pivot.columns else None
        rows.append(item)
    return rows


def update_league_rows(current: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    path = BASE_DIR / filename
    if not path.exists():
        return current
    updates = read_csv(path)
    by_year = {int(row.get("연도")): dict(row) for row in current if row.get("연도") is not None}
    for _, row in updates.iterrows():
        year = to_int(row.get("연도"))
        item = by_year.setdefault(year, {"연도": year})
        for key, value in row.items():
            if key == "연도":
                continue
            item[key] = round_or_none(value, 4)
    return [by_year[y] for y in sorted(by_year)]


def build_feature_importance() -> list[dict[str, Any]]:
    path = OUT_DIR / "feature_importance_coefficients.csv"
    if not path.exists():
        return []
    df = read_csv(path).head(12).copy()
    df["abs_coefficient"] = pd.to_numeric(df["abs_coefficient"], errors="coerce").fillna(0)
    total = df["abs_coefficient"].sum() or 1
    rows = []
    for _, row in df.iterrows():
        pct = float(row["abs_coefficient"]) / total * 100
        direction = "진출 가능성 ↑" if to_float(row.get("coefficient"), 0) >= 0 else "진출 가능성 ↓"
        rows.append(
            {
                "피처명": row.get("feature_ko") or row.get("feature"),
                "중요도(%)": round(pct, 1),
                "방향": direction,
            }
        )
    return rows


def build_auc_summary() -> dict[str, float]:
    path = OUT_DIR / "validation_leave_one_season.csv"
    if not path.exists():
        return {"시즌단위 평균": 0.0}

    validation = read_csv(path)
    auc = pd.to_numeric(validation.get("auc"), errors="coerce").dropna()
    if auc.empty:
        return {"시즌단위 평균": 0.0}
    return {"시즌단위 평균": round(float(auc.mean()), 3)}


def build_player_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hitter_path = RAW_2026_DIR / "player_hitter_basic.csv"
    pitcher_path = RAW_2026_DIR / "player_pitcher_basic.csv"
    hitter_rows: list[dict[str, Any]] = []
    pitcher_rows: list[dict[str, Any]] = []

    if hitter_path.exists():
        hitters = read_csv(hitter_path).copy()
        hitters["OPS_num"] = pd.to_numeric(hitters.get("OPS"), errors="coerce")
        for _, row in hitters.sort_values("OPS_num", ascending=False).head(10).iterrows():
            hitter_rows.append(
                {
                    "선수명": row.get("선수명"),
                    "팀": row.get("팀명"),
                    "AVG": round_or_none(row.get("AVG"), 3),
                    "OPS": round_or_none(row.get("OPS"), 3),
                    "OPS+": None,
                    "HR": to_int(row.get("HR")),
                    "RBI": to_int(row.get("RBI")),
                    "BABIP": None,
                }
            )

    if pitcher_path.exists():
        pitchers = read_csv(pitcher_path).copy()
        pitchers["ERA_num"] = pd.to_numeric(pitchers.get("ERA"), errors="coerce")
        for _, row in pitchers.sort_values("ERA_num").head(10).iterrows():
            pitcher_rows.append(
                {
                    "선수명": row.get("선수명"),
                    "팀": row.get("팀명"),
                    "ERA": round_or_none(row.get("ERA"), 2),
                    "WHIP": round_or_none(row.get("WHIP"), 2),
                    "K/9": None,
                    "BB/9": None,
                    "FIP": None,
                    "IP": row.get("IP"),
                }
            )

    return hitter_rows, pitcher_rows


def build_transfer_rows(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        RAW_2026_DIR / "player_trade.csv",
        BASE_DIR.parents[2] / "data" / "2026" / "2026_선수_이동_현황.csv",
    ]
    path = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)
    if path is None:
        return current

    df = read_csv(path)
    rows = []
    for _, row in df.iterrows():
        date = row.get("날짜")
        parsed = pd.to_datetime(date, errors="coerce")
        rows.append(
            {
                "날짜": parsed.date().isoformat() if not pd.isna(parsed) else date,
                "항목": row.get("항목"),
                "팀": row.get("팀"),
                "선수": row.get("선수"),
                "비고": row.get("비고"),
                "연도": 2026,
                "월": int(parsed.month) if not pd.isna(parsed) else None,
            }
        )
    return rows


def render_dashboard() -> dict[str, Any]:
    template = PREFERRED_HTML.read_text(encoding="utf-8")
    data = load_template_data(template)

    pred_rows, date_label = build_pred_2026()
    data["pred_2026"] = pred_rows
    data["auc"] = build_auc_summary()
    data["rank_pivot"] = build_rank_pivot()
    data["league_bat"] = update_league_rows(data.get("league_bat", []), "리그_타격환경.csv")
    data["league_pit"] = update_league_rows(data.get("league_pit", []), "리그_투구환경.csv")

    feature_rows = build_feature_importance()
    if feature_rows:
        data["feature_imp"] = feature_rows

    hitter_rows, pitcher_rows = build_player_rows()
    if hitter_rows:
        data["bat_top2026"] = hitter_rows
    if pitcher_rows:
        data["pit_top2026"] = pitcher_rows
    data["transfer"] = build_transfer_rows(data.get("transfer", []))

    payload = json.dumps(clean_for_json(data), ensure_ascii=False, separators=(",", ":"))
    html = re.sub(r"var D = .*?;\s*\nvar TC", f"var D = {payload};\nvar TC", template, flags=re.S)
    html = re.sub(r"기준: \d{4}-\d{2}-\d{2}", f"기준: {date_label}", html)
    html = re.sub(r"기준 \d{4}-\d{2}-\d{2}", f"기준 {date_label}", html)
    replacements = {
        "ML 앙상블 가을야구 예측": "확률 기반 가을야구 예측",
        "RF LOGO-CV": "시즌 단위 검증",
        "피처 중요도 (Random Forest)": "피처 중요도 (확률 모델)",
        "LR+RF+GBM 앙상블 &middot; LOGO-CV AUC 0.96": "확률 기반 모델 + 순위 보정 &middot; 시즌 단위 검증 AUC 0.864",
        "LR+RF+GBM 앙상블 &middot; AUC 0.960": "확률 기반 모델 &middot; 평균 AUC 0.864",
        'D.auc["RF"].toFixed(3)': '(Object.values(D.auc)[0] || 0).toFixed(3)',
        '{label:"모델 구성",val:"LR + RF + GBM 앙상블 (단순 평균)"}': '{label:"모델 구성",val:"확률 기반 모델 + 현재 순위 경쟁 보정"}',
    }
    for old, new in replacements.items():
        html = html.replace(old, new)

    CANONICAL_HTML.write_text(html, encoding="utf-8")
    PREFERRED_HTML.write_text(html, encoding="utf-8")
    return {
        "canonical_html": str(CANONICAL_HTML),
        "preferred_html": str(PREFERRED_HTML),
        "latest_date": date_label,
        "pred_rows": len(pred_rows),
    }


if __name__ == "__main__":
    print(json.dumps(render_dashboard(), ensure_ascii=False, indent=2))
