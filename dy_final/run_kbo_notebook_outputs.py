# -*- coding: utf-8 -*-
"""Refresh notebook-derived KBO prediction CSVs from the latest 2026 crawl.

This is intentionally small and dependency-light: the project venv does not have
nbconvert/nbclient, so we execute the notebook's code cells in order with a
plain Python namespace, then patch the 2026 team rows from the freshly crawled
raw CSV files before rebuilding the model outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def quiet_display(obj=None, *_, **__) -> None:
    if obj is None:
        return
    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.DataFrame):
            print(obj.head(10).to_string(index=False))
            return
        if isinstance(obj, pd.Series):
            print(obj.head(10).to_string())
            return
    except Exception:
        pass
    print(repr(obj))


def read_notebook_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def exec_cell(label: str, source: str, namespace: dict) -> None:
    print(f"\n--- execute {label} ---")
    code = compile(source, label, "exec")
    exec(code, namespace)


def read_raw_csv(path: Path, namespace: dict):
    read_csv_safe = namespace.get("read_csv_safe")
    if read_csv_safe is None:
        raise RuntimeError("read_csv_safe is not defined")
    return read_csv_safe(str(path))


def replace_year_rows(existing, new_rows, year: int):
    import pandas as pd  # type: ignore

    if existing is None or len(existing) == 0:
        return new_rows.copy()
    if "연도" not in existing.columns:
        return new_rows.copy()
    keep = existing[existing["연도"].astype("Int64") != int(year)].copy()
    combined = pd.concat([keep, new_rows], ignore_index=True, sort=False)
    return combined


def merge_prefixed(base, detail, on_cols: list[str], prefix: str):
    if detail is None or detail.empty:
        return base
    drop_cols = [c for c in detail.columns if c not in on_cols and f"{prefix}{c}" in base.columns]
    detail = detail.drop(columns=drop_cols, errors="ignore").copy()
    rename = {c: f"{prefix}{c}" for c in detail.columns if c not in on_cols}
    return base.merge(detail.rename(columns=rename), on=on_cols, how="left")


def build_team_composite(data: dict, year: int, namespace: dict):
    import numpy as np  # type: ignore

    rank = data["팀_순위"].copy()
    rank = rank[rank["연도"].astype("Int64") == int(year)].copy()
    if rank.empty:
        raise RuntimeError(f"{year} 팀_순위 rows are empty")

    out = rank.copy()
    on_cols = ["연도", "팀명"]
    sources = [
        ("팀_타자_기본기록", "bat_"),
        ("팀_타자_세부기록", "bat_"),
        ("팀_투수_기본기록", "pit_"),
        ("팀_투수_세부기록", "pit_"),
        ("팀_수비_기본기록", "def_"),
        ("팀_수비_세부기록", "def_"),
        ("팀_주루_기본기록", "run_"),
        ("팀_주루_세부기록", "run_"),
    ]
    for key, prefix in sources:
        src = data.get(key)
        if src is None or src.empty or "연도" not in src.columns:
            continue
        src_y = src[src["연도"].astype("Int64") == int(year)].copy()
        if src_y.empty:
            continue
        out = merge_prefixed(out, src_y, on_cols, prefix)

    # Normalize aliases expected by the notebook model cell.
    alias_pairs = {
        "pit_K/9": "pit_K9",
        "pit_BB/9": "pit_BB9",
        "pit_HR/9": "pit_HR9",
        "run_SB%_calc": "run_SB%",
        "bat_ISOP": "bat_ISO",
    }
    for src, dst in alias_pairs.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    derived_aliases = {
        "bat_득점/G": "득점/G_타자",
        "bat_HR/G": "HR/G_타자",
    }
    for src, dst in derived_aliases.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    out = out.drop(columns=list(derived_aliases.keys()), errors="ignore")

    if {"bat_R", "pit_R"}.issubset(out.columns):
        out["득실차"] = out["bat_R"] - out["pit_R"]
    if {"bat_R", "pit_R"}.issubset(out.columns):
        pythagorean_wp = namespace.get("pythagorean_wp")
        if pythagorean_wp is not None:
            out["기대승률"] = pythagorean_wp(out["bat_R"], out["pit_R"])
    if {"승률", "기대승률"}.issubset(out.columns):
        out["승률_운"] = (out["승률"] - out["기대승률"]).round(3)
    if "가을야구" in out.columns:
        out["가을야구"] = np.nan

    return out


def sync_latest_raw_2026(raw_year_dir: Path, namespace: dict, year: int = 2026) -> None:
    if not raw_year_dir.exists():
        print(f"[sync] raw dir missing, using clean files only: {raw_year_dir}")
        return

    data = namespace.get("data")
    file_proc_map = namespace.get("FILE_PROC_MAP")
    if not isinstance(data, dict) or not isinstance(file_proc_map, dict):
        raise RuntimeError("Notebook data/FILE_PROC_MAP is not ready")

    processors = dict(file_proc_map)
    f_tg = namespace.get("f_tg")
    if f_tg is not None:
        processors.update(
            {
                "팀_타자_세부기록": lambda df, y: f_tg(df, y, "팀_타자"),
                "팀_투수_세부기록": lambda df, y: f_tg(df, y, "팀_투수"),
                "팀_수비_세부기록": lambda df, y: f_tg(df, y, "팀_수비"),
                "팀_주루_세부기록": lambda df, y: f_tg(df, y, "팀_주루"),
            }
        )

    processed_keys: list[str] = []
    for key, processor in processors.items():
        raw_path = raw_year_dir / f"{key}.csv"
        if not raw_path.exists():
            continue
        raw = read_raw_csv(raw_path, namespace)
        processed = processor(raw, year)
        data[key] = replace_year_rows(data.get(key), processed, year)
        processed_keys.append(key)

    team_composite = build_team_composite(data, year, namespace)
    data["팀_종합"] = replace_year_rows(data.get("팀_종합"), team_composite, year)
    # Force the notebook helper to recompute these consistently for all years.
    data["팀_종합"] = data["팀_종합"].drop(columns=["득점/G_타자", "HR/G_타자"], errors="ignore")

    print(f"[sync] updated {len(processed_keys)} raw tables from {raw_year_dir}")
    print("[sync] rebuilt 2026 팀_종합 rows:", team_composite.shape)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Run notebook-derived KBO prediction outputs.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\dy_final\KBO_가을야구_예측_전처리_v2.ipynb"),
    )
    parser.add_argument(
        "--raw-year-dir",
        type=Path,
        default=Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\data\2026"),
    )
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = args.notebook.resolve()
    notebook_dir = notebook.parent
    os.chdir(notebook_dir)

    cells = read_notebook_cells(notebook)
    namespace: dict = {
        "__name__": "__main__",
        "__file__": str(notebook),
        "display": quiet_display,
    }

    try:
        # Notebook code cells, zero-based among code cells:
        # 0 setup, 1 utilities, 2 preprocessors, 3 clean data load.
        for idx in [0, 1, 2, 3]:
            exec_cell(f"notebook-code-cell-{idx + 1}", cells[idx], namespace)

        sync_latest_raw_2026(args.raw_year_dir.resolve(), namespace, args.year)

        # 4 master tables, 6 charts, 7 model, 8 auxiliary insight, 9 final save.
        # Code cell 5 only defines optional crawler helpers and is skipped here.
        for idx in [4, 6, 7, 8, 9]:
            exec_cell(f"notebook-code-cell-{idx + 1}", cells[idx], namespace)

        print("\n[done] notebook outputs refreshed")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
