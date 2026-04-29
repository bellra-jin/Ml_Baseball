# -*- coding: utf-8 -*-
"""Sync realtime KBO crawl data into DY experiment outputs.

Flow:
1. Copy the latest crawler CSV files from data/2026 to data/raw/2026.
2. Rebuild data/processed/2026 from the refreshed raw files.
3. Regenerate the DY experiment CSV, PNG, and HTML dashboard outputs.

This script is intentionally small so it can be called from the Windows
midnight scheduled task after kbo_realtime_crawler.py finishes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_DIR = EXPERIMENT_DIR.parents[2]
DATA_DIR = REPO_DIR / "data"
LIVE_YEAR_DIR = DATA_DIR / "2026"
RAW_YEAR_DIR = DATA_DIR / "raw" / "2026"
PROCESSED_YEAR_DIR = DATA_DIR / "processed" / "2026"
CRAWLER_PATH = REPO_DIR / "dy_final" / "kbo_realtime_crawler.py"

LIVE_TO_RAW_FILES = {
    "팀_일자별순위.csv": "team_daily_rank.csv",
    "팀_순위.csv": "team_final_rank.csv",
    "팀_타자_기본기록.csv": "team_hitter_basic.csv",
    "팀_투수_기본기록.csv": "team_pitcher_basic.csv",
    "팀_수비_기본기록.csv": "team_defense_basic.csv",
    "팀_주루_기본기록.csv": "team_runner_basic.csv",
    "타자_기본기록.csv": "player_hitter_basic.csv",
    "타자_세부기록.csv": "player_hitter_detail.csv",
    "투수_기본기록.csv": "player_pitcher_basic.csv",
    "투수_세부기록.csv": "player_pitcher_detail.csv",
    "수비_기본기록.csv": "player_defense_basic.csv",
    "주루_기본기록.csv": "player_runner_basic.csv",
}


def read_csv_flexible(path: Path):
    import pandas as pd

    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def latest_live_date() -> str | None:
    daily_path = LIVE_YEAR_DIR / "팀_일자별순위.csv"
    if not daily_path.exists():
        return None

    df = read_csv_flexible(daily_path)
    if "날짜" not in df.columns or df.empty:
        return None

    import pandas as pd

    dates = pd.to_datetime(df["날짜"].astype(str), errors="coerce")
    if dates.notna().sum() == 0:
        return None
    return dates.max().date().isoformat()


def run_crawler(year: int, include_profiles: bool = False) -> None:
    if not CRAWLER_PATH.exists():
        raise FileNotFoundError(f"Crawler script not found: {CRAWLER_PATH}")

    args = [
        sys.executable,
        str(CRAWLER_PATH),
        "--year",
        str(year),
        "--data-dir",
        str(DATA_DIR),
    ]
    if include_profiles:
        args.append("--profiles")

    subprocess.run(args, check=True)


def sync_live_to_raw() -> list[dict[str, Any]]:
    if not LIVE_YEAR_DIR.exists():
        raise FileNotFoundError(f"Live crawl directory not found: {LIVE_YEAR_DIR}")

    RAW_YEAR_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []

    for source_name, target_name in LIVE_TO_RAW_FILES.items():
        source = LIVE_YEAR_DIR / source_name
        target = RAW_YEAR_DIR / target_name
        if source.exists() and source.stat().st_size > 0:
            shutil.copy2(source, target)
            copied.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "bytes": target.stat().st_size,
                }
            )
        else:
            missing.append(source_name)

    if missing:
        print("[WARN] Missing or empty live files:", ", ".join(missing))

    return copied


def rebuild_processed_2026() -> None:
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    from src.preprocessing.build_preprocessed import save_preprocessed_year

    save_preprocessed_year(2026)


def rebuild_dy_outputs() -> dict[str, Any]:
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))

    import dy_experiment_pipeline as pipe

    return pipe.run_all()


def run(include_crawl: bool = False, include_profiles: bool = False) -> dict[str, Any]:
    if include_crawl:
        run_crawler(2026, include_profiles=include_profiles)

    live_date = latest_live_date()
    copied = sync_live_to_raw()
    rebuild_processed_2026()
    summary = rebuild_dy_outputs()

    result = {
        "live_date_before_sync": live_date,
        "copied_files": len(copied),
        "raw_dir": str(RAW_YEAR_DIR),
        "processed_dir": str(PROCESSED_YEAR_DIR),
        "dashboard_html": str(EXPERIMENT_DIR / "kbo_2022_2026_master_dashboard.html"),
        "pipeline_summary": summary,
    }
    (EXPERIMENT_DIR / "realtime_update_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh DY KBO outputs from realtime crawl data.")
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Run dy_final/kbo_realtime_crawler.py before syncing data/2026 into data/raw/2026.",
    )
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="Pass --profiles to the crawler when --crawl is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(include_crawl=args.crawl, include_profiles=args.profiles)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
