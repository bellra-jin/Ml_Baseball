"""
2026 시즌 raw CSV만 갱신하는 전용 크롤링 runner.

기존 전체 시즌 크롤러(`kbo_crawler.py`, `crawl_team_rank_daily.py`)의 함수를 재사용하되,
대상 연도는 2026으로 고정한다. 실행하면 `data/raw/2026/` 하위 CSV만 새로 저장하고,
2016~2025 raw 파일은 건드리지 않는다.

실행:
    uv run python -m src.web_crawling.crawl_kbo.update_2026_raw
"""

from pathlib import Path
import sys
import time

import pandas as pd


# ─────────────────────────────────────────────
# 로컬 크롤러 모듈 import 경로
# ─────────────────────────────────────────────
CRAWL_DIR = Path(__file__).resolve().parent
if str(CRAWL_DIR) not in sys.path:
    sys.path.insert(0, str(CRAWL_DIR))

import crawl_team_rank_daily as daily_rank  # noqa: E402
import kbo_crawler as kbo  # noqa: E402


# ─────────────────────────────────────────────
# 갱신 대상 설정
# ─────────────────────────────────────────────
TARGET_YEAR = 2026
OUTPUT_DIR = Path(kbo.OUTPUT_DIR)
YEAR_DIR = OUTPUT_DIR / str(TARGET_YEAR)


# ─────────────────────────────────────────────
# 일반 기록 CSV 저장
# ─────────────────────────────────────────────
def crawl_and_save_record(name: str, path: str, year: int = TARGET_YEAR) -> tuple[str, int] | None:
    """
    `kbo_crawler.py`의 TARGETS 항목 하나를 크롤링해서 `data/raw/{year}/{name}.csv`에 저장한다.

    기존 파일이 있어도 건너뛰지 않고 새 데이터로 덮어쓴다.
    단, 크롤링 중 오류가 나면 저장 단계까지 가지 않으므로 기존 파일은 그대로 남는다.
    """
    h1, r1 = kbo.crawl_record(name, path, year)
    if not r1:
        print("    → 데이터 없음 (저장 건너뜀)")
        return None

    # Basic1 + Basic2 구조인 타자 기본 기록은 기존 전체 크롤러와 동일하게 병합한다.
    if name in kbo.EXTRA_PAGE:
        print("    Basic2 크롤링 중...")
        h2, r2 = kbo.crawl_record(name + "_extra", kbo.EXTRA_PAGE[name], year)
        if r2:
            merge_keys = kbo.get_merge_keys(name, h1, h2)
            headers, rows = kbo.merge_extra_page(h1, r1, h2, r2, merge_keys=merge_keys)
        else:
            headers, rows = h1, r1
    else:
        headers, rows = h1, r1

    rows = kbo.drop_missing_rows(name, headers, rows)
    if not rows:
        print("    → 결측 제거 후 데이터 없음 (저장 건너뜀)")
        return None

    filepath, count = kbo.save_csv(name, year, headers, rows)
    print(f"    → 저장: {filepath} (총 {count}행)")
    return filepath, count


# ─────────────────────────────────────────────
# 2026 일반 기록 전체 갱신
# ─────────────────────────────────────────────
def update_record_csvs(year: int = TARGET_YEAR) -> tuple[int, list[str]]:
    """
    선수/팀 기록과 team_final_rank를 2026년만 갱신한다.

    `kbo_crawler.TARGETS`를 그대로 순회하므로 전체 크롤러와 같은 파일 구성을 만든다.
    """
    saved = 0
    errors = []

    for name, path in kbo.TARGETS.items():
        print(f"\n  [{year}년 {name}] 갱신 중...")
        try:
            result = crawl_and_save_record(name, path, year)
            if result:
                saved += 1
        except Exception as e:
            msg = f"{year}년 {name}: {e}"
            print(f"    → 오류: {e}")
            errors.append(msg)
        time.sleep(1)

    # 선수 basic/detail 파일의 선수 집합을 기존 전체 크롤러와 동일하게 맞춘다.
    kbo.align_player_record_pairs(year)
    return saved, errors


# ─────────────────────────────────────────────
# 2026 일자별 팀 순위 갱신
# ─────────────────────────────────────────────
def update_daily_rank_csv(year: int = TARGET_YEAR) -> tuple[int, list[str]]:
    """team_daily_rank.csv만 2026년 기준으로 새로 크롤링해 저장한다."""
    target_name = daily_rank.TARGET_NAME
    filepath = YEAR_DIR / f"{target_name}.csv"

    print(f"\n  [{year}년 {target_name}] 갱신 중...")
    try:
        headers, rows = daily_rank.crawl_daily_rank_year(year)
        if not rows:
            print("    → 데이터 없음 (저장 건너뜀)")
            return 0, []

        YEAR_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=headers if headers and len(headers) == len(rows[0]) else None)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"    → 저장: {filepath} (총 {len(df)}행)")
        return 1, []
    except Exception as e:
        msg = f"{year}년 {target_name}: {e}"
        print(f"    → 오류: {e}")
        return 0, [msg]


# ─────────────────────────────────────────────
# 2026 raw 전체 갱신 실행
# ─────────────────────────────────────────────
def main() -> None:
    YEAR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"  {TARGET_YEAR}년 raw CSV 갱신 시작")
    print(f"  대상 폴더: {YEAR_DIR}")
    print(f"{'=' * 50}")

    record_saved, record_errors = update_record_csvs(TARGET_YEAR)
    daily_saved, daily_errors = update_daily_rank_csv(TARGET_YEAR)
    errors = record_errors + daily_errors
    total_saved = record_saved + daily_saved

    print(f"\n{'=' * 50}")
    print(f"2026 raw 갱신 완료: {total_saved}개 파일 저장")
    if errors:
        print(f"오류 {len(errors)}건:")
        for err in errors:
            print(f"  - {err}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
