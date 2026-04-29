# -*- coding: utf-8 -*-
r"""KBO official-site crawler for regular-season data and midnight updates.

Dependencies expected in the project environment:
    pip install requests beautifulsoup4 pandas

Examples:
    python kbo_realtime_crawler.py --year 2026 --data-dir C:\Users\Admin\Documents\GitHub\Ml_Baseball\data
    python kbo_realtime_crawler.py --year 2026 --watch-midnight --movement
    python kbo_realtime_crawler.py --year 2026 --profiles --profile-limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.koreabaseball.com"
DEFAULT_DATA_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\data")
KST = ZoneInfo("Asia/Seoul")
GAME_TYPE_CODE = 1

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": BASE_URL,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

SEASON_FIELD = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason"
TEAM_RANK_YEAR_FIELD = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlYear"

DAILY_PREFIX = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$"
DAILY_DATE_INPUT = DAILY_PREFIX + "txtCanlendar"
DAILY_HF_DATE = DAILY_PREFIX + "hfSearchDate"
DAILY_HF_NEXT = DAILY_PREFIX + "hfNextDate"
DAILY_SERIES = DAILY_PREFIX + "ddlSeries"


RECORD_TARGETS = {
    # Player records
    "타자_기본기록": "/Record/Player/HitterBasic/Basic1.aspx",
    "타자_세부기록": "/Record/Player/HitterBasic/Detail1.aspx",
    "투수_기본기록": "/Record/Player/PitcherBasic/Basic1.aspx",
    "투수_세부기록": "/Record/Player/PitcherBasic/Detail1.aspx",
    "수비_기본기록": "/Record/Player/Defense/Basic.aspx",
    "주루_기본기록": "/Record/Player/Runner/Basic.aspx",
    # Team records
    "팀_타자_기본기록": "/Record/Team/Hitter/Basic1.aspx",
    "팀_타자_세부기록": "/Record/Team/Hitter/Detail1.aspx",
    "팀_투수_기본기록": "/Record/Team/Pitcher/Basic1.aspx",
    "팀_투수_세부기록": "/Record/Team/Pitcher/Detail1.aspx",
    "팀_수비_기본기록": "/Record/Team/Defense/Basic.aspx",
    "팀_수비_세부기록": "/Record/Team/Defense/Detail.aspx",
    "팀_주루_기본기록": "/Record/Team/Runner/Basic.aspx",
    "팀_주루_세부기록": "/Record/Team/Runner/Detail.aspx",
}

PLAYER_LIST_PATHS = {
    "타자": "/Record/Player/HitterBasic/Basic1.aspx",
    "투수": "/Record/Player/PitcherBasic/Basic1.aspx",
}

DETAIL_DIRS = {
    "타자": "HitterDetail",
    "투수": "PitcherDetail",
}

PROFILE_TABS = [
    ("통산기록", "Total.aspx", False),
    ("일자별기록", "Daily.aspx", True),
    ("경기별기록", "Game.aspx", True),
    ("상황별기록", "Situation.aspx", True),
]


class KBOCrawler:
    def __init__(self, data_dir: Path, year: int, delay: float = 0.45):
        self.data_dir = Path(data_dir)
        self.year = int(year)
        self.delay = delay
        self.session = requests.Session()

    def get(self, path_or_url: str) -> BeautifulSoup:
        url = path_or_url if path_or_url.startswith("http") else BASE_URL + path_or_url
        resp = self.session.get(url, headers=REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    def post(self, path_or_url: str, fields: dict) -> BeautifulSoup:
        url = path_or_url if path_or_url.startswith("http") else BASE_URL + path_or_url
        resp = self.session.post(url, data=fields, headers=REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    @staticmethod
    def extract_form_fields(soup: BeautifulSoup, include_image_inputs: bool = False) -> dict:
        fields = {}
        form = soup.find("form", id="mainForm") or soup.find("form")
        if not form:
            return fields
        for inp in form.find_all("input"):
            name = inp.get("name", "")
            input_type = inp.get("type", "").lower()
            if name and (include_image_inputs or input_type != "image"):
                fields[name] = inp.get("value", "")
        for sel in form.find_all("select"):
            name = sel.get("name", "")
            if not name:
                continue
            selected = sel.find("option", selected=True)
            fields[name] = selected.get("value", "") if selected else ""
        return fields

    @staticmethod
    def parse_postback_target(href: str) -> str | None:
        m = re.search(r"__doPostBack\('([^']*)'\s*,\s*'([^']*)'\)", href)
        return m.group(1) if m else None

    @classmethod
    def get_pager_info(cls, soup: BeautifulSoup) -> dict:
        info = {}
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "__doPostBack" not in href or "ucPager" not in href:
                continue
            target = cls.parse_postback_target(href)
            if not target:
                continue
            text = a.get_text(strip=True)
            title = a.get("title", "")
            if text.isdigit():
                info[int(text)] = target
            elif text in {"다음", ">", "›", "»"} or "다음" in title:
                info["next_group"] = target
        return info

    @staticmethod
    def parse_table(soup: BeautifulSoup) -> tuple[list[str], list[list[str]]]:
        table = (
            soup.find("table", class_="tData")
            or soup.select_one("div.record_wrap table")
            or soup.select_one("div.tbl-type02 table")
            or soup.find("table")
        )
        if not table:
            return [], []
        thead = table.find("thead")
        if thead:
            headers = [c.get_text(strip=True) for c in thead.find_all(["th", "td"])]
        else:
            first_tr = table.find("tr")
            headers = [c.get_text(strip=True) for c in first_tr.find_all(["th", "td"])] if first_tr else []
        tbody = table.find("tbody") or table
        rows = []
        for tr in tbody.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cols:
                rows.append(cols)
        return headers, rows

    def switch_season(self, path: str, soup: BeautifulSoup, field_name: str, year: int) -> BeautifulSoup:
        fields = self.extract_form_fields(soup)
        fields[field_name] = str(year)
        fields["__EVENTTARGET"] = field_name
        fields["__EVENTARGUMENT"] = ""
        time.sleep(self.delay)
        return self.post(path, fields)

    def save_rows(self, name: str, headers: list[str], rows: list[list[str]]) -> Path:
        year_dir = self.data_dir / str(self.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        path = year_dir / f"{name}.csv"
        if rows:
            df = pd.DataFrame(rows, columns=headers if headers and len(headers) == len(rows[0]) else None)
        else:
            df = pd.DataFrame(columns=headers)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[saved] {path} ({len(df):,} rows)")
        return path

    def crawl_record(self, name: str, path: str) -> Path:
        soup = self.get(path)
        soup = self.switch_season(path, soup, SEASON_FIELD, self.year)
        all_rows: list[list[str]] = []
        headers: list[str] = []
        current_page = 1

        while True:
            page_headers, rows = self.parse_table(soup)
            if rows:
                if not headers:
                    headers = page_headers
                all_rows.extend(rows)
                print(f"  {name} page {current_page}: {len(rows):,} rows")

            pager = self.get_pager_info(soup)
            next_target = pager.get(current_page + 1)
            if not next_target and "next_group" in pager:
                fields = self.extract_form_fields(soup)
                fields["__EVENTTARGET"] = pager["next_group"]
                fields["__EVENTARGUMENT"] = ""
                time.sleep(self.delay)
                soup = self.post(path, fields)
                pager = self.get_pager_info(soup)
                next_target = pager.get(current_page + 1)

            if not next_target:
                break

            fields = self.extract_form_fields(soup)
            fields["__EVENTTARGET"] = next_target
            fields["__EVENTARGUMENT"] = ""
            time.sleep(self.delay)
            soup = self.post(path, fields)
            current_page += 1

        return self.save_rows(name, headers, all_rows)

    def crawl_team_rank(self) -> Path:
        path = "/Record/TeamRank/TeamRank.aspx"
        soup = self.get(path)
        soup = self.switch_season(path, soup, TEAM_RANK_YEAR_FIELD, self.year)
        headers, rows = self.parse_table(soup)
        return self.save_rows("팀_순위", headers, rows)

    def post_daily_date(self, fields: dict, date_str: str) -> BeautifulSoup:
        fields = dict(fields)
        fields["__EVENTTARGET"] = ""
        fields["__EVENTARGUMENT"] = ""
        fields[DAILY_DATE_INPUT] = date_str
        fields[DAILY_HF_DATE] = date_str
        fields[DAILY_SERIES] = "0"
        time.sleep(self.delay)
        return self.post("/Record/TeamRank/TeamRankDaily.aspx", fields)

    def crawl_daily_rank(self) -> Path:
        path = "/Record/TeamRank/TeamRankDaily.aspx"
        soup = self.get(path)
        fields = self.extract_form_fields(soup)
        soup = self.post_daily_date(fields, f"{self.year}0101")
        fields = self.extract_form_fields(soup)
        current_date = fields.get(DAILY_HF_NEXT, "").strip()

        all_rows = []
        headers = []
        seen = set()
        while current_date and current_date.startswith(str(self.year)) and current_date not in seen:
            seen.add(current_date)
            soup = self.post_daily_date(fields, current_date)
            page_headers, rows = self.parse_table(soup)
            if rows:
                if not headers:
                    headers = ["날짜"] + page_headers
                all_rows.extend([[current_date] + row for row in rows])
            fields = self.extract_form_fields(soup)
            current_date = fields.get(DAILY_HF_NEXT, "").strip()
            if len(seen) % 20 == 0:
                print(f"  daily rank through {max(seen)}: {len(all_rows):,} rows")

        return self.save_rows("팀_일자별순위", headers, all_rows)

    def crawl_player_movement(self) -> Path:
        """Crawl player movement rows from the KBO Ajax endpoint.

        The visible Trade.aspx page renders an empty table shell first and then
        fills rows through /ws/Player.asmx/GetTradeList. Reading only the HTML
        therefore saves a header-only CSV, so this method uses the same Ajax
        endpoint as the official page.
        """
        path = "/ws/Player.asmx/GetTradeList"
        headers = ["날짜", "항목", "팀", "선수", "비고"]
        all_rows: list[list[str]] = []
        page_no = 1
        list_count = 100

        while True:
            payload = {
                "seasonId": str(self.year),
                "monthId": "0",
                "bdSc": "0",
                "teamName": "",
                "searchIf": "",
                "pageNo": str(page_no),
                "listCount": str(list_count),
            }
            ajax_headers = dict(REQUEST_HEADERS)
            ajax_headers.update(
                {
                    "Referer": BASE_URL + "/Player/Trade.aspx",
                    "X-Requested-With": "XMLHttpRequest",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                }
            )
            url = BASE_URL + path
            resp = self.session.post(url, data=payload, headers=ajax_headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            page_rows: list[list[str]] = []
            for item in data.get("rows", []):
                cells = item.get("row", [])
                row = [str(cell.get("Text", "")).strip() for cell in cells[: len(headers)]]
                if len(row) == len(headers) and row[0].startswith(str(self.year)):
                    page_rows.append(row)

            all_rows.extend(page_rows)
            total_cnt = int(data.get("totalCnt") or len(all_rows))
            print(f"  movement page {page_no}: {len(page_rows):,} rows / total {total_cnt:,}")

            if page_no * list_count >= total_cnt or not page_rows:
                break
            page_no += 1
            time.sleep(self.delay)

        return self.save_rows(f"{self.year}_선수_이동_현황", headers, all_rows)

    def extract_player_ids(self, soup: BeautifulSoup, position: str) -> dict[str, str]:
        players = {}
        detail_dir = DETAIL_DIRS[position]
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if detail_dir in href and "playerId=" in href:
                m = re.search(r"playerId=(\d+)", href)
                if m:
                    name = a.get_text(strip=True)
                    if name:
                        players[m.group(1)] = name
        return players

    def crawl_player_ids(self, position: str) -> dict[str, str]:
        path = PLAYER_LIST_PATHS[position]
        soup = self.get(path)
        soup = self.switch_season(path, soup, SEASON_FIELD, self.year)
        all_players = {}
        current_page = 1

        while True:
            players = self.extract_player_ids(soup, position)
            all_players.update(players)
            pager = self.get_pager_info(soup)
            next_target = pager.get(current_page + 1)
            if not next_target and "next_group" in pager:
                fields = self.extract_form_fields(soup)
                fields["__EVENTTARGET"] = pager["next_group"]
                fields["__EVENTARGUMENT"] = ""
                soup = self.post(path, fields)
                pager = self.get_pager_info(soup)
                next_target = pager.get(current_page + 1)
            if not next_target:
                break
            fields = self.extract_form_fields(soup)
            fields["__EVENTTARGET"] = next_target
            fields["__EVENTARGUMENT"] = ""
            time.sleep(self.delay)
            soup = self.post(path, fields)
            current_page += 1
        return all_players

    @staticmethod
    def safe_filename(text: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', "_", text)

    def parse_player_profile_info(self, soup: BeautifulSoup, player_id: str, position: str) -> dict:
        info = {"선수ID": player_id, "포지션구분": position}
        player_info = soup.find("div", class_="player_info")
        if not player_info:
            return info
        team = player_info.find("h4", class_="team")
        if team:
            info["팀명"] = team.get_text(strip=True)
        basic = player_info.find("div", class_="player_basic")
        if not basic:
            return info
        for li in basic.find_all("li"):
            label = li.find("strong")
            value = li.find("span")
            if label and value:
                info[label.get_text(strip=True).rstrip(":")] = value.get_text(strip=True)
        return info

    def save_profile_df(self, position: str, player_name: str, player_id: str, tab_name: str, df: pd.DataFrame) -> Path:
        out_dir = self.data_dir / str(self.year) / "프로필" / position
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.safe_filename(player_name)}_{player_id}_{tab_name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def parse_profile_tables(self, soup: BeautifulSoup) -> list[pd.DataFrame]:
        frames = []
        record_area = soup.find("div", class_="player_records") or soup
        section = None
        for child in record_area.children:
            if not getattr(child, "name", None):
                continue
            if child.name == "h5":
                section = child.get_text(strip=True)
            table = child.find("table") if hasattr(child, "find") else None
            if not table:
                continue
            headers, rows = self.parse_table(BeautifulSoup(str(table), "html.parser"))
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=headers if headers and len(headers) == len(rows[0]) else None)
            if section:
                df.insert(0, "분류", section)
            frames.append(df)
        return frames

    def crawl_player_profiles(self, profile_limit: int | None = None) -> None:
        for position in ["타자", "투수"]:
            players = self.crawl_player_ids(position)
            items = list(players.items())
            if profile_limit:
                items = items[:profile_limit]
            print(f"[profiles] {position}: {len(items):,} players")
            for i, (player_id, player_name) in enumerate(items, 1):
                detail_dir = DETAIL_DIRS[position]
                print(f"  {i}/{len(items)} {player_name} ({player_id})")
                basic_url = f"/Record/Player/{detail_dir}/Basic.aspx?playerId={player_id}"
                soup = self.get(basic_url)
                info = self.parse_player_profile_info(soup, player_id, position)
                self.save_profile_df(position, player_name, player_id, "프로필", pd.DataFrame([info]))
                time.sleep(self.delay)

                for tab_name, tab_file, use_year in PROFILE_TABS:
                    if use_year:
                        url = (
                            f"/Record/Player/{detail_dir}/{tab_file}"
                            f"?playerId={player_id}&seasonCode={self.year}&gameTypeCode={GAME_TYPE_CODE}"
                        )
                    else:
                        url = f"/Record/Player/{detail_dir}/{tab_file}?playerId={player_id}"
                    soup = self.get(url)
                    frames = self.parse_profile_tables(soup)
                    if frames:
                        df = pd.concat(frames, ignore_index=True)
                    else:
                        df = pd.DataFrame()
                    self.save_profile_df(position, player_name, player_id, tab_name, df)
                    time.sleep(self.delay)

    def run_once(self, include_movement: bool = True, include_profiles: bool = False, profile_limit: int | None = None) -> None:
        print(f"=== KBO {self.year} regular-season update start: {datetime.now(KST):%Y-%m-%d %H:%M:%S %Z} ===")
        for name, path in RECORD_TARGETS.items():
            print(f"[record] {name}")
            self.crawl_record(name, path)
        print("[rank] yearly team rank")
        self.crawl_team_rank()
        print("[rank] daily team rank")
        self.crawl_daily_rank()
        if include_movement:
            print("[movement] player movement")
            self.crawl_player_movement()
        if include_profiles:
            print("[profiles] player profile tabs")
            self.crawl_player_profiles(profile_limit=profile_limit)
        print(f"=== KBO update finished: {datetime.now(KST):%Y-%m-%d %H:%M:%S %Z} ===")


def seconds_until_next_midnight() -> float:
    now = datetime.now(KST)
    target = datetime.combine(now.date() + timedelta(days=1), dt_time(0, 0), tzinfo=KST)
    return max((target - now).total_seconds(), 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl KBO regular-season data from the official site.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--delay", type=float, default=0.45)
    parser.add_argument("--watch-midnight", action="store_true", help="Run once at every 00:00 KST.")
    parser.add_argument("--no-movement", action="store_true", help="Skip player movement page.")
    parser.add_argument("--profiles", action="store_true", help="Also crawl all player profile tabs.")
    parser.add_argument("--profile-limit", type=int, default=None, help="Limit players per position for test runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawler = KBOCrawler(data_dir=args.data_dir, year=args.year, delay=args.delay)

    if not args.watch_midnight:
        crawler.run_once(
            include_movement=not args.no_movement,
            include_profiles=args.profiles,
            profile_limit=args.profile_limit,
        )
        return

    while True:
        wait_seconds = seconds_until_next_midnight()
        print(f"[scheduler] sleeping {wait_seconds / 3600:.2f} hours until next 00:00 KST")
        time.sleep(wait_seconds)
        try:
            crawler.run_once(
                include_movement=not args.no_movement,
                include_profiles=args.profiles,
                profile_limit=args.profile_limit,
            )
        except Exception as exc:
            print(f"[scheduler] update failed: {exc}")
            time.sleep(60)


if __name__ == "__main__":
    main()
