import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re

BASE_URL = "https://www.koreabaseball.com"
OUTPUT_DIR = "data/raw"
YEARS = list(range(2016, 2027))
# YEARS = [2016]

TARGETS = {
    # 선수 기록
    "player_hitter_basic":  "/Record/Player/HitterBasic/Basic1.aspx",
    "player_hitter_detail": "/Record/Player/HitterBasic/Detail1.aspx",
    "player_pitcher_basic": "/Record/Player/PitcherBasic/Basic1.aspx",
    "player_pitcher_detail":"/Record/Player/PitcherBasic/Detail1.aspx",
    "player_defense_basic": "/Record/Player/Defense/Basic.aspx",
    "player_runner_basic":  "/Record/Player/Runner/Basic.aspx",
    # 팀 기록
    "team_hitter_basic":    "/Record/Team/Hitter/Basic1.aspx",
    "team_pitcher_basic":   "/Record/Team/Pitcher/Basic1.aspx",
    "team_defense_basic":   "/Record/Team/Defense/Basic.aspx",
    "team_runner_basic":    "/Record/Team/Runner/Basic.aspx",
    # 팀 순위
    "team_final_rank":      "/Record/TeamRank/TeamRank.aspx",
}

# Basic1 크롤 후 추가 페이지(Basic2)를 merge해야 하는 대상
# key: TARGETS의 name, value: 추가 페이지 경로
EXTRA_PAGE = {
    "player_hitter_basic": "/Record/Player/HitterBasic/Basic2.aspx",
    "team_hitter_basic": "/Record/Team/Hitter/Basic2.aspx"
}

SEASON_FIELD = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason"
YEAR_FIELD = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlYear"
TEAM_FIELD = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam"
TEAM_FILTER_TARGETS = {
    "player_hitter_basic",
    "player_hitter_basic_extra",
    "player_hitter_detail",
    "player_pitcher_basic",
    "player_pitcher_detail",
}
DROP_MISSING_ROW_TARGETS = {
    "player_hitter_basic",
    "player_hitter_detail",
    "player_pitcher_basic",
    "player_pitcher_detail",
}
MISSING_VALUES = {"", "-"}
MISSING_FILTER_COLUMN_BY_TARGET = {
    "player_hitter_basic": "AVG",
    "player_hitter_detail": "AVG",
    "player_pitcher_basic": "ERA",
    "player_pitcher_detail": "ERA",
}

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


def extract_form_fields(soup):
    """form 내 모든 input + select 필드 추출"""
    fields = {}
    form = soup.find("form", id="mainForm") or soup.find("form")
    if not form:
        return fields
    for inp in form.find_all("input"):
        name = inp.get("name", "")
        if name:
            fields[name] = inp.get("value", "")
    for sel in form.find_all("select"):
        name = sel.get("name", "")
        if name:
            selected = sel.find("option", selected=True)
            fields[name] = selected.get("value", "") if selected else ""
    return fields


def parse_postback_target(href):
    """javascript:__doPostBack('target','arg') 에서 target 추출"""
    inner = href[href.find("(") + 1 : href.rfind(")")]
    parts = [p.strip().strip("'\"") for p in inner.split(",")]
    return parts[0] if parts else None


def get_pager_info(soup):
    """페이지네이션 버튼 정보 반환: {page_num(int): target, "다음": target}"""
    info = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "__doPostBack" not in href or "ucPager" not in href:
            continue
        target = parse_postback_target(href)
        if not target:
            continue
        text = a.get_text(strip=True)
        if text.isdigit():
            info[int(text)] = target
        elif text in ("다음", ">", "▶"):
            info["다음"] = target
    return info


def parse_table(soup):
    """메인 데이터 테이블 파싱. (headers, rows) 반환"""
    table = (
        soup.find("table", class_="tData")
        or soup.select_one("div.record_wrap table")
        or soup.select_one("div#cphContents_cphContents_cphContents_udpContent table")
        or soup.find("table")
    )
    if not table:
        return [], []

    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
    else:
        first_tr = table.find("tr")
        headers = (
            [cell.get_text(strip=True) for cell in first_tr.find_all(["th", "td"])]
            if first_tr
            else []
        )

    tbody = table.find("tbody") or table
    parsed_rows = []
    has_player_id = False
    for tr in tbody.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            player_id = extract_player_id(tr)
            if player_id:
                has_player_id = True
            parsed_rows.append((player_id, cols))

    if has_player_id and "선수ID" not in headers:
        insert_at = 1 if headers and headers[0] == "순위" else 0
        headers.insert(insert_at, "선수ID")
        rows = []
        for player_id, cols in parsed_rows:
            cols.insert(insert_at, player_id)
            rows.append(cols)
    else:
        rows = [cols for _, cols in parsed_rows]

    return headers, rows


def extract_player_id(row):
    """선수 링크에서 playerId 값을 추출한다."""
    for a in row.find_all("a", href=True):
        match = re.search(r"playerId=(\d+)", a["href"])
        if match:
            return match.group(1)
    return ""


def switch_year(session, url, soup, year):
    """연도 드롭다운 변경 POST → 해당 연도 첫 페이지 soup 반환"""
    fields = extract_form_fields(soup)
    year_field = get_year_field(fields, soup, year)
    fields[year_field] = str(year)
    fields["__EVENTTARGET"] = year_field
    fields["__EVENTARGUMENT"] = ""
    resp = session.post(url, data=fields, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_year_field(fields, soup, year):
    """페이지별 연도 select 필드명을 찾는다."""
    if SEASON_FIELD in fields:
        return SEASON_FIELD
    if YEAR_FIELD in fields:
        return YEAR_FIELD

    year_text = str(year)
    for select in soup.find_all("select"):
        name = select.get("name", "")
        if not name:
            continue
        if select.find("option", value=year_text):
            return name

    raise ValueError(f"{year}년 선택 필드를 찾을 수 없습니다.")


def get_team_options(soup):
    """팀 드롭다운에서 선택 가능한 팀 목록 반환: [(value, label), ...]"""
    select = soup.find("select", attrs={"name": TEAM_FIELD}) or soup.find(
        "select", id="cphContents_cphContents_cphContents_ddlTeam_ddlTeam"
    )
    if not select:
        return []

    options = []
    for option in select.find_all("option"):
        value = option.get("value", "").strip()
        label = option.get_text(strip=True)
        if value and label and label not in ("전체", "팀 선택"):
            options.append((value, label))
    return options


def switch_team(session, url, soup, team_value):
    """팀 드롭다운 변경 POST → 해당 팀 첫 페이지 soup 반환"""
    fields = extract_form_fields(soup)
    fields[TEAM_FIELD] = team_value
    fields["__EVENTTARGET"] = TEAM_FIELD
    fields["__EVENTARGUMENT"] = ""
    resp = session.post(url, data=fields, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def row_key(headers, row):
    """선수 중복 제거 키. 선수ID+팀명이 있으면 그 조합을 우선 사용한다."""
    if "선수ID" in headers:
        player_id_idx = headers.index("선수ID")
        if player_id_idx < len(row) and row[player_id_idx]:
            team_idx = headers.index("팀명") if "팀명" in headers else None
            team = row[team_idx] if team_idx is not None and team_idx < len(row) else ""
            return row[player_id_idx], team

    if "선수명" in headers:
        player_idx = headers.index("선수명")
        team_idx = headers.index("팀명") if "팀명" in headers else None
        player = row[player_idx] if player_idx < len(row) else ""
        team = row[team_idx] if team_idx is not None and team_idx < len(row) else ""
        return player, team
    return tuple(row)


def crawl_pages(session, url, soup, year, label):
    all_rows = []
    headers = []
    current_page = 1

    while True:
        h, rows = parse_table(soup)
        if not rows:
            break

        if not headers and h:
            headers = h
        all_rows.extend(rows)
        print(f"    {year}년 {label} {current_page}페이지: {len(rows)}행 (누계 {len(all_rows)}행)")

        pager = get_pager_info(soup)
        next_target = pager.get(current_page + 1)

        if not next_target and "다음" in pager:
            fields = extract_form_fields(soup)
            fields["__EVENTTARGET"] = pager["다음"]
            fields["__EVENTARGUMENT"] = ""
            time.sleep(0.5)
            resp = session.post(url, data=fields, headers=REQUEST_HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            pager = get_pager_info(soup)
            next_target = pager.get(current_page + 1)

        if not next_target:
            break

        fields = extract_form_fields(soup)
        fields["__EVENTTARGET"] = next_target
        fields["__EVENTARGUMENT"] = ""
        time.sleep(0.5)
        resp = session.post(url, data=fields, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        current_page += 1

    return headers, all_rows


def crawl_record(name, path, year):
    url = BASE_URL + path
    session = requests.Session()

    resp = session.get(url, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    soup = switch_year(session, url, soup, year)
    time.sleep(0.5)

    headers, all_rows = crawl_pages(session, url, soup, year, "전체")

    if name not in TEAM_FILTER_TARGETS:
        return headers, all_rows

    team_options = get_team_options(soup)
    if not team_options:
        print("    팀 선택 옵션 없음")
        return headers, all_rows

    seen = {row_key(headers, row) for row in all_rows}
    added = 0
    duplicated = 0

    for team_value, team_label in team_options:
        print(f"    팀 선택 크롤링: {team_label}")
        team_soup = switch_team(session, url, soup, team_value)
        time.sleep(0.5)
        team_headers, team_rows = crawl_pages(session, url, team_soup, year, team_label)

        if not headers and team_headers:
            headers = team_headers

        for row in team_rows:
            key = row_key(headers, row)
            if key in seen:
                duplicated += 1
                continue
            seen.add(key)
            all_rows.append(row)
            added += 1

    print(f"    팀 선택 추가: 신규 {added}행, 중복 제외 {duplicated}행")
    return headers, all_rows


def merge_extra_page(h1, r1, h2, r2, merge_keys=("선수명", "팀명"), from_col="G"):
    """Basic1 데이터에 Basic2의 추가 컬럼을 merge한다.

    from_col 위치부터 시작하되, Basic1에 이미 있는 컬럼은 제외한다.
    merge_keys(선수명, 팀명)로 join한다.
    """
    df1 = pd.DataFrame(r1, columns=h1)
    df2 = pd.DataFrame(r2, columns=h2)

    start = h2.index(from_col) if from_col in h2 else len(merge_keys)
    extra_cols = [c for c in h2[start:] if c not in set(h1)]

    if not extra_cols:
        return list(df1.columns), df1.values.tolist()

    mk = list(merge_keys)
    merged = df1.merge(df2[mk + extra_cols], on=mk, how="left")
    print(f"    Basic2 추가 컬럼: {extra_cols}")
    return list(merged.columns), merged.values.tolist()


def get_merge_keys(name, h1, h2):
    """Basic1/Basic2 병합 키 선택. 선수ID가 있으면 동명이인 방지를 위해 우선 사용한다."""
    if name.startswith("team_"):
        return ("팀명",)
    if all(col in h1 and col in h2 for col in ("선수ID", "팀명")):
        return ("선수ID", "팀명")
    if "선수ID" in h1 and "선수ID" in h2:
        return ("선수ID",)
    return ("선수명", "팀명")


def drop_missing_rows(name, headers, rows):
    """학습용 선수 기록에서 핵심 지표가 결측인 행만 제거한다."""
    if name not in DROP_MISSING_ROW_TARGETS or not rows:
        return rows

    filter_col = MISSING_FILTER_COLUMN_BY_TARGET.get(name)
    if not filter_col or filter_col not in headers:
        return rows

    filter_idx = headers.index(filter_col)
    cleaned = [
        row for row in rows
        if filter_idx >= len(row) or str(row[filter_idx]).strip() not in MISSING_VALUES
    ]
    removed = len(rows) - len(cleaned)
    if removed:
        print(f"    {filter_col} 결측 행 제거: {removed}행")
    return cleaned


def save_csv(name, year, headers, rows):
    year_dir = os.path.join(OUTPUT_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)
    df = pd.DataFrame(rows, columns=headers if len(headers) == len(rows[0]) else None)
    filepath = os.path.join(year_dir, f"{name}.csv")
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    return filepath, len(df)


def get_dataframe_key_columns(df):
    """선수 기록 파일끼리 선수 집합을 맞출 때 사용할 키 컬럼을 고른다."""
    if {"선수ID", "팀명"}.issubset(df.columns):
        return ["선수ID", "팀명"]
    if {"선수명", "팀명"}.issubset(df.columns):
        return ["선수명", "팀명"]
    return []


def align_player_record_pair(year, basic_name, detail_name):
    """basic/detail 선수 집합을 교집합으로 맞춘다."""
    year_dir = os.path.join(OUTPUT_DIR, str(year))
    basic_path = os.path.join(year_dir, f"{basic_name}.csv")
    detail_path = os.path.join(year_dir, f"{detail_name}.csv")

    if not os.path.exists(basic_path) or not os.path.exists(detail_path):
        return

    basic_df = pd.read_csv(basic_path, dtype=str).fillna("")
    detail_df = pd.read_csv(detail_path, dtype=str).fillna("")
    basic_keys = get_dataframe_key_columns(basic_df)
    detail_keys = get_dataframe_key_columns(detail_df)

    if not basic_keys or basic_keys != detail_keys:
        print(f"  [{basic_name}/{detail_name}] 선수 집합 정렬 건너뜀 (키 컬럼 불일치)")
        return

    basic_key_set = set(map(tuple, basic_df[basic_keys].values.tolist()))
    detail_key_set = set(map(tuple, detail_df[detail_keys].values.tolist()))
    common_keys = basic_key_set & detail_key_set

    if not common_keys:
        print(f"  [{basic_name}/{detail_name}] 선수 집합 정렬 건너뜀 (공통 선수 없음)")
        return

    basic_mask = basic_df[basic_keys].apply(tuple, axis=1).isin(common_keys)
    detail_mask = detail_df[detail_keys].apply(tuple, axis=1).isin(common_keys)
    aligned_basic = basic_df.loc[basic_mask].copy()
    aligned_detail = detail_df.loc[detail_mask].copy()

    basic_removed = len(basic_df) - len(aligned_basic)
    detail_removed = len(detail_df) - len(aligned_detail)
    if not basic_removed and not detail_removed:
        return

    aligned_basic.to_csv(basic_path, index=False, encoding="utf-8-sig")
    aligned_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(
        f"  [{basic_name}/{detail_name}] 선수 집합 정렬: "
        f"basic {basic_removed}행 제거, detail {detail_removed}행 제거, 공통 {len(common_keys)}행"
    )


def align_player_record_pairs(year):
    """학습용 basic/detail 파일의 선수 집합을 동일하게 맞춘다."""
    align_player_record_pair(year, "player_hitter_basic", "player_hitter_detail")
    align_player_record_pair(year, "player_pitcher_basic", "player_pitcher_detail")


def main():
    total_files = 0
    errors = []

    for year in YEARS:
        print(f"\n{'='*40}")
        print(f"  {year}년 크롤링 시작")
        print(f"{'='*40}")

        for name, path in TARGETS.items():
            filepath = os.path.join(OUTPUT_DIR, str(year), f"{name}.csv")
            if os.path.exists(filepath):
                print(f"\n  [{name}] → 이미 존재, 건너뜀")
                continue

            print(f"\n  [{name}]")
            try:
                h1, r1 = crawl_record(name, path, year)

                if not r1:
                    print(f"    → 데이터 없음 (건너뜀)")
                    continue

                # Basic2 등 추가 페이지가 있으면 merge
                if name in EXTRA_PAGE:
                    print(f"    Basic2 크롤링 중...")
                    h2, r2 = crawl_record(name + "_extra", EXTRA_PAGE[name], year)
                    if r2:
                        merge_keys = get_merge_keys(name, h1, h2)
                        headers, rows = merge_extra_page(h1, r1, h2, r2, merge_keys=merge_keys)
                    else:
                        headers, rows = h1, r1
                else:
                    headers, rows = h1, r1

                rows = drop_missing_rows(name, headers, rows)
                if not rows:
                    print(f"    → 결측 제거 후 데이터 없음 (건너뜀)")
                    continue

                filepath, count = save_csv(name, year, headers, rows)
                print(f"    → 저장: {filepath} (총 {count}행)")
                total_files += 1

            except Exception as e:
                msg = f"{year}년 {name}: {e}"
                print(f"    → 오류: {e}")
                errors.append(msg)

            time.sleep(1)

        align_player_record_pairs(year)

    print(f"\n{'='*40}")
    print(f"전체 완료: {total_files}개 파일 저장")
    if errors:
        print(f"오류 {len(errors)}건:")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
