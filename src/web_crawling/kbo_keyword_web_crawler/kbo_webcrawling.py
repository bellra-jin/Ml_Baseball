import pandas as pd
import asyncio
import sys
import threading
from playwright.async_api import async_playwright

# =====================================================================
# Playwright 헬퍼
# =====================================================================

_bg_loop = None
_bg_thread = None


def _ensure_bg_loop():
    global _bg_loop, _bg_thread
    if _bg_loop is None or not _bg_loop.is_running():
        if sys.platform == "win32":
            _bg_loop = asyncio.ProactorEventLoop()
        else:
            _bg_loop = asyncio.new_event_loop()
        _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        _bg_thread.start()
    return _bg_loop


async def run_pw(coro):
    loop = _ensure_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    while not future.done():
        await asyncio.sleep(0.05)
    return future.result()


# =====================================================================
# 메인 실행 함수
# =====================================================================

async def main():
    pw = await run_pw(async_playwright().start())
    brower = await run_pw(pw.chromium.launch(headless=False))
    page = await run_pw(brower.new_page())

    # ------------------------------------------------------------------
    # 1단계: 스크롤로 데이터 충분히 로드
    # ------------------------------------------------------------------
    url = "http://search.naver.com/search.naver?where=news&query=프로야구+승부예측"
    await run_pw(page.goto(url))
    await run_pw(page.wait_for_timeout(2000))

    tits = page.locator('a[data-heatmap-target=".tit"]')

    MAX_SCROLL = 50
    TARGET_COUNT = 1000

    for i in range(MAX_SCROLL):
        await run_pw(page.keyboard.press("End"))
        await run_pw(page.wait_for_timeout(2000))

        scroll_count = await run_pw(tits.count())
        print(f"스크롤 {i+1}회 - 제목 개수: {scroll_count}")

        if scroll_count >= TARGET_COUNT:
            print("목표 개수 달성, 스크롤 중단")
            break

    # ------------------------------------------------------------------
    # 2단계: 전체 기사 수집
    # ------------------------------------------------------------------
    count = await run_pw(tits.count())
    print(f"수집할 총 기사 수: {count}")

    rows = []

    for i in range(count):
        tit = tits.nth(i)

        title = await run_pw(tit.text_content())
        url_link = await run_pw(tit.get_attribute("href"))

        parent = tit.locator("xpath=..")
        body = parent.locator('a[data-heatmap-target=".body"]')
        has_body = await run_pw(body.count())

        if has_body > 0:
            news_content = await run_pw(body.text_content())
        else:
            news_content = ""

        rows.append({
            "제목": title.strip(),
            "링크주소": url_link,
            "내용": news_content[:200]
        })

        if (i + 1) % 50 == 0:
            print(f"{i+1}개 완료...")

    await run_pw(brower.close())
    await run_pw(pw.stop())

    # ------------------------------------------------------------------
    # 3단계: DataFrame 저장
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["제목"])
    print(f"최종 수집 건수: {len(df)}")

    df.to_csv(
        "c:/data_analysis/01_assignment_2nd/프로야구_뉴스.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"저장 완료 (전체): {len(df)}건")

    df[["제목"]].to_csv(
        "c:/data_analysis/01_assignment_2nd/프로야구_제목.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"저장 완료 (제목): {len(df)}건")


if __name__ == "__main__":
    asyncio.run(main())
