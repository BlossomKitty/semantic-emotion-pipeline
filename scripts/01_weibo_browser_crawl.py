# -*- coding: utf-8 -*-
r"""Fetch personal Weibo text through a real browser context.

This is useful when plain requests-based crawlers receive wbBotDetector pages.
It uses Playwright with a persistent local browser profile, so you can log in
once and reuse the same browser session.

常用命令：

1. 全量抓取，支持断点续跑和逐页落盘：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\01_weibo_browser_crawl.py --delay-ms 20000

2. 只测试前 3 页：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\01_weibo_browser_crawl.py --max-pages 3 --delay-ms 20000

3. 从指定页继续，例如从第 20 页开始：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\01_weibo_browser_crawl.py --start-page 20 --delay-ms 20000

4. 修复已有数据中未展开的“...全文”长微博：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\01_weibo_browser_crawl.py --repair-existing

5. 从已有 raw JSONL 重新生成带时间戳的清洗文件：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\01_weibo_browser_crawl.py --rebuild-processed

首次使用如缺少依赖：
   pip install playwright
   python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_ID = "3666468881"
DEFAULT_PROFILE_DIR = PROJECT_ROOT / ".browser_profiles" / "weibo"
DEFAULT_RAW_JSONL = PROJECT_ROOT / "data" / "raw" / "weibo_browser" / f"{DEFAULT_USER_ID}_original.jsonl"
DEFAULT_TXT = PROJECT_ROOT / "data" / "processed" / "weibo" / f"{DEFAULT_USER_ID}_browser_original_clean.txt"
DEFAULT_PROCESSED_JSONL = PROJECT_ROOT / "data" / "processed" / "weibo" / f"{DEFAULT_USER_ID}_browser_original_clean.jsonl"
DEFAULT_STATE = PROJECT_ROOT / "data" / "raw" / "weibo_browser" / f"{DEFAULT_USER_ID}_browser_state.json"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "01_weibo_browser_crawl.log"


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def clean_text(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def anonymize_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+|网页链接", "[URL]", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[EMAIL]", text)
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE]", text)
    text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[ID_CARD]", text)
    text = re.sub(r"@\S+", "[USER]", text)
    text = re.sub(r"\b(?:QQ|qq|微信|wechat|WeChat)[:：]?\s*[\w-]{5,}\b", "[CONTACT]", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_weibo_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    formats = (
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue
    return None


def format_publish_time(value: Any) -> str:
    parsed = parse_weibo_datetime(value)
    if parsed:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "")


def format_publish_date(value: Any) -> str:
    parsed = parse_weibo_datetime(value)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    text = str(value or "")
    return text[:10] if len(text) >= 10 else text


async def fetch_json_in_page(page, url: str) -> dict[str, Any]:
    result = await page.evaluate(
        """async (url) => {
            const response = await fetch(url, {
                credentials: 'include',
                headers: {
                    'accept': 'application/json, text/plain, */*',
                    'x-requested-with': 'XMLHttpRequest'
                }
            });
            const text = await response.text();
            return {
                status: response.status,
                contentType: response.headers.get('content-type') || '',
                text
            };
        }""",
        url,
    )
    if "application/json" not in result["contentType"] and not result["text"].lstrip().startswith("{"):
        preview = result["text"][:300].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response: status={result['status']} preview={preview}")
    return json.loads(result["text"])


def looks_truncated(text: str) -> bool:
    text = clean_text(text)
    return bool(re.search(r"(?:\.{3}|…)\s*全文", text))


async def enrich_long_text(page, mblog: dict[str, Any]) -> dict[str, Any]:
    text = str(mblog.get("text") or mblog.get("raw_text") or "")
    needs_extend = bool(mblog.get("isLongText")) or looks_truncated(text)
    weibo_id = str(mblog.get("id") or "")
    if not needs_extend or not weibo_id:
        return mblog

    extend_url = f"https://m.weibo.cn/statuses/extend?id={weibo_id}"
    try:
        payload = await fetch_json_in_page(page, extend_url)
    except Exception as exc:
        log(f"Long text fetch failed for {weibo_id}: {exc}")
        return mblog

    long_text = ((payload.get("data") or {}).get("longTextContent") or "").strip()
    if long_text:
        enriched = dict(mblog)
        enriched["text"] = long_text
        enriched["raw_text"] = long_text
        enriched["long_text_fetched"] = True
        return enriched
    return mblog


def parse_cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") or {}
    cards = data.get("cards") or []
    weibos: list[dict[str, Any]] = []
    for card in cards:
        if card.get("card_type") == 11 and card.get("card_group"):
            cards_to_parse = card.get("card_group") or []
        else:
            cards_to_parse = [card]
        for item in cards_to_parse:
            if item.get("card_type") != 9:
                continue
            mblog = item.get("mblog") or {}
            if mblog:
                weibos.append(mblog)
    return weibos


def normalize_weibo(mblog: dict[str, Any], user_id: str, anonymize: bool) -> dict[str, Any] | None:
    if mblog.get("retweeted_status"):
        return None
    text = clean_text(mblog.get("text") or mblog.get("raw_text") or "")
    if anonymize:
        text = anonymize_text(text)
    if not text:
        return None
    return {
        "id": str(mblog.get("id") or ""),
        "bid": mblog.get("bid") or "",
        "user_id": user_id,
        "created_at": mblog.get("created_at") or "",
        "text": text,
        "source": "weibo_browser",
        "attitudes_count": mblog.get("attitudes_count", 0),
        "comments_count": mblog.get("comments_count", 0),
        "reposts_count": mblog.get("reposts_count", 0),
    }


def load_existing_records(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not path.is_file():
        return records, seen
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(record.get("id") or "")
            if not record_id or record_id in seen:
                continue
            records.append(record)
            seen.add(record_id)
    return records, seen


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"completed_pages": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"completed_pages": []}
    if not isinstance(state.get("completed_pages"), list):
        state["completed_pages"] = []
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            parse_weibo_datetime(item.get("created_at")) or datetime.max,
            str(item.get("id") or ""),
        ),
    )


def write_clean_txt(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for item in sorted_records(records):
        created_at = format_publish_time(item.get("created_at"))
        text = str(item.get("text") or "")
        chunks.append(f"[{created_at}]\n{text}" if created_at else text)
    path.write_text("\n\n".join(chunks), encoding="utf-8")


def write_processed_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted_records(records):
            processed = {
                "id": item.get("id", ""),
                "created_at": format_publish_time(item.get("created_at")),
                "date": format_publish_date(item.get("created_at")),
                "source": item.get("source", "weibo_browser"),
                "text": item.get("text", ""),
            }
            handle.write(json.dumps(processed, ensure_ascii=False) + "\n")


def rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted_records(records)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def repair_existing_long_texts(
    page,
    raw_jsonl: Path,
    output_txt: Path,
    processed_jsonl: Path,
    anonymize: bool,
) -> None:
    records, _ = load_existing_records(raw_jsonl)
    if not records:
        log(f"No existing records found: {raw_jsonl}")
        return

    repaired = 0
    candidates = [record for record in records if looks_truncated(str(record.get("text") or ""))]
    log(f"Loaded {len(records)} records, found {len(candidates)} truncated candidates")

    for index, record in enumerate(candidates, start=1):
        weibo_id = str(record.get("id") or "")
        if not weibo_id:
            continue
        log(f"Repairing {index}/{len(candidates)}: {weibo_id}")
        extend_url = f"https://m.weibo.cn/statuses/extend?id={weibo_id}"
        try:
            payload = await fetch_json_in_page(page, extend_url)
        except Exception as exc:
            log(f"Repair failed for {weibo_id}: {exc}")
            continue
        long_text = clean_text(((payload.get("data") or {}).get("longTextContent") or "").strip())
        if anonymize:
            long_text = anonymize_text(long_text)
        if long_text and not looks_truncated(long_text):
            record["text"] = long_text
            record["long_text_fetched"] = True
            repaired += 1

    rewrite_jsonl(raw_jsonl, records)
    write_clean_txt(output_txt, records)
    write_processed_jsonl(processed_jsonl, records)
    log(f"Repaired {repaired} records")
    log(f"Raw JSONL: {raw_jsonl}")
    log(f"Processed JSONL: {processed_jsonl}")
    log(f"Clean TXT: {output_txt}")


def scan_truncated(raw_jsonl: Path, processed_jsonl: Path) -> None:
    raw_records, _ = load_existing_records(raw_jsonl)
    processed_records, _ = load_existing_records(processed_jsonl)
    raw_candidates = [record for record in raw_records if looks_truncated(str(record.get("text") or ""))]
    processed_candidates = [record for record in processed_records if looks_truncated(str(record.get("text") or ""))]
    log(f"Raw records: {len(raw_records)}, truncated: {len(raw_candidates)}")
    log(f"Processed records: {len(processed_records)}, truncated: {len(processed_candidates)}")
    for record in processed_candidates[:20]:
        log(f"- {record.get('created_at')} id={record.get('id')}")


async def run(args: argparse.Namespace) -> None:
    raw_jsonl = Path(args.raw_jsonl).resolve()
    output_txt = Path(args.output_txt).resolve()
    processed_jsonl = Path(args.processed_jsonl).resolve()
    state_path = Path(args.state).resolve()
    raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_processed:
        records, _ = load_existing_records(raw_jsonl)
        write_clean_txt(output_txt, records)
        write_processed_jsonl(processed_jsonl, records)
        log(f"Rebuilt processed outputs from {len(records)} records")
        log(f"Processed JSONL: {processed_jsonl}")
        log(f"Clean TXT: {output_txt}")
        return

    if args.scan_truncated:
        scan_truncated(raw_jsonl, processed_jsonl)
        return

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install playwright && python -m playwright install chromium") from exc

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            viewport={"width": 1280, "height": 900},
            locale="zh-CN",
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(f"https://m.weibo.cn/u/{args.user_id}", wait_until="domcontentloaded")

        input("If Weibo asks for login or verification, complete it in the browser, then press Enter here...")

        if args.repair_existing:
            await repair_existing_long_texts(page, raw_jsonl, output_txt, processed_jsonl, args.anonymize)
            await context.close()
            return

        user_url = f"https://m.weibo.cn/api/container/getIndex?containerid=100505{args.user_id}"
        user_payload = await fetch_json_in_page(page, user_url)
        user_info = (user_payload.get("data") or {}).get("userInfo") or {}
        screen_name = user_info.get("screen_name", "")
        statuses_count = int(user_info.get("statuses_count") or 0)
        full_page_count = max(1, math.ceil(statuses_count / args.count))
        page_count = args.max_pages or full_page_count
        log(f"User: {screen_name} ({args.user_id}), statuses={statuses_count}, pages={page_count}")

        records, seen = load_existing_records(raw_jsonl)
        state = load_state(state_path) if args.resume else {"completed_pages": []}
        completed_pages = {int(page) for page in state.get("completed_pages", [])}
        if records:
            log(f"Loaded {len(records)} existing records")
        if completed_pages and args.resume:
            log(f"Resume enabled, completed pages: {min(completed_pages)}-{max(completed_pages)} ({len(completed_pages)} pages)")

        since_date = datetime.strptime(args.since_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

        with raw_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
            for page_no in range(args.start_page, page_count + 1):
                if args.resume and page_no in completed_pages:
                    log(f"Skipping completed page {page_no}/{page_count}")
                    continue
                list_url = (
                    "https://m.weibo.cn/api/container/getIndex?"
                    f"containerid=230413{args.user_id}&page={page_no}&count={args.count}"
                )
                log(f"Fetching page {page_no}/{page_count}")
                payload = await fetch_json_in_page(page, list_url)
                page_records = 0
                for mblog in parse_cards(payload):
                    mblog = await enrich_long_text(page, mblog)
                    record = normalize_weibo(mblog, args.user_id, args.anonymize)
                    if not record or record["id"] in seen:
                        continue
                    created_dt = parse_weibo_datetime(record.get("created_at"))
                    if created_dt and created_dt < since_date:
                        log("Reached since_date boundary.")
                        page_count = page_no
                        break
                    if created_dt and end_date and created_dt > end_date:
                        continue
                    seen.add(record["id"])
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    page_records += 1
                handle.flush()
                completed_pages.add(page_no)
                state.update(
                    {
                        "user_id": args.user_id,
                        "screen_name": screen_name,
                        "statuses_count": statuses_count,
                        "full_page_count": full_page_count,
                        "last_completed_page": page_no,
                        "completed_pages": sorted(completed_pages),
                    }
                )
                write_state(state_path, state)
                write_clean_txt(output_txt, records)
                write_processed_jsonl(processed_jsonl, records)
                log(f"Saved {page_records} original records from page {page_no}")
                if page_no >= page_count:
                    break
                await page.wait_for_timeout(args.delay_ms)

        write_clean_txt(output_txt, records)
        write_processed_jsonl(processed_jsonl, records)
        log(f"Raw JSONL: {raw_jsonl}")
        log(f"Processed JSONL: {processed_jsonl}")
        log(f"Clean TXT: {output_txt}")
        log(f"State: {state_path}")
        await context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl Weibo through a real browser context")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--raw-jsonl", default=str(DEFAULT_RAW_JSONL))
    parser.add_argument("--output-txt", default=str(DEFAULT_TXT))
    parser.add_argument("--processed-jsonl", default=str(DEFAULT_PROCESSED_JSONL))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means crawl all available pages")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--delay-ms", type=int, default=12000)
    parser.add_argument("--since-date", default="1900-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--anonymize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-existing", action="store_true", help="repair already saved records with truncated long text")
    parser.add_argument("--rebuild-processed", action="store_true", help="rebuild processed JSONL/TXT from existing raw JSONL")
    parser.add_argument("--scan-truncated", action="store_true", help="scan existing raw/processed files for truncated long text")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
