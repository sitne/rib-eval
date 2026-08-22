#!/usr/bin/env python3
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"
INFO = CACHE / "match_info.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

DATE_RE = re.compile(r'"startDate":"(\d{4}-\d{2}-\d{2})T')


def main():
    info = json.loads(INFO.read_text()) if INFO.exists() else {}
    ids = sorted({f.stem.split("-")[0] for f in (CACHE / "replays").glob("*.json")}, key=int)
    todo = [m for m in ids if m not in info]
    print(f"{len(ids)} matches, {len(todo)} to fetch", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()
        done = 0
        for mid in todo:
            try:
                r = page.goto(f"https://rib.gg/matches/{mid}", wait_until="domcontentloaded", timeout=30000)
                if r.status != 200:
                    info[mid] = {"title": None, "date": None}
                else:
                    html = page.content()
                    m = DATE_RE.search(html)
                    info[mid] = {
                        "title": page.title().split(" | ")[0].strip(),
                        "date": m.group(1) if m else None,
                    }
            except Exception as e:
                print(f"[err] {mid}: {e}", flush=True)
                continue
            done += 1
            if done % 20 == 0:
                INFO.write_text(json.dumps(info, ensure_ascii=False, indent=0))
                print(f"... {done}/{len(todo)}", flush=True)
            time.sleep(0.7)
        browser.close()

    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=0))
    ok = sum(1 for v in info.values() if v.get("title"))
    print(f"done: {ok}/{len(info)} with titles")


if __name__ == "__main__":
    main()
