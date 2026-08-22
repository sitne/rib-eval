#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "replays"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

FETCH_JS = """
async (url) => {
    const r = await fetch(url);
    const t = await r.text();
    return {status: r.status, body: t};
}
"""


def open_session(pw):
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=UA)
    page = ctx.new_page()
    page.goto("https://rib.gg/matches", wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(3000)
    return browser, page


def probe(page, match_id):
    res = page.evaluate(FETCH_JS, f"https://rib.gg/api/matches/{match_id}/replay-data?mapId={match_id}-m1")
    return res["status"] == 200


def fetch_match(page, match_id):
    got = []
    for mi in range(1, 4):
        map_id = f"{match_id}-m{mi}"
        out = CACHE / f"{map_id}.json"
        if out.exists():
            got.append(map_id)
            continue
        res = page.evaluate(FETCH_JS, f"https://rib.gg/api/matches/{match_id}/replay-data?mapId={map_id}")
        if res["status"] != 200:
            break
        try:
            json.loads(res["body"])
        except json.JSONDecodeError:
            break
        out.write_text(res["body"])
        got.append(map_id)
        print(f"  saved {map_id} ({len(res['body']) // 1024} KB)", flush=True)
        time.sleep(1.0)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", help="explicit match ids")
    ap.add_argument("--scan", nargs=2, type=int, metavar=("LO", "HI"), help="probe id range for existing replays")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser, page = open_session(pw)
        ids = list(args.matches or [])
        if args.scan:
            lo, hi = args.scan
            print(f"scanning {lo}..{hi}", flush=True)
            for mid in range(lo, hi):
                if probe(page, str(mid)):
                    print(f"[hit] {mid}", flush=True)
                    if str(mid) not in ids:
                        ids.append(str(mid))
                time.sleep(0.3)
        if args.limit:
            ids = ids[: args.limit]
        total = 0
        for mid in ids:
            print(f"match {mid}", flush=True)
            total += len(fetch_match(page, mid))
        browser.close()
    print(f"done: {total} map replays in cache")


if __name__ == "__main__":
    main()
