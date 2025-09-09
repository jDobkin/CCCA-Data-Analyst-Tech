#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carbon Mapper API ingest (annotated methane plumes)
- Loads .env
- Verbose progress + clear errors
- BBox/date filtering (client-side)
- Writes JSONL append-only archive
"""

import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:
    load_dotenv = None

import requests

DEF_API_BASE = "https://api.carbonmapper.org/api/v1"
RAW_DIR = Path("data/raw/carbonmapper")


def load_env():
    # Load .env if available
    if load_dotenv:
        load_dotenv()
    api_base = os.getenv("CARBON_MAPPER_API_BASE", DEF_API_BASE).rstrip("/")
    token = os.getenv("CARBON_MAPPER_API_TOKEN", "")
    return api_base, token


def headers(token: str) -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if token:
        # try bearer; we’ll fallback to x-api-key on 401/403
        h["Authorization"] = f"Bearer {token}"
        h["x-api-key"] = token
    return h


def parse_args():
    ap = argparse.ArgumentParser(
        description="Download Carbon Mapper annotated plumes (CH4)."
    )
    ap.add_argument(
        "--bbox", type=float, nargs=4, default=None, help="minlon minlat maxlon maxlat"
    )
    ap.add_argument(
        "--start", type=str, default=None, help="UTC start date (YYYY-MM-DD)"
    )
    ap.add_argument("--end", type=str, default=None, help="UTC end date (YYYY-MM-DD)")
    ap.add_argument(
        "--gas", type=str, default="CH4", help="Gas filter hint (default CH4)"
    )
    ap.add_argument("--limit", type=int, default=1000, help="Page size")
    ap.add_argument("--max-rows", type=int, default=20000, help="Stop after N rows")
    ap.add_argument(
        "--out-jsonl", type=str, default=str(RAW_DIR / "plumes_annotated.jsonl")
    )
    ap.add_argument("--endpoint", type=str, default="/catalog/plumes/annotated")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def to_utc_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def within_bbox(
    lon: float, lat: float, bbox: Tuple[float, float, float, float]
) -> bool:
    minx, miny, maxx, maxy = bbox
    return (minx <= lon <= maxx) and (miny <= lat <= maxy)


def fetch_page(
    base: str, ep: str, token: str, params: Dict[str, Any], timeout: int, verbose: bool
) -> Dict[str, Any]:
    url = f"{base}{ep}"
    h = headers(token)
    r = requests.get(url, headers=h, params=params, timeout=timeout)
    if verbose:
        print(f"GET {r.url} -> {r.status_code}")
    if r.status_code in (401, 403) and token:
        # try again without Bearer (some APIs only accept x-api-key)
        h.pop("Authorization", None)
        r = requests.get(url, headers=h, params=params, timeout=timeout)
        if verbose:
            print(f"Retry (x-api-key only) {r.url} -> {r.status_code}")
    r.raise_for_status()
    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Non-JSON response from {url}: {e}")


def main():
    args = parse_args()
    api_base, token = load_env()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Echo config
    print(">>> carbon_mapper_api.py starting")
    print(f"cwd: {Path.cwd()}")
    print(f"api_base: {api_base}")
    print(f"endpoint: {args.endpoint}")
    print(f"token_present: {bool(token)}")
    print(f"bbox: {args.bbox}  start: {args.start}  end: {args.end}  gas: {args.gas}")
    print(f"out: {args.out_jsonl}  limit: {args.limit}  max_rows: {args.max_rows}")
    if not token:
        print(
            "[warn] CARBON_MAPPER_API_TOKEN is empty. You likely need a token to access data."
        )

    # Parse date filters
    dt_start = to_utc_date(args.start)
    dt_end = to_utc_date(args.end)

    offset = 0
    written = 0
    pages = 0

    params = {
        "limit": args.limit,
        "offset": offset,
        "sort": "desc",
        "gas": args.gas,  # harmless if server ignores it; we filter client-side anyway
    }

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fout = outp.open("a", encoding="utf-8")

    try:
        while True:
            params["offset"] = offset
            try:
                data = fetch_page(
                    api_base, args.endpoint, token, params, args.timeout, args.verbose
                )
            except requests.HTTPError as e:
                print(f"[HTTP ERROR] {e.response.status_code} {e.response.text[:200]}")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                break

            items = data.get("items") or data.get("data") or []
            total = data.get("total_count")
            pages += 1
            print(
                f"[page {pages}] got {len(items)} items (offset={offset}) total={total}"
            )

            if not items:
                break

            kept_this_page = 0
            for it in items:
                # Extract coords if present
                lon = lat = None
                geom = it.get("geometry_json") or it.get("geometry") or {}
                coords = geom.get("coordinates") if isinstance(geom, dict) else None
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    lon, lat = float(coords[0]), float(coords[1])

                # Client-side filters
                if args.bbox and (
                    lon is None
                    or lat is None
                    or not within_bbox(lon, lat, tuple(args.bbox))
                ):
                    continue

                ts = (
                    it.get("scene_timestamp")
                    or it.get("timestamp")
                    or it.get("acquisition_time")
                )
                if (dt_start or dt_end) and ts:
                    try:
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    except Exception:
                        dt = None
                    if dt:
                        if dt_start and dt < dt_start:
                            continue
                        if dt_end and dt > dt_end:
                            continue

                fout.write(json.dumps(it) + "\n")
                written += 1
                kept_this_page += 1

                if written >= args.max_rows:
                    break

            print(f"  kept {kept_this_page} items this page (written total={written})")

            if written >= args.max_rows:
                break

            got = len(items)
            offset += got
            if got == 0 or (total is not None and offset >= int(total)):
                break

            time.sleep(0.2)  # be polite

        print(f"[OK] Appended {written} items → {outp}")
    finally:
        fout.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] user cancelled")
        sys.exit(1)
