#!/usr/bin/env python3
import argparse, hashlib, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE = "https://ftp.sron.nl/pub/memo/CSVs"


def md5sum(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def filename_for_week(year: int, week: int) -> str:
    # SRON’s naming conventions as: SRON_Weekly_Methane_Plumes_{year}_wk{week:02d}_vYYYYMMDD.csv
    # The version date can vary; we’ll first try without version, then fall back to listing if needed.
    # Easiest robust approach: fetch directory index and pick the newest that matches year/week.
    return f"SRON_Weekly_Methane_Plumes_{year}_wk{week:02d}"


def find_best_match(stem: str) -> str:
    idx = requests.get(BASE + "/", timeout=60)
    idx.raise_for_status()
    # crude match of filenames in listing
    import re

    pattern = re.compile(rf'href="({stem}_v\d{{8}}\.csv)"', re.IGNORECASE)
    matches = pattern.findall(idx.text)
    if not matches:
        raise FileNotFoundError(f"No match found on index for {stem}")
    # pick lexicographically max (latest version date in name)
    return max(matches)


def download_one(url: str, out_dir: Path, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    out_path = out_dir / name
    if out_path.exists() and not force:
        print(f"[skip] {name} exists")
        return out_path

    with requests.get(url, stream=True, timeout=300) as r:
        if r.status_code == 404:
            raise FileNotFoundError(f"404 for {url}")
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        with out_path.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    got += len(chunk)
        print(
            f"[ok] {name} ({got/1e6:.1f} MB{' of '+str(total/1e6)+' MB' if total else ''})"
        )
    print(f"[md5] {name}: {md5sum(out_path)}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Download SRON weekly CSVs from SRON FTP/HTTP."
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Year(s), e.g., 2022 2023 2024",
    )
    ap.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        default=list(range(1, 54)),
        help="Week numbers (1..53)",
    )
    ap.add_argument("--out", default="data/raw/sron", help="Output directory")
    ap.add_argument(
        "--force", action="store_true", help="Redownload even if file exists"
    )
    ap.add_argument("--max-workers", type=int, default=4, help="Parallel downloads")
    args = ap.parse_args()

    out_dir = Path(args.out)
    jobs = []
    for y in args.years:
        for w in args.weeks:
            stem = filename_for_week(y, w)
            try:
                fname = find_best_match(stem)
                url = f"{BASE}/{fname}"
                jobs.append(url)
            except FileNotFoundError:
                print(f"[miss] No file published for {y} wk{w:02d}")

    if not jobs:
        print("No files to download. Check years/weeks.", file=sys.stderr)
        sys.exit(1)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(download_one, url, out_dir, args.force) for url in jobs]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
# run as python src\config\ingest\download_sron_ftp.py --years 2024
