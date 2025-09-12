#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Scrapes the CER year page to find the current CSV URL (or accept --csv-url).
- Downloads and cleans to a consistent schema.
- Output: data/external/nger_<year>.csv and data/interim/nger_<year>_clean.parquet

Example (2023–24 with methane breakdown):
  python src/nger_ingest.py --year 2023-24

Optional direct CSV URL:
  python src/nger_ingest.py --year 2023-24 --csv-url "https://cer.gov.au/.../2023-24-baselines-and-emissions-table.csv"
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import sys
import io
import requests
import pandas as pd

CER_PAGES = {
    # Where to scrape the CSV link for each reporting year
    "2023-24": "https://cer.gov.au/markets/reports-and-data/safeguard-data/2023-24-baselines-and-emissions-data",
    "2022-23": "https://cer.gov.au/markets/reports-and-data/safeguard-facility-covered-emissions-data-2022-23",
}

# Detect useful columns in the CSV across schema variants
NAME_COL_CANDS = ["Facility name", "Facility", "facility", "FACILITY", "Facility_Name"]
EMITTER_COL_CANDS = [
    "Responsible emitter",
    "Responsible emitter name",
    "Company",
    "Emitter",
]
STATE_COL_CANDS = ["State", "Jurisdiction"]
TOTAL_COL_CANDS = [
    "Total covered emissions (t CO2-e)",
    "Covered emissions (t CO2-e)",
    "Total covered emissions",
    "Covered Emissions",
]
CH4_COL_CANDS = [
    "Methane (t CO2-e)",
    "CH4 (t CO2-e)",
    "Methane tCO2-e",
    "Covered emissions – methane (t CO2-e)",
    "Methane",
]


def choose_first(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # Try case-insensitive fallback
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def find_csv_link_from_page(url: str) -> str:
    """Fetch CER page and extract the first CSV link."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # Very light scrape: look for hrefs that end with .csv
    hrefs = re.findall(r'href="([^"]+\.csv)"', r.text, flags=re.IGNORECASE)
    if not hrefs:
        raise RuntimeError("Could not find a CSV link on the CER page.")
    # Many CER links are relative; normalize
    from urllib.parse import urljoin

    return urljoin(url, hrefs[0])


def load_csv(csv_url: str) -> pd.DataFrame:
    r = requests.get(csv_url, timeout=60)
    r.raise_for_status()
    # Some CER CSVs are tiny; read from memory
    return pd.read_csv(io.BytesIO(r.content), encoding="utf-8")


def clean_table(df: pd.DataFrame, year: str) -> pd.DataFrame:
    # Strip whitespace from headers
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Pick columns
    c_fac = choose_first(df, NAME_COL_CANDS) or "facility"
    c_emit = choose_first(df, EMITTER_COL_CANDS)
    c_state = choose_first(df, STATE_COL_CANDS)
    c_total = choose_first(df, TOTAL_COL_CANDS)
    c_ch4 = choose_first(df, CH4_COL_CANDS)  # might be None for 2022–23 and older

    # Build output frame with harmonized names
    out = pd.DataFrame()
    out["year"] = year
    out["facility"] = df[c_fac].astype(str).str.strip() if c_fac in df else pd.NA
    out["responsible_emitter"] = (
        df[c_emit].astype(str).str.strip() if c_emit in df else pd.NA
    )
    out["state"] = df[c_state].astype(str).str.strip() if c_state in df else pd.NA

    def _to_float(s):
        try:
            return pd.to_numeric(s, errors="coerce")
        except Exception:
            return pd.Series([pd.NA] * len(s))

    out["covered_emissions_tCO2e"] = _to_float(df[c_total]) if c_total in df else pd.NA
    out["methane_tCO2e"] = _to_float(df[c_ch4]) if c_ch4 in df else pd.NA

    # Add convenient flags/meta
    out["has_methane_breakdown"] = out["methane_tCO2e"].notna()
    # No coordinates in CER tables; keep placeholders for future geocoding
    out["lon"] = pd.NA
    out["lat"] = pd.NA

    # Drop obvious empty rows
    out = out.dropna(subset=["facility"], how="any")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Ingest NGER/Safeguard facility data from CER (CSV as API)."
    )
    ap.add_argument(
        "--year", required=True, help='Reporting year label, e.g. "2023-24"'
    )
    ap.add_argument(
        "--csv-url",
        default=None,
        help="Optional direct CSV URL (skip page scrape).",
    )
    ap.add_argument(
        "--out-dir-external",
        default="data/external",
        help="Where to save the raw CSV.",
    )
    ap.add_argument(
        "--out-dir-interim",
        default="data/interim",
        help="Where to save the cleaned Parquet.",
    )
    args = ap.parse_args()

    year = args.year.strip()
    page_url = CER_PAGES.get(year)
    if not args.csv - url and not page_url:
        print(
            f"[error] Don’t know the CER page for {year}. Provide --csv-url explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Resolve CSV URL
    if args.csv_url:
        csv_url = args.csv_url
    else:
        csv_url = find_csv_link_from_page(page_url)

    print(f"[info] Year: {year}")
    print(f"[info] CSV:  {csv_url}")

    # Fetch, clean, and write
    df_raw = load_csv(csv_url)
    df_clean = clean_table(df_raw, year)

    # Outputs
    out_dir_ext = Path(args.out_dir_external)
    out_dir_ext.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir_ext / f"nger_{year.replace('/','-')}.csv"
    df_raw.to_csv(out_csv, index=False)
    print(f"[OK] raw CSV  → {out_csv} (rows={len(df_raw)})")

    out_dir_int = Path(args.out_dir_interim)
    out_dir_int.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir_int / f"nger_{year.replace('/','-')}_clean.parquet"
    try:
        df_clean.to_parquet(out_parquet, index=False)
        print(f"[OK] clean PQ → {out_parquet} (rows={len(df_clean)})")
    except Exception as e:
        print(f"[warn] Parquet write failed ({e}); writing CSV fallback.")
        out_csv_clean = out_dir_int / f"nger_{year.replace('/','-')}_clean.csv"
        df_clean.to_csv(out_csv_clean, index=False)
        print(f"[OK] clean CSV → {out_csv_clean} (rows={len(df_clean)})")


if __name__ == "__main__":
    main()
