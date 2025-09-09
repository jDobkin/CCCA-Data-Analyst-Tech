#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean + Merge Carbon Mapper with existing TROPOMI unified file.

Features
- Optional API ingest -> JSONL archive (data/raw/carbonmapper/plumes_annotated.jsonl)
- Clean/validate -> GeoJSON + Parquet (data/interim/cm_plumes_clean.*)
- Merge with unified TROPOMI file (data/interim/plumes_unified.parquet)
- Emits merged Parquet/GeoPackage ready for cross-provider dedup (your existing src/dedup.py)

CLI examples (run from repo root):
  # Just clean + merge an existing JSONL (already ingested):
  python src/clean/clean_and_merge_carbonmapper.py --clip-au

  # Ingest from API (Australia bbox + date window), then clean + merge:
  python src/clean/clean_and_merge_carbonmapper.py --ingest --bbox 112 -44 154 -9 --start 2019-01-01 --end 2025-12-31 --clip-au

  # After this script, run the cross-provider dedup using the merged output:
  python src/dedup.py --in-parquet data/interim/unified_plus_cm.parquet
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Dotenv
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests

# Config

DEF_API_BASE = "https://api.carbonmapper.org/api/v1"
RAW_JSONL = Path("data/raw/carbonmapper/plumes_annotated.jsonl")
OUT_CM_GEOJSON = Path("data/interim/cm_plumes_clean.geojson")
OUT_CM_PARQUET = Path("data/interim/cm_plumes_clean.parquet")

UNIFIED_IN_DEFAULT = Path(
    "data/interim/plumes_unified.parquet"
)  # your SRON+Kayrros unified
MERGED_OUT_PARQUET = Path("data/interim/unified_plus_cm.parquet")
MERGED_OUT_GPKG = Path("data/interim/unified_plus_cm.gpkg")

AUS_BBOX = (112.0, -44.0, 154.0, -9.0)
KM = 1000.0

# --------------------------- Utils ---------------------------


def load_env():
    if load_dotenv:
        load_dotenv()
    api_base = os.getenv("CARBON_MAPPER_API_BASE", DEF_API_BASE).rstrip("/")
    token = os.getenv("CARBON_MAPPER_API_TOKEN", "")
    return api_base, token


def headers(token: str) -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
        h["x-api-key"] = token  # fallback pattern
    return h


def within_bbox(
    lon: float, lat: float, bbox: Tuple[float, float, float, float]
) -> bool:
    minx, miny, maxx, maxy = bbox
    return (minx <= lon <= maxx) and (miny <= lat <= maxy)


def to_utc_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def safe_to_parquet(gdf: gpd.GeoDataFrame, path: Path):
    try:
        gdf.to_parquet(path, index=False)
        print(f"[OK] Parquet → {path}")
    except Exception as e:
        print(f"[warn] Parquet write failed (install pyarrow): {e}")


# Ingest


def fetch_page(
    base: str,
    endpoint: str,
    token: str,
    params: Dict[str, Any],
    timeout: int,
    verbose: bool,
) -> Dict[str, Any]:
    url = f"{base}{endpoint}"
    h = headers(token)
    r = requests.get(url, headers=h, params=params, timeout=timeout)
    if verbose:
        print(f"GET {r.url} -> {r.status_code}")
    if r.status_code in (401, 403) and token:
        h.pop("Authorization", None)
        r = requests.get(url, headers=h, params=params, timeout=timeout)
        if verbose:
            print(f"Retry (x-api-key only) {r.url} -> {r.status_code}")
    r.raise_for_status()
    return r.json()


def ingest_api(
    jsonl_out: Path,
    bbox: Optional[Tuple[float, float, float, float]],
    start: Optional[str],
    end: Optional[str],
    gas: str,
    limit: int,
    max_rows: int,
    endpoint: str,
    timeout: int,
    verbose: bool,
) -> int:
    api_base, token = load_env()
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    print(">>> Ingest from Carbon Mapper API")
    print(f"api_base={api_base} endpoint={endpoint} token_present={bool(token)}")
    print(
        f"bbox={bbox} start={start} end={end} gas={gas} limit={limit} max_rows={max_rows}"
    )

    dt_start = to_utc_date(start)
    dt_end = to_utc_date(end)

    params = {"limit": limit, "offset": 0, "sort": "desc", "gas": gas}
    written = 0
    pages = 0

    with jsonl_out.open("a", encoding="utf-8") as fout:
        while True:
            try:
                data = fetch_page(api_base, endpoint, token, params, timeout, verbose)
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
                f"[page {pages}] got {len(items)} items (offset={params['offset']}) total={total}"
            )

            if not items:
                break

            kept = 0
            for it in items:
                lon = lat = None
                geom = it.get("geometry_json") or it.get("geometry") or {}
                coords = geom.get("coordinates") if isinstance(geom, dict) else None
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    lon, lat = float(coords[0]), float(coords[1])

                # client-side bbox filter
                if bbox and (
                    lon is None or lat is None or not within_bbox(lon, lat, bbox)
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
                kept += 1
                if written >= max_rows:
                    break

            print(f"  kept {kept} items this page (written total={written})")

            if written >= max_rows:
                break

            got = len(items)
            params["offset"] += got
            if got == 0 or (total is not None and params["offset"] >= int(total)):
                break

            time.sleep(0.2)

    print(f"[OK] Appended {written} records → {jsonl_out}")
    return written


# Clean carbonmapper data


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def clean_carbon_mapper(in_jsonl: Path, clip_au: bool) -> gpd.GeoDataFrame:
    raw = load_jsonl(in_jsonl)
    if not raw:
        print(f"[warn] No records found in {in_jsonl}")
        return gpd.GeoDataFrame(columns=[], geometry=[], crs="EPSG:4326")

    rows = []
    for it in raw:
        geom = it.get("geometry_json") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        lon, lat = (None, None)
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            lon, lat = float(coords[0]), float(coords[1])

        plume_id = it.get("plume_id") or it.get("id")
        scene_ts = it.get("scene_timestamp") or it.get("timestamp")
        gas = it.get("gas")
        emission = it.get("emission_auto")  # kg/hr (per examples)
        emission_unc = it.get("emission_uncertainty_auto")
        sector = it.get("sector")
        instrument = it.get("instrument")
        platform = it.get("platform")

        rows.append(
            {
                "provider": "CARBON_MAPPER",
                "plume_id_src": plume_id,
                "gas": gas,
                "obs_datetime_utc": scene_ts,
                "lon": lon,
                "lat": lat,
                "emission_tph": (
                    (float(emission) / 1000.0) if emission is not None else None
                ),
                "emission_unc_tph": (
                    (float(emission_unc) / 1000.0) if emission_unc is not None else None
                ),
                "sector": sector,
                "instrument": instrument,
                "platform": platform,
            }
        )

    df = pd.DataFrame(rows)
    # datetime
    df["obs_datetime_utc"] = pd.to_datetime(
        df["obs_datetime_utc"], utc=True, errors="coerce"
    )
    # coords sane
    df = df[
        (df["lon"].between(-180, 180, inclusive="both"))
        & (df["lat"].between(-90, 90, inclusive="both"))
    ]
    # non-negative
    for c in ["emission_tph", "emission_unc_tph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[(df[c].notna()) & (df[c] < 0), c] = pd.NA

    # clip AU (bbox)
    if clip_au:
        minx, miny, maxx, maxy = AUS_BBOX
        df = df[
            (df["lon"] >= minx)
            & (df["lon"] <= maxx)
            & (df["lat"] >= miny)
            & (df["lat"] <= maxy)
        ]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[
            Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None
            for xy in zip(df["lon"], df["lat"])
        ],
        crs="EPSG:4326",
    ).dropna(subset=["geometry"])

    # light intra-CM dedup: 30 min, 5 km
    gdf = gdf.sort_values("obs_datetime_utc").reset_index(drop=True)
    gdf_m = gdf.to_crs(3857)
    max_dt = pd.Timedelta(minutes=30)
    max_dist_m = 5 * KM
    to_drop = set()

    for i in range(len(gdf_m)):
        if i in to_drop:
            continue
        ti = gdf_m.at[i, "obs_datetime_utc"]
        gi = gdf_m.at[i, "geometry"]
        if pd.isna(ti) or gi is None:
            continue
        lo, hi = ti - max_dt, ti + max_dt
        win = gdf_m[
            (gdf_m["obs_datetime_utc"] >= lo) & (gdf_m["obs_datetime_utc"] <= hi)
        ]
        if len(win) <= 1:
            continue
        dists = win.geometry.distance(gi)
        close = win[(dists <= max_dist_m)]
        if len(close) > 1:
            idxs = close.index.tolist()
            for j in idxs[1:]:
                to_drop.add(j)

    gdf_clean = gdf_m.drop(index=list(to_drop)).copy().to_crs(4326)
    return gdf_clean


# Merge


def read_unified_tropomi(unified_in: Path) -> gpd.GeoDataFrame:
    if not unified_in.exists():
        raise FileNotFoundError(f"Unified TROPOMI file not found: {unified_in}")
    # try parquet first, fallback to GPKG
    gdf: Optional[gpd.GeoDataFrame] = None
    if unified_in.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(unified_in)
    elif unified_in.suffix.lower() in (".gpkg", ".geopackage"):
        gdf = gpd.read_file(unified_in)
    else:
        # try both
        try:
            gdf = gpd.read_parquet(unified_in)
        except Exception:
            gdf = gpd.read_file(unified_in)
    return gdf


def merge_frames(
    unified_tropomi: gpd.GeoDataFrame, cm: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # Ensure columns align we keep a superset and fill missing
    needed_cols = [
        "provider",
        "plume_id_src",
        "obs_datetime_utc",
        "lon",
        "lat",
        "emission_tph",
        "emission_unc_tph",
        "sector",
        "instrument",
        "platform",
    ]
    for df in (unified_tropomi, cm):
        for c in needed_cols:
            if c not in df.columns:
                df[c] = None

    # dtypes for key columns
    unified_tropomi["obs_datetime_utc"] = pd.to_datetime(
        unified_tropomi["obs_datetime_utc"], utc=True, errors="coerce"
    )
    cm["obs_datetime_utc"] = pd.to_datetime(
        cm["obs_datetime_utc"], utc=True, errors="coerce"
    )

    # Concat & ensure geometry/crs
    gdf_all = gpd.GeoDataFrame(
        pd.concat(
            [
                unified_tropomi[needed_cols + ["geometry"]],
                cm[needed_cols + ["geometry"]],
            ],
            ignore_index=True,
        ),
        geometry="geometry",
        crs=unified_tropomi.crs if unified_tropomi.crs else "EPSG:4326",
    )

    # make sure lon/lat cols exist (helpful for web exports)
    if "lon" not in gdf_all.columns or "lat" not in gdf_all.columns:
        gdf_all["lon"] = gdf_all.geometry.x
        gdf_all["lat"] = gdf_all.geometry.y

    return gdf_all


# CLI


def parse_args():
    ap = argparse.ArgumentParser(
        description="Ingest (optional) + Clean Carbon Mapper + Merge with TROPOMI unified."
    )
    ap.add_argument(
        "--ingest", action="store_true", help="Call Carbon Mapper API before cleaning."
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        help="minlon minlat maxlon maxlat for ingest filter",
    )
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="UTC start date (YYYY-MM-DD) for ingest filter",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="UTC end date (YYYY-MM-DD) for ingest filter",
    )
    ap.add_argument(
        "--gas", type=str, default="CH4", help="Gas filter hint (default CH4)"
    )
    ap.add_argument("--limit", type=int, default=1000, help="Ingest page size")
    ap.add_argument("--max-rows", type=int, default=50000, help="Ingest hard cap")
    ap.add_argument(
        "--endpoint",
        type=str,
        default="/catalog/plumes/annotated",
        help="API endpoint path",
    )
    ap.add_argument("--timeout", type=int, default=60)

    ap.add_argument(
        "--in-jsonl",
        type=str,
        default=str(RAW_JSONL),
        help="Input JSONL (if not ingesting now)",
    )
    ap.add_argument(
        "--clip-au", action="store_true", help="Clip cleaned CM to Australia bbox."
    )

    ap.add_argument(
        "--unified-in",
        type=str,
        default=str(UNIFIED_IN_DEFAULT),
        help="Path to existing unified TROPOMI file",
    )
    ap.add_argument("--merged-parquet", type=str, default=str(MERGED_OUT_PARQUET))
    ap.add_argument("--merged-gpkg", type=str, default=str(MERGED_OUT_GPKG))

    ap.add_argument("--out-cm-geojson", type=str, default=str(OUT_CM_GEOJSON))
    ap.add_argument("--out-cm-parquet", type=str, default=str(OUT_CM_PARQUET))

    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    print(">>> clean_and_merge_carbonmapper starting")
    print(f"cwd: {Path.cwd()}")

    # 1) Optional ingest
    if args.ingest:
        bbox = tuple(args.bbox) if args.bbox else None
        RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
        ingest_api(
            jsonl_out=Path(args.in_jsonl),
            bbox=bbox,
            start=args.start,
            end=args.end,
            gas=args.gas,
            limit=args.limit,
            max_rows=args.max_rows,
            endpoint=args.endpoint,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    # 2) Clean Carbon Mapper
    gdf_cm = clean_carbon_mapper(Path(args.in_jsonl), clip_au=args.clip_au)
    if gdf_cm.empty:
        print("[warn] Carbon Mapper clean produced 0 rows.")
    else:
        Path(args.out_cm_geojson).parent.mkdir(parents=True, exist_ok=True)
        gdf_cm.to_file(args.out_cm_geojson, driver="GeoJSON")
        print(f"[OK] CM GeoJSON → {args.out_cm_geojson}")
        safe_to_parquet(gdf_cm, Path(args.out_cm_parquet))

    # 3) Merge with existing unified TROPOMI (SRON+Kayrros)
    uni_path = Path(args.unified_in)
    gdf_uni = read_unified_tropomi(uni_path)
    # Ensure WGS84
    if gdf_uni.crs is None:
        gdf_uni.set_crs(4326, inplace=True)
    elif gdf_uni.crs.to_epsg() != 4326:
        gdf_uni = gdf_uni.to_crs(4326)

    merged = merge_frames(gdf_uni, gdf_cm)
    Path(args.merged_parquet).parent.mkdir(parents=True, exist_ok=True)
    safe_to_parquet(merged, Path(args.merged_parquet))
    try:
        merged.to_file(args.merged_gpkg, layer="unified_plus_cm", driver="GPKG")
        print(f"[OK] GPKG → {args.merged_gpkg}")
    except Exception as e:
        print(f"[warn] GPKG write failed: {e}")

    print("=== SUMMARY ===")
    print(f"TROPOMI unified rows: {len(gdf_uni)}")
    print(f"Carbon Mapper rows:   {len(gdf_cm)}")
    print(f"Merged total rows:    {len(merged)}")
    print("\nNext steps:")
    print(f"  1) Cross-provider dedup:")
    print(f"     python src/dedup.py --in-parquet {args.merged_parquet}")
    print("  2) Re-run clustering on the dedup result as usual.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] user cancelled")
        sys.exit(1)
