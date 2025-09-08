#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean & unify SRON and Kayrros plume CSVs into a canonical schema.
- Expands Windows wildcards (e.g., *.csv)
- Clips to an Australia bbox if requested
- Writes Parquet and GeoPackage outputs
"""

import argparse, glob, hashlib, math, sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---- Australia bbox in EPSG:4326 (lon_min, lat_min, lon_max, lat_max)
AU_BBOX = (112.9, -43.9, 153.7, -9.0)


def _hash_id(*parts: str) -> str:
    s = "|".join(parts)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _clean_coords(lat: float, lon: float) -> bool:
    try:
        lat = float(lat)
        lon = float(lon)
        return (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        )
    except Exception:
        return False


def _in_bbox(lat: float, lon: float, bbox) -> bool:
    minx, miny, maxx, maxy = bbox
    return (minx <= lon <= maxx) and (miny <= lat <= maxy)


# -------- SRON --------
def read_sron_csv(path: Path) -> pd.DataFrame:
    # Expected columns: date (YYYYMMDD), time_UTC (HH:MM:SS), lat, lon, source_rate_t/h, uncertainty_t/h
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # make sure date is 8 chars "YYYYMMDD"
    df["date"] = df["date"].astype(str).str.zfill(8)
    dt = pd.to_datetime(
        df["date"] + " " + df["time_UTC"],
        format="%Y%m%d %H:%M:%S",
        utc=True,
        errors="coerce",
    )

    out = pd.DataFrame(
        {
            "provider": "SRON",
            "obs_datetime_utc": dt,
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "lon": pd.to_numeric(df["lon"], errors="coerce"),
            "emission_tph": pd.to_numeric(df.get("source_rate_t/h"), errors="coerce"),
            "emission_unc_tph": pd.to_numeric(
                df.get("uncertainty_t/h"), errors="coerce"
            ),
            "sector": pd.Series([pd.NA] * len(df), dtype="object"),
            "country": pd.Series([pd.NA] * len(df), dtype="object"),
        }
    )

    # filter bad rows
    out = out[
        out.apply(
            lambda r: pd.notna(r["obs_datetime_utc"])
            and _clean_coords(r["lat"], r["lon"]),
            axis=1,
        )
    ].copy()
    out.loc[out["emission_tph"] < 0, "emission_tph"] = pd.NA
    out.loc[out["emission_unc_tph"] < 0, "emission_unc_tph"] = pd.NA

    lat_r = out["lat"].round(3).astype(str)
    lon_r = out["lon"].round(3).astype(str)
    date_str = out["obs_datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["plume_id_src"] = [
        _hash_id("SRON", d, la, lo) for d, la, lo in zip(date_str, lat_r, lon_r)
    ]

    return out


# -------- Kayrros --------
def read_kayrros_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in col:
                return col[c]
        raise KeyError(f"Missing expected column among: {cands}")

    date_col = pick("date")
    lat_col = pick("source latitude rough", "latitude", "lat")
    lon_col = pick("source longitude rough", "longitude", "lon")
    rate_col = pick(
        "emission rate (tons/hour)",
        "emission rate (tons per hour)",
        "emission_rate_tph",
    )
    unc_col = pick(
        "uncertainty (tons/hour)", "uncertainty (tons per hour)", "uncertainty_tph"
    )
    cat_col = pick("category", "sector")
    country_col = col.get("country")

    dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    dt = dt.dt.tz_convert("UTC") if dt.dt.tz is not None else dt.dt.tz_localize("UTC")

    out = pd.DataFrame(
        {
            "provider": "KAYRROS",
            "obs_datetime_utc": dt,
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "emission_tph": pd.to_numeric(df[rate_col], errors="coerce"),
            "emission_unc_tph": pd.to_numeric(df[unc_col], errors="coerce"),
            "sector": df[cat_col].astype("string").str.strip().str.lower(),
            "country": (
                df[country_col].astype("string").str.strip()
                if country_col
                else pd.Series([pd.NA] * len(df), dtype="object")
            ),
        }
    )

    out = out[
        out.apply(
            lambda r: pd.notna(r["obs_datetime_utc"])
            and _clean_coords(r["lat"], r["lon"]),
            axis=1,
        )
    ].copy()
    out.loc[out["emission_tph"] < 0, "emission_tph"] = pd.NA
    out.loc[out["emission_unc_tph"] < 0, "emission_unc_tph"] = pd.NA

    lat_r = out["lat"].round(3).astype(str)
    lon_r = out["lon"].round(3).astype(str)
    date_str = out["obs_datetime_utc"].dt.strftime("%Y-%m-%dT00:00:00Z")
    out["plume_id_src"] = [
        _hash_id("KAYRROS", d, la, lo) for d, la, lo in zip(date_str, lat_r, lon_r)
    ]

    return out


# -------- unify --------
def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[
            Point(xy) if (pd.notna(xy[0]) and pd.notna(xy[1])) else None
            for xy in zip(df["lon"], df["lat"])
        ],
        crs="EPSG:4326",
    )
    return gdf.dropna(subset=["geometry"])


def clean_and_unify(
    sron_files: Iterable[Path], kayrros_files: Iterable[Path], clip_australia: bool
) -> gpd.GeoDataFrame:
    frames: List[pd.DataFrame] = []
    for p in sron_files:
        frames.append(read_sron_csv(Path(p)))
    for p in kayrros_files:
        frames.append(read_kayrros_csv(Path(p)))

    if not frames:
        raise RuntimeError("No input files found. Check your --sron / --kayrros paths.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df.apply(lambda r: _clean_coords(r["lat"], r["lon"]), axis=1)].copy()

    if clip_australia:
        df = df[
            df.apply(lambda r: _in_bbox(r["lat"], r["lon"], AU_BBOX), axis=1)
        ].copy()

    df = df.sort_values(["obs_datetime_utc", "provider"]).reset_index(drop=True)
    gdf = to_geodataframe(df)

    print("=== QA summary ===")
    print("rows:", len(gdf))
    print("providers:", gdf["provider"].value_counts(dropna=False).to_dict())
    if len(gdf):
        print(
            "time range:",
            gdf["obs_datetime_utc"].min(),
            "→",
            gdf["obs_datetime_utc"].max(),
        )

    return gdf


def main():
    ap = argparse.ArgumentParser(
        description="Clean & unify SRON and Kayrros plume CSVs."
    )
    ap.add_argument(
        "--sron", nargs="*", default=[], help="SRON CSV file(s) or patterns"
    )
    ap.add_argument(
        "--kayrros", nargs="*", default=[], help="Kayrros CSV file(s) or patterns"
    )
    ap.add_argument(
        "--out-parquet",
        default="data/interim/plumes_unified.parquet",
        help="Output Parquet path",
    )
    ap.add_argument(
        "--out-gpkg",
        default="data/interim/plumes_unified.gpkg",
        help="Output GeoPackage path",
    )
    ap.add_argument("--layer", default="plumes_unified", help="GPKG layer name")
    ap.add_argument("--clip-au", action="store_true", help="Clip to Australia bbox")
    args = ap.parse_args()

    # Expand wildcards (works on Windows too)
    sron_files = [Path(p) for pat in args.sron for p in glob.glob(str(pat))]
    kayrros_files = [Path(p) for pat in args.kayrros for p in glob.glob(str(pat))]
    print("sron_files:", [str(p) for p in sron_files])
    print("kayrros_files:", [str(p) for p in kayrros_files])

    gdf = clean_and_unify(sron_files, kayrros_files, clip_australia=args.clip_au)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_gpkg = Path(args.out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_parquet(out_parquet, index=False)
    gdf.to_file(out_gpkg, layer=args.layer, driver="GPKG")

    print(f"[OK] Wrote {len(gdf)} rows → {out_parquet}")
    print(f"[OK] Wrote GPKG layer '{args.layer}' → {out_gpkg}")


if __name__ == "__main__":
    try:
        print(">>> clean_and_unify.py starting")
        print("argv:", sys.argv)
        main()
    except Exception as e:
        import traceback

        print("!! ERROR:", e)
        traceback.print_exc()
