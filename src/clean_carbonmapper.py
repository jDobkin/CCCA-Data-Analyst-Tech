#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

KM = 1000.0


def parse_args():
    ap = argparse.ArgumentParser(
        description="Clean/validate Carbon Mapper plumes JSONL → unified schema"
    )
    ap.add_argument(
        "--in-jsonl", default="data/raw/carbonmapper/plumes_annotated.jsonl"
    )
    ap.add_argument("--out-geojson", default="data/interim/cm_plumes_clean.geojson")
    ap.add_argument("--out-parquet", default="data/interim/cm_plumes_clean.parquet")
    ap.add_argument("--clip-au", action="store_true", help="Clip to Australia bbox")
    return ap.parse_args()


def au_bbox() -> Tuple[float, float, float, float]:
    return (112.0, -44.0, 154.0, -9.0)


def in_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bbox
    return minx <= lon <= maxx and miny <= lat <= maxy


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    items = []
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


def main():
    args = parse_args()
    inp = Path(args.in_jsonl)
    if not inp.exists():
        raise FileNotFoundError(inp)

    raw = load_jsonl(inp)
    if not raw:
        print("[warn] no records in JSONL")
        return

    rows = []
    for it in raw:
        geom = it.get("geometry_json") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        lon, lat = (None, None)
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            lon, lat = float(coords[0]), float(coords[1])

        # Source fields
        plume_id = it.get("plume_id") or it.get("id")
        scene_ts = it.get("scene_timestamp")
        gas = it.get("gas")
        emission = it.get("emission_auto")  # kg/hr per example
        emission_unc = it.get("emission_uncertainty_auto")
        sector = it.get("sector")
        instrument = it.get("instrument")
        platform = it.get("platform")

        # Schema
        rows.append(
            {
                "provider": "CARBON_MAPPER",
                "plume_id_src": plume_id,
                "gas": gas,
                "obs_datetime_utc": scene_ts,
                "lon": lon,
                "lat": lat,
                # convert kg/hr → t/h if present
                "emission_tph": (
                    (float(emission) / 1000.0) if emission is not None else None
                ),
                "emission_unc_tph": (
                    (float(emission_unc) / 1000.0) if emission_unc is not None else None
                ),
                "sector": sector,
                "instrument": instrument,
                "platform": platform,
                # keep raw for traceability
                "_raw": it,
            }
        )

    df = pd.DataFrame(rows)

    # Validation
    # - datetime parse
    df["obs_datetime_utc"] = pd.to_datetime(
        df["obs_datetime_utc"], utc=True, errors="coerce"
    )
    # - coord sanity
    df = df[
        (df["lon"].between(-180, 180, inclusive="both"))
        & (df["lat"].between(-90, 90, inclusive="both"))
    ]
    # - non-negative emissions
    for c in ["emission_tph", "emission_unc_tph"]:
        if c in df.columns:
            df.loc[df[c].notna() & (df[c] < 0), c] = None

    # AU clip (fast bbox clip)
    if args.clip_au:
        bbox = au_bbox()
        df = df[
            df.apply(
                lambda r: (
                    in_bbox(float(r["lon"]), float(r["lat"]), bbox)
                    if pd.notna(r["lon"])
                    else False
                ),
                axis=1,
            )
        ]

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[
            Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None
            for xy in zip(df["lon"], df["lat"])
        ],
        crs="EPSG:4326",
    ).dropna(subset=["geometry"])

    # Light dedup within Carbon Mapper itself (30 min 5 km)
    # Project to meters for distance
    gdf_m = gdf.to_crs(3857)
    gdf_m["dt"] = gdf_m["obs_datetime_utc"]
    gdf_m = gdf_m.sort_values("dt").reset_index(drop=True)

    # simple time-window + nearest trick
    to_drop = set()
    max_dt = pd.Timedelta(minutes=30)
    max_dist_m = 5 * 1000.0

    for i in range(len(gdf_m)):
        if i in to_drop:
            continue
        ti = gdf_m.at[i, "dt"]
        gi = gdf_m.at[i, "geometry"]
        if pd.isna(ti) or gi is None:
            continue
        # window neighbors within 30 min
        lo = ti - max_dt
        hi = ti + max_dt
        win = gdf_m[(gdf_m["dt"] >= lo) & (gdf_m["dt"] <= hi)]
        if len(win) <= 1:
            continue
        # distance filter
        dists = win.geometry.distance(gi)
        close = win[(dists <= max_dist_m)]
        # keep the first; drop others
        if len(close) > 1:
            idxs = close.index.tolist()
            for j in idxs[1:]:
                to_drop.add(j)

    gdf_clean = gdf_m.drop(index=list(to_drop)).copy()
    gdf_clean = gdf_clean.drop(columns=["dt"])

    out_gj = Path(args.out_geojson)
    out_gj.parent.mkdir(parents=True, exist_ok=True)
    out_pq = Path(args.out_parquet)
    out_pq.parent.mkdir(parents=True, exist_ok=True)

    gdf_clean.to_file(out_gj, driver="GeoJSON")
    try:
        gdf_clean.to_parquet(out_pq, index=False)
    except Exception as e:
        print(f"[warn] parquet write failed (install pyarrow): {e}")

    print("=== Carbon Mapper Clean ===")
    print(f"input rows:  {len(df)}")
    print(f"kept rows:   {len(gdf_clean)}  (dedup dropped {len(to_drop)})")
    print(f"[OK] GeoJSON → {out_gj}")
    if out_pq.exists():
        print(f"[OK] Parquet → {out_pq}")


if __name__ == "__main__":
    main()
