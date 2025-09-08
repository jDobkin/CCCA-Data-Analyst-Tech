#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, pickle
from pathlib import Path
from typing import List
import pandas as pd
import geopandas as gpd

CANON_COLS = [
    "provider",
    "obs_datetime_utc",
    "lat",
    "lon",
    "emission_tph",
    "emission_unc_tph",
    "sector",
    "country",
    "plume_id_src",
    "geometry",
]


def read_any(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        gdf = gpd.read_parquet(p)
    elif ext in (".gpkg", ".gdb", ".geojson", ".json", ".shp"):
        kwargs = {"layer": layer} if (layer and ext == ".gpkg") else {}
        gdf = gpd.read_file(p, **kwargs)
    else:
        # last resort: try csv with lon/lat
        df = pd.read_csv(p)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        )
    # ensure canon columns exist (fill missing)
    for c in CANON_COLS:
        if c not in gdf.columns and c != "geometry":
            gdf[c] = pd.NA
    gdf = gdf[[c for c in CANON_COLS if c != "geometry"] + ["geometry"]]
    gdf["obs_datetime_utc"] = pd.to_datetime(
        gdf["obs_datetime_utc"], utc=True, errors="coerce"
    )
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def build_spatial_index_pickle(gdf: gpd.GeoDataFrame, out_pkl: Path) -> None:
    """Persist a shapely STRtree so later code can reload quickly."""
    from shapely.strtree import STRtree

    geoms = gdf.geometry.values
    tree = STRtree(geoms)
    # map geometry object id -> row index for quick reverse lookup
    geom_to_idx = {id(geom): i for i, geom in enumerate(geoms)}
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump({"tree": tree, "geom_to_idx": geom_to_idx}, f)


def main():
    ap = argparse.ArgumentParser(
        description="Merge cleaned plume files into one GeoDataFrame."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of cleaned files (parquet/gpkg/geojson/csv)",
    )
    ap.add_argument("--layer", default=None, help="Layer name (for GeoPackage inputs)")
    ap.add_argument(
        "--out-parquet", default="data/interim/merged.parquet", help="Output GeoParquet"
    )
    ap.add_argument(
        "--out-gpkg", default="data/interim/merged.gpkg", help="Output GeoPackage"
    )
    ap.add_argument(
        "--sindex-pkl",
        default="data/interim/merged_sindex.pkl",
        help="Serialized spatial index",
    )
    args = ap.parse_args()

    parts: List[gpd.GeoDataFrame] = []
    for p in args.inputs:
        gdf = read_any(Path(p), layer=args.layer)
        parts.append(gdf)

    if not parts:
        raise RuntimeError("No inputs provided/readable.")

    merged = pd.concat(parts, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged = merged.dropna(subset=["geometry"]).reset_index(drop=True)

    # Ensure key fields present
    merged = merged.sort_values(
        ["obs_datetime_utc", "provider", "plume_id_src"], na_position="last"
    ).reset_index(drop=True)

    # Write outputs
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(args.out_parquet, index=False)
    merged.to_file(args.out_gpkg, layer="merged", driver="GPKG")

    # Build & persist a spatial index
    build_spatial_index_pickle(merged, Path(args.sindex_pkl))

    print(f"[OK] Merged rows: {len(merged)}")
    print(f"[OK] GeoParquet → {args.out_parquet}")
    print(f"[OK] GeoPackage → {args.out_gpkg} (layer=merged)")
    print(f"[OK] STRtree    → {args.sindex_pkl}")


if __name__ == "__main__":
    main()
