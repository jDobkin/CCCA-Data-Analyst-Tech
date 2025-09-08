#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd

KM = 1000.0
MAX_DIST_M = 5 * KM
MAX_DT_MIN = 30


def choose_best(row):
    """
    row has columns from both sides with suffixes _left (SRON) and _right (KAYRROS).
    Return 'left' or 'right' for which to keep, using:
      1) lower emission_unc_tph (non-null wins)
      2) then non-null emission_tph
      3) then provider preference: KAYRROS over SRON
    """
    ul = row["emission_unc_tph_left"]
    ur = row["emission_unc_tph_right"]
    el = row["emission_tph_left"]
    er = row["emission_tph_right"]

    # 1) uncertainty (non-null beats null; lower beats higher)
    if pd.notna(ul) or pd.notna(ur):
        if pd.isna(ul) and pd.notna(ur):
            return "right"
        if pd.isna(ur) and pd.notna(ul):
            return "left"
        if pd.notna(ul) and pd.notna(ur):
            if ur < ul:
                return "right"
            if ul < ur:
                return "left"
    # 2) emission present?
    if pd.notna(er) and pd.isna(el):
        return "right"
    if pd.notna(el) and pd.isna(er):
        return "left"
    # 3) provider preference (adjust if you prefer SRON)
    return "right"  # prefer Kayrros in ties


def main():
    ap = argparse.ArgumentParser(
        description="Cross-provider dedup within 5 km and ±30 min (SRON vs KAYRROS)."
    )
    ap.add_argument(
        "--in-parquet",
        default="data/interim/plumes_unified.parquet",
        help="Unified Parquet",
    )
    ap.add_argument("--in-gpkg", default=None, help="Alternative: input GeoPackage")
    ap.add_argument(
        "--layer", default="plumes_unified", help="Layer name when reading GPKG"
    )
    ap.add_argument(
        "--out-parquet",
        default="data/processed/plumes_crossdedup.parquet",
        help="Output Parquet",
    )
    ap.add_argument(
        "--out-gpkg",
        default="data/processed/plumes_crossdedup.gpkg",
        help="Output GeoPackage",
    )
    ap.add_argument(
        "--pairs-csv",
        default="data/processed/crossdup_pairs.csv",
        help="CSV of matched pairs & decisions",
    )
    ap.add_argument(
        "--epsg-meters",
        type=int,
        default=3857,
        help="Projected CRS in meters for distance calc",
    )
    ap.add_argument(
        "--max-dist-m",
        type=float,
        default=MAX_DIST_M,
        help="Distance threshold (meters)",
    )
    ap.add_argument(
        "--max-dt-min", type=int, default=MAX_DT_MIN, help="Time threshold (minutes)"
    )
    args = ap.parse_args()

    # ---- Read inputs
    in_path = Path(args.in_parquet)
    if in_path.exists():
        gdf = gpd.read_parquet(in_path)
    elif args.in_gpkg:
        gdf = gpd.read_file(args.in_gpkg, layer=args.layer)
    else:
        raise FileNotFoundError(f"Input not found: {in_path} (or provide --in-gpkg)")

    gdf["obs_datetime_utc"] = pd.to_datetime(
        gdf["obs_datetime_utc"], utc=True, errors="coerce"
    )

    # ---- Split providers
    sron = gdf[gdf["provider"].str.upper() == "SRON"].copy()
    kay = gdf[gdf["provider"].str.upper() == "KAYRROS"].copy()
    if sron.empty or kay.empty:
        print(
            "[info] One provider missing; nothing to cross-dedup. Writing input to outputs."
        )
        out_gdf = gdf
        Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)
        try:
            out_gdf.to_parquet(args.out_parquet, index=False)
        except Exception as e:
            print(f"[warn] Parquet write failed: {e}")
        out_gdf.to_file(args.out_gpkg, layer="plumes_crossdedup", driver="GPKG")
        return

    # Project to meter CRS for spatial join
    sron_m = sron.to_crs(epsg=args.epsg_meters)
    kay_m = kay.to_crs(epsg=args.epsg_meters)

    # Spatial nearest join (within max distance)
    cand = gpd.sjoin_nearest(
        sron_m,
        kay_m[
            [
                "plume_id_src",
                "obs_datetime_utc",
                "emission_tph",
                "emission_unc_tph",
                "geometry",
            ]
        ],
        how="inner",
        max_distance=args.max_dist_m,
        distance_col="dist_m",
    )

    # Time filter + window
    dt = (cand["obs_datetime_utc_right"] - cand["obs_datetime_utc_left"]).abs()
    cand = cand[dt <= pd.Timedelta(minutes=args.max_dt_min)].copy()

    if cand.empty:
        print(
            "[info] No cross-provider duplicates found under thresholds. Writing input to outputs."
        )
        out_gdf = gdf
        Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)
        try:
            out_gdf.to_parquet(args.out_parquet, index=False)
        except Exception as e:
            print(f"[warn] Parquet write failed: {e}")
        out_gdf.to_file(args.out_gpkg, layer="plumes_crossdedup", driver="GPKG")
        return

    # Decide which to keep for each pair
    # Keep only the best Kayrros match per SRON candidate (sjoin_nearest already gives nearest; grouping helps if ties)
    # Prepare columns for decision
    cols_keep = [
        "plume_id_src_left",
        "obs_datetime_utc_left",
        "emission_tph_left",
        "emission_unc_tph_left",
        "plume_id_src_right",
        "obs_datetime_utc_right",
        "emission_tph_right",
        "emission_unc_tph_right",
        "dist_m",
    ]
    cand = cand[cols_keep].copy()
    cand["decision"] = cand.apply(choose_best, axis=1)

    # Save a CSV of pairs & decisions
    Path(args.pairs_csv).parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(args.pairs_csv, index=False)

    # Build sets of ids to drop
    drop_ids = set()
    for _, r in cand.iterrows():
        if r["decision"] == "left":
            # keep SRON, drop Kayrros
            drop_ids.add(r["plume_id_src_right"])
        else:
            # keep Kayrros, drop SRON
            drop_ids.add(r["plume_id_src_left"])

    dedup = gdf[~gdf["plume_id_src"].isin(drop_ids)].copy()

    # Write outputs
    Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)
    try:
        dedup.to_parquet(args.out_parquet, index=False)
    except Exception as e:
        print(f"[warn] Parquet write failed: {e}")
    dedup.to_file(args.out_gpkg, layer="plumes_crossdedup", driver="GPKG")

    print("=== CROSS-PROVIDER DEDUP ===")
    print(f"Pairs evaluated: {len(cand)}")
    print(f"Records removed: {len(drop_ids)}")
    print(f"Rows remaining:  {len(dedup)}")
    print(f"[OK] Pairs CSV  → {args.pairs_csv}")
    print(f"[OK] GPKG       → {args.out_gpkg}")
    if Path(args.out_parquet).exists():
        print(f"[OK] Parquet    → {args.out_parquet}")


if __name__ == "__main__":
    main()
