#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088


def to_radians(lat, lon):
    return np.radians(np.c_[lat, lon])


def same_day_window(t1: pd.Timestamp, t2: pd.Timestamp, days: int) -> bool:
    if pd.isna(t1) or pd.isna(t2):
        return False
    d = abs((t2.normalize() - t1.normalize()).days)
    return d <= days


def score_row(unc, em):
    # Lower uncertainty, non-null emission beats null.
    # (False, small, False) is best.
    unc_is_null = pd.isna(unc)
    em_is_null = pd.isna(em)
    unc_val = float("inf") if unc_is_null else float(unc)
    return (unc_is_null, unc_val, em_is_null)


def main():
    ap = argparse.ArgumentParser(
        description="Cross-provider dedup within distance & ±day window."
    )
    ap.add_argument(
        "--in-parquet",
        default="data/interim/merged.parquet",
        help="Input merged GeoParquet",
    )
    ap.add_argument("--in-gpkg", default=None, help="Alternative input GPKG")
    ap.add_argument("--layer", default="merged", help="GPKG layer name")
    ap.add_argument(
        "--km", type=float, default=30.0, help="Max great-circle distance (km)"
    )
    ap.add_argument("--days", type=int, default=1, help="± day window for 'same day'")
    ap.add_argument(
        "--prefer",
        choices=["KAYRROS", "SRON"],
        default="KAYRROS",
        help="Tie-break provider preference",
    )
    ap.add_argument(
        "--keep-both",
        action="store_true",
        help="Keep both duplicates, just link via crossref_id",
    )
    ap.add_argument(
        "--out-parquet",
        default="data/processed/plumes_dedup.parquet",
        help="Output GeoParquet",
    )
    ap.add_argument(
        "--out-gpkg", default="data/processed/plumes_dedup.gpkg", help="Output GPKG"
    )
    ap.add_argument(
        "--pairs-csv",
        default="data/processed/crossref_pairs.csv",
        help="Pairs CSV (decisions)",
    )
    args = ap.parse_args()

    # Read
    in_p = Path(args.in_parquet)
    if in_p.exists():
        gdf = gpd.read_parquet(in_p)
    elif args.in_gpkg:
        gdf = gpd.read_file(args.in_gpkg, layer=args.layer)
    else:
        raise FileNotFoundError(f"Input not found: {in_p} (or provide --in-gpkg)")

    gdf["obs_datetime_utc"] = pd.to_datetime(
        gdf["obs_datetime_utc"], utc=True, errors="coerce"
    )
    gdf = gdf.dropna(subset=["lat", "lon"]).copy()

    # Split providers
    a = gdf[gdf["provider"].str.upper() == "SRON"].copy()
    b = gdf[gdf["provider"].str.upper() == "KAYRROS"].copy()
    if a.empty or b.empty:
        print("[info] Need both SRON and KAYRROS present; nothing to cross-dedup.")
        out = gdf
        Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)
        try:
            out.to_parquet(args.out_parquet, index=False)
        except Exception as e:
            print(f"[warn] Parquet write failed: {e}")
        out.to_file(args.out_gpkg, layer="plumes_dedup", driver="GPKG")
        return

    # BallTree on radians with haversine (radius in radians)
    rad_b = to_radians(b["lat"].to_numpy(), b["lon"].to_numpy())
    tree = BallTree(rad_b, metric="haversine")
    rad_a = to_radians(a["lat"].to_numpy(), a["lon"].to_numpy())
    r_rad = args.km / EARTH_KM

    # Query neighbors within radius
    ind = tree.query_radius(rad_a, r=r_rad)

    pairs = []
    for i_a, neigh_idx in enumerate(ind):
        if len(neigh_idx) == 0:
            continue
        t_a = a.iloc[i_a]["obs_datetime_utc"]
        for j in neigh_idx:
            t_b = b.iloc[j]["obs_datetime_utc"]
            if not same_day_window(t_a, t_b, args.days):
                continue
            pairs.append((a.index[i_a], b.index[j]))

    if not pairs:
        print("[info] No cross-provider duplicates under thresholds; writing input.")
        out = gdf
        Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)
        try:
            out.to_parquet(args.out_parquet, index=False)
        except Exception as e:
            print(f"[warn] Parquet write failed: {e}")
        out.to_file(args.out_gpkg, layer="plumes_dedup", driver="GPKG")
        return

    # Decide winners in each pair (one-to-one best).
    # Group by SRON record and pick the best KAYRROS match (and vice-versa)
    # but final drops are done on the pairwise decisions.
    records = []
    for ia, ib in pairs:
        ra, rb = gdf.loc[ia], gdf.loc[ib]
        # Score both; lower tuple = better
        sa = score_row(ra["emission_unc_tph"], ra["emission_tph"])
        sb = score_row(rb["emission_unc_tph"], rb["emission_tph"])
        if sa < sb:
            keep = ia
            drop = ib
        elif sb < sa:
            keep = ib
            drop = ia
        else:
            # tiepip  prefer provider
            pref = args.prefer.upper()
            keep = ib if rb["provider"].upper() == pref else ia
            drop = ia if keep == ib else ib

        # assign a crossref_id for this matched pair
        crossref_id = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            "|".join(sorted([str(ra["plume_id_src"]), str(rb["plume_id_src"])])),
        )
        records.append(
            {
                "sron_index": ia if ra["provider"].upper() == "SRON" else ib,
                "kay_index": ib if rb["provider"].upper() == "KAYRROS" else ia,
                "keep_index": keep,
                "drop_index": drop,
                "crossref_id": str(crossref_id),
                "dist_km_approx": args.km,  # approx (we didn’t save exact; optional to compute)
                "dt_days_max": args.days,
            }
        )

    pairs_df = pd.DataFrame.from_records(records).drop_duplicates(
        subset=["sron_index", "kay_index"]
    )
    Path(args.pairs_csv).parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(args.pairs_csv, index=False)

    # Tag crossref_id on kept & dropped rows
    gdf["crossref_id"] = pd.NA
    for _, r in pairs_df.iterrows():
        gdf.at[r["sron_index"], "crossref_id"] = r["crossref_id"]
        gdf.at[r["kay_index"], "crossref_id"] = r["crossref_id"]

    if args.keep_both:
        dedup = gdf.copy()
    else:
        drop_ids = set(pairs_df["drop_index"].tolist())
        dedup = gdf.drop(index=drop_ids).copy()

    # Write outputs
    Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)
    try:
        dedup.to_parquet(args.out_parquet, index=False)
    except Exception as e:
        print(f"[warn] Parquet write failed: {e}")
    dedup.to_file(args.out_gpkg, layer="plumes_dedup", driver="GPKG")

    print(
        "=== CROSS-PROVIDER DEDUP (±{d} day, ≤{k} km) ===".format(
            d=args.days, k=args.km
        )
    )
    print(f"Pairs found:     {len(pairs_df)}")
    print(f"Rows removed:    {len(gdf)-len(dedup)}  (use --keep-both to retain)")
    print(f"Rows remaining:  {len(dedup)}")
    print(f"[OK] Pairs CSV  → {args.pairs_csv}")
    print(f"[OK] GPKG       → {args.out_gpkg}")
    if Path(args.out_parquet).exists():
        print(f"[OK] Parquet    → {args.out_parquet}")


if __name__ == "__main__":
    main()
