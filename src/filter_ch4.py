#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python src\filter_ch4.py --in-parquet "data\interim\unified_plus_cm.parquet" --out-parquet "data\interim\unified_plus_cm_ch4.parquet" --include-unknown

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd


def main():
    ap = argparse.ArgumentParser(
        description="Filter unified_plus_cm to CH4 (and optionally unknown gas)."
    )
    ap.add_argument("--in-parquet", default="data/interim/unified_plus_cm.parquet")
    ap.add_argument("--out-parquet", default="data/interim/unified_plus_cm_ch4.parquet")
    ap.add_argument(
        "--include-unknown",
        action="store_true",
        help="Also keep rows where gas is null/empty (common for SRON/Kayrros).",
    )
    args = ap.parse_args()

    inp = Path(args.in_parquet)
    if not inp.exists():
        raise FileNotFoundError(inp)

    gdf = gpd.read_parquet(inp)
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)

    # Normalize gas to string for filtering
    if "gas" not in gdf.columns:
        gdf["gas"] = pd.NA
    gas_str = gdf["gas"].astype("string").str.upper()

    mask_ch4 = gas_str.eq("CH4")
    if args.include_unknown:
        mask_unknown = gas_str.isna() | gas_str.eq("") | gas_str.eq("NAN")
        mask = mask_ch4 | mask_unknown
    else:
        mask = mask_ch4

    out = gdf[mask].copy()
    out.to_parquet(args.out_parquet, index=False)
    print(
        f"[OK] Filtered → {args.out_parquet} (rows: {len(out)})  | include_unknown={args.include_unknown}"
    )


if __name__ == "__main__":
    main()
