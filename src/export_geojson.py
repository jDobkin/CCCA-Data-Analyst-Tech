#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export processed data to GeoJSON for the webmap (includes `provider` for filtering)
and optionally appends Carbon Mapper points.

Examples:
  # Append Carbon Mapper and clip to AU:
  python src/export_geojson.py ^
    --points-in data/processed/au_plumes_clustered.parquet ^
    --provider-ref data/interim/unified_plus_cm.parquet ^
    --cm-in data/interim/cm_plumes_clean.geojson ^
    --clip-au
"""

from pathlib import Path
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

AUS_BBOX = (112.0, -44.0, 154.0, -9.0)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Export AU points + clusters GeoJSON for webmap (with provider), with optional Carbon Mapper append."
    )
    ap.add_argument(
        "--points-in",
        default="data/processed/au_plumes_clustered.parquet",
        help="Primary points source (Parquet/GPKG/GeoJSON).",
    )
    ap.add_argument("--points-layer", default=None, help="Layer for GPKG points.")

    ap.add_argument(
        "--cm-in",
        default=None,
        help="Optional Carbon Mapper points (GeoJSON/Parquet/GPKG). Will be appended with provider='CARBON_MAPPER'.",
    )
    ap.add_argument("--cm-layer", default=None, help="Layer for GPKG CM input.")

    ap.add_argument(
        "--clusters-in",
        default="data/processed/clusters_au.gpkg",
        help="Clusters polygons (GPKG/GeoJSON).",
    )
    ap.add_argument(
        "--clusters-layer", default="clusters_au", help="Layer for clusters GPKG."
    )

    ap.add_argument(
        "--provider-ref",
        default=None,
        help="Optional reference file containing `provider` (e.g., data/interim/unified_plus_cm.parquet). "
        "Will merge by `plume_id_src` if needed for the PRIMARY points.",
    )
    ap.add_argument(
        "--out-dir",
        default="webmap-app/public/data",
        help="Webmap public/data directory.",
    )
    ap.add_argument(
        "--points-out",
        default="au_plumes.geojson",
        help="Output filename for combined points.",
    )
    ap.add_argument(
        "--clusters-out",
        default="clusters_au.geojson",
        help="Output filename for clusters.",
    )
    ap.add_argument(
        "--clip-au", action="store_true", help="Clip to Australia bbox before export."
    )
    return ap.parse_args()


def load_any(path: Path, layer: str | None):
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def clip_to_aus_bbox(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = AUS_BBOX
    bbox = box(minx, miny, maxx, maxy)
    return gdf[gdf.intersects(bbox) | gdf.within(bbox)].copy()


def ensure_provider_from_ref(
    points: gpd.GeoDataFrame,
    provider_ref_path: str | None,
    ref_layer: str | None = None,
) -> gpd.GeoDataFrame:
    """Ensure PRIMARY points have a `provider`. If missing and a ref is provided,
    merge `provider` on `plume_id_src`."""
    have_provider = "provider" in points.columns and points["provider"].notna().any()
    if have_provider:
        points["provider"] = points["provider"].astype("string").str.upper()
        return points

    if not provider_ref_path:
        print(
            "[warn] primary points: `provider` missing and no --provider-ref provided."
        )
        points["provider"] = pd.NA
        return points

    refp = Path(provider_ref_path)
    if not refp.exists():
        print(
            f"[warn] --provider-ref not found: {refp}. Proceeding without provider on primary points."
        )
        points["provider"] = pd.NA
        return points

    ref = load_any(refp, ref_layer)
    if "plume_id_src" not in points.columns or "plume_id_src" not in ref.columns:
        print(
            "[warn] Cannot merge provider for primary points (need plume_id_src in both)."
        )
        points["provider"] = pd.NA
        return points

    ref_small = ref.loc[:, ["plume_id_src", "provider"]].copy()
    ref_small["provider"] = ref_small["provider"].astype("string").str.upper()
    out = points.merge(ref_small, on="plume_id_src", how="left", suffixes=("", "_ref"))
    out["provider"] = out["provider"].fillna(out["provider_ref"])
    out.drop(columns=[c for c in out.columns if c.endswith("_ref")], inplace=True)
    return out


def load_primary_points(
    p_in: Path, layer: str | None, provider_ref: str | None, clip_au: bool
) -> gpd.GeoDataFrame:
    pts = load_any(p_in, layer)
    pts = pts[pts.geometry.notna() & pts.geometry.geom_type.isin(["Point"])].copy()
    if "plume_id_src" not in pts.columns:
        pts["plume_id_src"] = pts.get(
            "plume_id_src", pd.Series(pts.index, index=pts.index).astype(str)
        )
    for c in [
        "provider",
        "emission_tph",
        "emission_unc_tph",
        "obs_datetime_utc",
        "sector",
        "plume_id_src",
    ]:
        if c not in pts.columns:
            pts[c] = pd.NA
    pts = ensure_provider_from_ref(pts, provider_ref)
    if clip_au:
        pts = clip_to_aus_bbox(pts)
    pts["provider"] = pts["provider"].astype("string").str.upper()
    return pts


def load_carbon_mapper(
    cm_path: Path, layer: str | None, clip_au: bool
) -> gpd.GeoDataFrame:
    cm = load_any(cm_path, layer)
    cm = cm[cm.geometry.notna() & cm.geometry.geom_type.isin(["Point"])].copy()

    # Ensure required fields
    for c in [
        "plume_id_src",
        "emission_tph",
        "emission_unc_tph",
        "obs_datetime_utc",
        "sector",
    ]:
        if c not in cm.columns:
            cm[c] = pd.NA

    # Assign provider and stable id if needed
    cm["provider"] = "CARBON_MAPPER"
    if "plume_id_src" not in cm.columns or cm["plume_id_src"].isna().all():
        # create a stable-ish id
        cm["plume_id_src"] = cm.index.astype(str).radd("CM_")

    if clip_au:
        cm = clip_to_aus_bbox(cm)

    # Normalize types
    cm["provider"] = cm["provider"].astype("string").str.upper()
    return cm


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PRIMARY points (SRON/KAYRROS etc.)
    primary = load_primary_points(
        Path(args.points_in), args.points_layer, args.provider_ref, args.clip_au
    )

    # CARBON MAPPER points (optional append)
    if args.cm_in:
        cm_path = Path(args.cm_in)
        cm = load_carbon_mapper(cm_path, args.cm_layer, args.clip_au)
        # Align columns before concat
        cols = sorted(
            set(primary.columns) | set(cm.columns) | {"provider", "plume_id_src"}
        )
        primary = primary.reindex(columns=cols)
        cm = cm.reindex(columns=cols)
        combined = pd.concat([primary, cm], ignore_index=True)
    else:
        combined = primary

    # Write combined POINTS
    points_path = out_dir / args.points_out
    gpd.GeoDataFrame(combined, geometry="geometry", crs=4326).to_file(
        points_path, driver="GeoJSON"
    )
    print(
        f"[OK] points → {points_path} (rows={len(combined)}) | providers=\n{combined['provider'].astype(str).str.upper().value_counts(dropna=False)}"
    )

    # Write CLUSTERS if available
    cl_path = Path(args.clusters_in)
    if cl_path.exists():
        cl = load_any(cl_path, args.clusters_layer)
        if args.clip_au:
            cl = clip_to_aus_bbox(cl)
        clusters_path = out_dir / args.clusters_out
        cl.to_file(clusters_path, driver="GeoJSON")
        print(f"[OK] clusters → {clusters_path} (rows={len(cl)})")
    else:
        print(f"[info] clusters input not found: {cl_path} (skipping)")


if __name__ == "__main__":
    main()
