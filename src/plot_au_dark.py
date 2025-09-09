#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dark-themed Australia methane plume plotter (provider-colored)
- Points colored by provider (SRON / KAYRROS / CARBON_MAPPER / other)
- Optional clusters rendered UNDER points (fill + outline)
- Optional: force Carbon Mapper points to draw last (on top) and size differently
- Optional: export a web-ready GeoJSON

Examples:
    # Points + clusters, Carbon Mapper on top (slightly smaller)
    python src/plot_au_dark.py ^
      --in "data/processed/plumes_dedup.parquet" ^
      --clusters-in "data/processed/clusters_au.gpkg" ^
      --clusters-layer "clusters_au" ^
      --clip-au ^
      --cm-on-top --cm-size-scale 0.8 ^
      --size-scale 0.35 --min-size 3 --max-size 90 ^
      --out-png "outputs/maps/au_provider_with_CM_on_top.png"
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely import wkb, wkt

try:
    from shapely import set_precision as _shp_set_precision  # shapely >= 2.0
except Exception:
    _shp_set_precision = None

# Optional local dataset provider
try:
    from geodatasets import get_path as _gd_get_path  # type: ignore
except Exception:
    _gd_get_path = None


# Australia polygon loader
# -------------------------
def load_aus_polygon() -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame [EPSG:4326] for Australia."""
    # 1) geodatasets (if installed)
    if _gd_get_path is not None:
        for key in [
            "naturalearth.cultural.vectors-admin_0_countries",
            "naturalearth_lowres",
        ]:
            try:
                world = gpd.read_file(_gd_get_path(key)).to_crs(4326)
                name_col = next(
                    (
                        c
                        for c in ("ADMIN", "name", "NAME", "SOVEREIGNT")
                        if c in world.columns
                    ),
                    None,
                )
                if name_col is not None:
                    aus = world.loc[
                        world[name_col]
                        .astype(str)
                        .str.contains("Australia", case=False, na=False)
                    ]
                    if len(aus):
                        return gpd.GeoDataFrame(geometry=[aus.unary_union], crs=4326)
            except Exception:
                pass
    # 2) Natural Earth via HTTPS
    try:
        ne_zip = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(ne_zip).to_crs(4326)
        if "ADMIN" in world.columns:
            aus = world.loc[world["ADMIN"] == "Australia"]
            if len(aus):
                return gpd.GeoDataFrame(geometry=[aus.unary_union], crs=4326)
    except Exception:
        pass
    # 3) Fallback: AU bbox
    bbox_geom = box(112.0, -44.0, 154.0, -10.0)
    return gpd.GeoDataFrame(geometry=[bbox_geom], crs=4326)


# Robust readers
# -------------------------
def _detect_lon_lat_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    lon_cands = ["lon", "lng", "longitude", "x"]
    lat_cands = ["lat", "latitude", "y"]
    lon = next((c for c in lon_cands if c in df.columns), None)
    lat = next((c for c in lat_cands if c in df.columns), None)
    return (lon, lat) if lon and lat else None


def read_any(path: str | Path) -> gpd.GeoDataFrame:
    """Read GeoParquet/Parquet/GeoJSON/GPKG/Shapefile → GeoDataFrame (EPSG:4326)."""
    path = str(path)
    # 1) Try GeoParquet
    try:
        gdf = gpd.read_parquet(path)
        return gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
    except Exception:
        pass
    # 2) Parquet with WKB/WKT or lon/lat
    if Path(path).suffix.lower() == ".parquet":
        pdf = pd.read_parquet(path)
        geom_col = next(
            (c for c in ("geometry", "geom", "wkb", "wkt") if c in pdf.columns), None
        )
        if geom_col is not None:
            try:
                probe = pdf[geom_col].dropna().head(1)
                if len(probe) and isinstance(
                    probe.iloc[0], (bytes, bytearray, memoryview)
                ):
                    geom = pdf[geom_col].apply(
                        lambda b: wkb.loads(b) if pd.notna(b) else None
                    )
                else:
                    geom = pdf[geom_col].apply(
                        lambda s: wkt.loads(s) if pd.notna(s) else None
                    )
                gdf = gpd.GeoDataFrame(pdf, geometry=geom, crs=4326)
                return gdf
            except Exception:
                pass
        ll = _detect_lon_lat_columns(pdf)
        if ll is not None:
            lon, lat = ll
            return gpd.GeoDataFrame(
                pdf, geometry=gpd.points_from_xy(pdf[lon], pdf[lat], crs=4326), crs=4326
            )
    # 3) Vector via fiona
    gdf = gpd.read_file(path)
    return gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)


# Sizing & export helpers
# -------------------------
def size_from_emission(
    series: Iterable[float], min_size: float, max_size: float
) -> np.ndarray:
    s = pd.Series(series).astype(float)
    s = s.where(np.isfinite(s), 0.0).clip(lower=0.0)
    s_norm = np.sqrt(s / s.max()) if (s > 0).any() else pd.Series(np.zeros(len(s)))
    return (min_size + (max_size - min_size) * s_norm).to_numpy()


def export_geojson(
    gdf: gpd.GeoDataFrame,
    out_geojson: Path,
    keep_cols: Optional[Iterable[str]] = None,
    precision: int = 5,
    clip_aus: bool = False,
) -> None:
    g = gdf.copy()
    if not all(g.geometry.geom_type == "Point"):
        g["geometry"] = g.geometry.representative_point()
    if clip_aus:
        try:
            g = gpd.clip(g, load_aus_polygon())
        except Exception:
            pass
    if keep_cols:
        keep_cols = [c for c in keep_cols if c in g.columns]
        g = g[keep_cols + [g.geometry.name]]
    if _shp_set_precision is not None and precision is not None:
        grid = 10 ** (-precision)
        g["geometry"] = g.geometry.apply(lambda geom: _shp_set_precision(geom, grid))
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out_geojson, driver="GeoJSON")
    print(f"[OK] Exported GeoJSON → {out_geojson}")


# Provider coloring
# -------------------------
PROVIDER_COLOR_MAP: Dict[str, str] = {
    "sron": "#4c9cff",  # blue
    "kayrros": "#00e6a8",  # teal
    "carbon_mapper": "#ff70c2",  # pink
    "other": "#8a8fa3",  # gray
}


def normalize_provider(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "other"
    s = str(value).strip().lower()
    if s in ("sron",):
        return "sron"
    if s in ("kayrros",):
        return "kayrros"
    if s in ("carbon_mapper", "carbon-mapper", "carbon mapper", "cm"):
        return "carbon_mapper"
    return s if s in PROVIDER_COLOR_MAP else "other"


# CLI :'(
# -------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(
        description="Plot methane plumes over Australia (dark theme, provider-colored)",
        formatter_class=p,
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Points input (GeoParquet/Parquet/GPKG/GeoJSON/Shapefile)",
    )
    ap.add_argument(
        "--out-png",
        default=None,
        help="Output PNG (default: plots/<input>_au_dark.png)",
    )
    ap.add_argument(
        "--out-geojson", default=None, help="Optional: also export a web-ready GeoJSON"
    )
    ap.add_argument(
        "--geojson-precision",
        type=int,
        default=5,
        help="Coordinate decimals for GeoJSON (Shapely>=2)",
    )
    ap.add_argument("--title", default="Methane Plumes – Australia (dark)")
    ap.add_argument(
        "--emission-col",
        default="emission_tph",
        help="Field with emission in tons/hour",
    )
    ap.add_argument(
        "--source-col",
        default="provider",
        help="Category field for color (default: provider)",
    )
    ap.add_argument(
        "--min-size", type=float, default=6.0, help="Min marker size (pt^2)"
    )
    ap.add_argument(
        "--max-size", type=float, default=180.0, help="Max marker size (pt^2)"
    )
    ap.add_argument(
        "--size-scale", type=float, default=0.7, help="Global size multiplier"
    )
    ap.add_argument("--alpha", type=float, default=0.9, help="Point alpha (0–1)")
    ap.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    ap.add_argument("--clip-au", action="store_true", help="Clip to Australia polygon")
    ap.add_argument("--debug", action="store_true")

    # Clusters underlay (optional)
    ap.add_argument(
        "--clusters-in",
        default=None,
        help="Optional clusters polygons (GPKG/GeoJSON/Parquet)",
    )
    ap.add_argument(
        "--clusters-layer", default=None, help="Layer name if clusters-in is GPKG"
    )
    ap.add_argument(
        "--cluster-fill-color", default="#00e6a8", help="Cluster fill color"
    )
    ap.add_argument(
        "--cluster-fill-alpha", type=float, default=0.18, help="Cluster fill opacity"
    )
    ap.add_argument(
        "--cluster-outline-color", default="#00e6a8", help="Cluster outline color"
    )
    ap.add_argument(
        "--cluster-outline-alpha",
        type=float,
        default=0.85,
        help="Cluster outline opacity",
    )
    ap.add_argument(
        "--cluster-outline-width", type=float, default=0.8, help="Cluster outline width"
    )
    ap.add_argument(
        "--no-clusters",
        action="store_true",
        help="Skip plotting clusters even if clusters-in is given",
    )

    # Carbon Mapper on top
    ap.add_argument(
        "--cm-on-top", action="store_true", help="Draw CARBON_MAPPER last (on top)"
    )
    ap.add_argument(
        "--cm-size-scale",
        type=float,
        default=None,
        help="Extra multiplier for CARBON_MAPPER point sizes (e.g. 0.8)",
    )
    ap.add_argument(
        "--only-providers",
        type=str,
        default=None,
        help="Comma-separated list of providers to include (e.g. 'SRON,KAYRROS')",
    )

    return ap.parse_args(argv)


# Main
# -------------------------
def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Points
    pts = read_any(in_path)
    pts = pts.to_crs(4326) if pts.crs else pts.set_crs(4326)

    # Clusters (optional)
    clusters = None
    if args.clusters_in:
        clp = Path(args.clusters_in)
        if clp.exists():
            clusters = read_any(clp)
            if args.clip_au and clusters is not None and not clusters.empty:
                try:
                    clusters = gpd.clip(clusters, load_aus_polygon())
                except Exception:
                    pass
        else:
            print(f"[warn] clusters not found: {clp} (skipping)")

    # Output path
    out_png = (
        Path(args.out_png)
        if args.out_png
        else Path("plots") / f"{in_path.stem}_au_dark.png"
    )

    # Optional GeoJSON export (before plotting)
    if args.out_geojson:
        keep = [
            "emission_tph",
            "provider",
            "operator",
            "name",
            "facility",
            "plume_id_src",
            "obs_datetime_utc",
        ]
        export_geojson(
            gdf=pts,
            out_geojson=Path(args.out_geojson),
            keep_cols=keep,
            precision=args.geojson_precision,
            clip_aus=bool(args.clip_au),
        )

    # Prepare points geometry
    if not all(pts.geometry.geom_type == "Point"):
        pts = pts.copy()
        pts["geometry"] = pts.geometry.representative_point()
    if args.clip_au:
        pts = gpd.clip(pts, load_aus_polygon())

    # ---- Styling
    bg = "#0f1117"
    axbg = "#12151c"
    fg = "#d0d4dc"
    gridc = "#2a2f3a"
    fig, ax = plt.subplots(figsize=(12, 10), dpi=args.dpi)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(axbg)

    # AU outline
    if args.clip_au:
        try:
            load_aus_polygon().boundary.plot(
                ax=ax, color="#2b313b", linewidth=0.6, zorder=1
            )
        except Exception:
            pass

    # ---- Clusters beneath points
    if clusters is not None and not clusters.empty:
        try:
            clusters.plot(
                ax=ax,
                color=args.cluster_fill_color,
                alpha=float(args.cluster_fill_alpha),
                linewidth=0,
                zorder=2,
            )
        except Exception:
            pass
        try:
            clusters.boundary.plot(
                ax=ax,
                color=args.cluster_outline_color,
                alpha=float(args.cluster_outline_alpha),
                linewidth=float(args.cluster_outline_width),
                zorder=3,
            )
        except Exception:
            pass

    # ---- Points (provider-colored)
    if args.emission_col not in pts.columns:
        raise ValueError(f"Column '{args.emission_col}' not found in data.")

    sizes = size_from_emission(
        pts[args.emission_col], args.min_size, args.max_size
    ) * float(args.size_scale)
    cats = (
        pts[args.source_col].apply(normalize_provider)
        if args.source_col and args.source_col in pts.columns
        else pd.Series(["other"] * len(pts), index=pts.index)
    )
    pts = pts.assign(_cat=cats, _size=sizes)

    groups: List[Tuple[str, gpd.GeoDataFrame]] = list(pts.groupby("_cat", sort=False))
    if args.cm_on_top:
        groups.sort(key=lambda kv: 0 if kv[0] != "carbon_mapper" else 1)

    handles, labels = [], []
    z = 10  # ensure above cluster layers
    for cat, sub in groups:
        color = PROVIDER_COLOR_MAP.get(cat, "#8a8fa3")
        size_vec = sub["_size"].values
        if args.cm_size_scale is not None and cat == "carbon_mapper":
            size_vec = size_vec * float(args.cm_size_scale)
        sc = ax.scatter(
            sub.geometry.x.values,
            sub.geometry.y.values,
            s=size_vec,
            c=color,
            edgecolors=("#ffffff" if args.size_scale <= 0.9 else "#d9d9d9"),
            linewidths=0.25,
            alpha=args.alpha,
            zorder=z,
            label=cat.upper(),
        )
        z += 1
        handles.append(sc)
        labels.append(cat.upper())

    # Cosmetics
    for spine in ax.spines.values():
        spine.set_color(gridc)
    ax.tick_params(colors=fg, labelsize=9)
    ax.grid(True, color=gridc, linewidth=0.6, alpha=0.4)
    ax.set_title(args.title, color=fg, fontsize=18, pad=12, loc="left")
    ax.set_xlabel("Longitude", color=fg)
    ax.set_ylabel("Latitude", color=fg)

    # Provider legend
    if args.source_col:
        leg1 = ax.legend(
            handles=handles,
            labels=labels,
            title="Provider",
            title_fontsize=11,
            fontsize=10,
            loc="lower left",
            frameon=True,
            facecolor=axbg,
            edgecolor=gridc,
        )
        for t in leg1.get_texts():
            t.set_color(fg)
        leg1.get_title().set_color(fg)
        ax.add_artist(leg1)

    # Emission size legend
    try:
        vals = pts[args.emission_col].astype(float).where(lambda x: x > 0).dropna()
        if len(vals):
            vmax = float(vals.max())
            ref = np.unique(
                np.round(np.array([vmax * 0.1, vmax * 0.25, vmax * 0.5, vmax]), 2)
            )
            ref = ref[ref > 0]
            if len(ref):
                ref_sizes = size_from_emission(
                    ref, args.min_size, args.max_size
                ) * float(args.size_scale)
                for r, rs in zip(ref, ref_sizes):
                    ax.scatter([], [], s=rs, c="#d0d4dc", alpha=0.8, label=f"{r:g} tph")
                leg2 = ax.legend(
                    title="Emission (tph)",
                    title_fontsize=11,
                    fontsize=10,
                    loc="lower right",
                    frameon=True,
                    facecolor=axbg,
                    edgecolor=gridc,
                )
                for t in leg2.get_texts():
                    t.set_color(fg)
                leg2.get_title().set_color(fg)
    except Exception:
        pass

    fig.tight_layout(pad=1.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), dpi=args.dpi)
    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
