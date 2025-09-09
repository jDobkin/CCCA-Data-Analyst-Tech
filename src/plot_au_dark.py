#!/usr/bin/env python3
"""
Dark-themed Australia methane plume plotter

Features
- Reads GeoParquet/Parquet/GeoPackage/Shapefile/etc. (points or polygons)
- Optional clip to Australia polygon (robust to GeoPandas 1.0+ where datasets were removed)
- Source-type coloring + emissions-based marker sizing
- PNG export with dark modern styling

Example
    python src/plot_au_dark.py `
        --in "data/interim/unified_plus_cm.parquet" `
        --clip-au `
        --size-scale 0.7 --min-size 4 --max-size 160 `
        --out-png "plots/au_plumes_dark.png"

Requirements
    pip install geopandas geodatasets shapely pandas pyarrow matplotlib

Notes
- If `geodatasets` is not available, the script will try a Natural Earth URL,
  then fall back to an Australia bounding box.
- If features are polygons, the script plots `representative_point()`
  so labels/points lie inside the polygons.
"""


import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from shapely import wkb, wkt

try:
    from shapely import set_precision as _shp_set_precision  # shapely >= 2.0
except Exception:
    _shp_set_precision = None
try:
    from geodatasets import get_path as _gd_get_path  # type: ignore
except Exception:
    _gd_get_path = None


# -------------------------
# Australia polygon loader
# -------------------------
def load_aus_polygon() -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame [EPSG:4326] for Australia.

    Tries geodatasets (local), then Natural Earth over HTTPS, then a bbox fallback.
    """
    # 1) Preferred: geodatasets (local after first install)
    if _gd_get_path is not None:
        for key in [
            "naturalearth.cultural.vectors-admin_0_countries",  # NE Admin-0 modern key
            "naturalearth_lowres",  # legacy name if present
        ]:
            try:
                world = gpd.read_file(_gd_get_path(key)).to_crs(4326)
                # Pick the best candidate name column available
                name_col = None
                for cand in ("ADMIN", "name", "NAME", "SOVEREIGNT"):
                    if cand in world.columns:
                        name_col = cand
                        break
                if name_col is not None:
                    aus_geom = world.loc[
                        world[name_col]
                        .astype(str)
                        .str.contains("Australia", case=False, na=False),
                        "geometry",
                    ]
                    if len(aus_geom):
                        geom = aus_geom.unary_union
                        return gpd.GeoDataFrame(geometry=[geom], crs=4326)
            except Exception:
                pass  # try next option

    # 2) Fallback: Natural Earth Admin-0 via URL
    try:
        ne_zip = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(ne_zip).to_crs(4326)
        if "ADMIN" in world.columns:
            aus_geom = world.loc[world["ADMIN"] == "Australia", "geometry"]
            if len(aus_geom):
                geom = aus_geom.unary_union
                return gpd.GeoDataFrame(geometry=[geom], crs=4326)
    except Exception:
        pass

    # 3) Last resort: generous Australia bbox
    bbox_geom = box(112.0, -44.0, 154.0, -10.0)
    return gpd.GeoDataFrame(geometry=[bbox_geom], crs=4326)


# Robust data loader
# -------------------------
def _detect_lon_lat_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    lon_cands = ["lon", "lng", "longitude", "x"]
    lat_cands = ["lat", "latitude", "y"]
    lon = next((c for c in lon_cands if c in df.columns), None)
    lat = next((c for c in lat_cands if c in df.columns), None)
    if lon and lat:
        return lon, lat
    return None


def read_any(path: str | Path) -> gpd.GeoDataFrame:
    """Read a dataset as GeoDataFrame.

    Supports GeoParquet, plain Parquet with WKB/WKT or lon/lat,
    and vector formats supported by fiona. Ensures EPSG:4326.
    """
    path = str(path)
    # 1) Try GeoParquet with GeoPandas
    try:
        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf.set_crs(4326, inplace=True)
        else:
            gdf = gdf.to_crs(4326)
        return gdf
    except Exception:
        pass

    # 2) Try generic Parquet -> build geometry
    if Path(path).suffix.lower() == ".parquet":
        pdf = pd.read_parquet(path)
        geom_col = None
        # WKB/WKT
        for c in ("geometry", "geom", "wkb", "wkt"):
            if c in pdf.columns:
                geom_col = c
                break
        if geom_col is not None:
            try:
                # WKB
                geoms = pdf[geom_col].dropna().head(1)
                if len(geoms) and isinstance(
                    geoms.iloc[0], (bytes, bytearray, memoryview)
                ):
                    geom = pdf[geom_col].apply(
                        lambda b: wkb.loads(b) if pd.notna(b) else None
                    )
                else:
                    # WKT
                    geom = pdf[geom_col].apply(
                        lambda s: wkt.loads(s) if pd.notna(s) else None
                    )
                gdf = gpd.GeoDataFrame(pdf, geometry=geom, crs=4326)
                return gdf
            except Exception:
                pass
        # lon/latallback
        ll = _detect_lon_lat_columns(pdf)
        if ll is not None:
            lon, lat = ll
            gdf = gpd.GeoDataFrame(
                pdf,
                geometry=gpd.points_from_xy(pdf[lon], pdf[lat], crs=4326),
                crs=4326,
            )
            return gdf

    # 3) Any fiona-supported vector
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    else:
        gdf = gdf.to_crs(4326)
    return gdf


# Emission -> size mapping
# -------------------------
def size_from_emission(
    series: Iterable[float], min_size: float, max_size: float
) -> np.ndarray:
    """Map emissions (tph) to point sizes (points^2) with sqrt scaling.

    - Non-positive values map to `min_size`
    - Largest value maps near `max_size`
    """
    s = pd.Series(series).astype(float)
    s = s.where(np.isfinite(s), 0.0).clip(lower=0.0)
    if (s > 0).any():
        # sqrt scaling gives
        s_norm = np.sqrt(s / s.max())
    else:
        s_norm = pd.Series(np.zeros(len(s)))
    return (min_size + (max_size - min_size) * s_norm).to_numpy()


def export_geojson(
    gdf: gpd.GeoDataFrame,
    out_geojson: Path,
    keep_cols: Optional[Iterable[str]] = None,
    precision: int = 5,
    clip_aus: bool = False,
) -> None:
    """Export a lightweight GeoJSON for web maps.

    - Converts polygons to representative points
    - Optionally clips to Australia
    - Keeps only selected columns if provided (plus geometry)
    - Rounds coordinates to ~10^-precision degrees when Shapely>=2 is available
    """
    g = gdf.copy()
    if not all(g.geometry.geom_type == "Point"):
        g["geometry"] = g.geometry.representative_point()

    if clip_aus:
        try:
            aus = load_aus_polygon()
            g = gpd.clip(g, aus)
        except Exception:
            pass

    if keep_cols is not None:
        keep_cols = [c for c in keep_cols if c in g.columns]
        g = g[keep_cols + [g.geometry.name]]

    # Coordinate precision
    if _shp_set_precision is not None and precision is not None:
        grid = 10 ** (-precision)
        g["geometry"] = g.geometry.apply(lambda geom: _shp_set_precision(geom, grid))

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out_geojson, driver="GeoJSON")
    print(f"Exported GeoJSON: {out_geojson}")


# Colors for source categories
# -------------------------
DEFAULT_COLOR_MAP: Dict[str, str] = {
    # normalized keys (lowercase)
    "coal": "#00c2a8",  # teal
    "coal mine methane": "#00c2a8",
    "cmm": "#00c2a8",
    "oil": "#ff8c42",  # orange
    "oil & gas": "#ff8c42",
    "gas": "#ff8c42",
    "o&g": "#ff8c42",
    "landfill": "#c4d13f",  # olive
    "wastewater": "#9b7bd1",  # purple
    "agriculture": "#4caf50",  # green
    "unknown": "#8e8e8e",  # mid-gray
    "other": "#4aa3ff",  # blue
}


def normalize_source(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    s = str(value).strip().lower()
    # light normalization
    if "coal" in s:
        return "coal"
    if "landfill" in s:
        return "landfill"
    if "wastewater" in s or "sewage" in s:
        return "wastewater"
    if "agri" in s or "cattle" in s or "feedlot" in s:
        return "agriculture"
    if "gas" in s or "oil" in s or "o&g" in s or "petrol" in s:
        return "oil & gas"
    if s in DEFAULT_COLOR_MAP:
        return s
    return "other"


# Plotting
# -------------------------
def plot_points(
    gdf: gpd.GeoDataFrame,
    out_png: Path,
    title: str,
    emission_col: str,
    source_col: Optional[str],
    min_size: float,
    max_size: float,
    size_scale: float,
    alpha: float = 0.9,
    clip_aus: bool = False,
    dpi: int = 220,
) -> None:
    # Prepare geometry: if not points, use representative points
    if not all(gdf.geometry.geom_type == "Point"):
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.representative_point()

    # Optionally clip to Australia
    if clip_aus:
        aus = load_aus_polygon()
        gdf = gpd.clip(gdf, aus)

    # Ensure emission column exists
    if emission_col not in gdf.columns:
        raise ValueError(
            f"Column '{emission_col}' not found in data. Available: {list(gdf.columns)[:20]}..."
        )

    # Compute sizes
    sizes = size_from_emission(gdf[emission_col], min_size, max_size) * float(
        size_scale
    )

    # Colors by source
    if source_col and source_col in gdf.columns:
        cats = gdf[source_col].apply(normalize_source)
    else:
        cats = pd.Series(["other"] * len(gdf), index=gdf.index)
        source_col = None

    gdf = gdf.assign(_cat=cats, _size=sizes)

    # ----------------- Figure styling (dark) -----------------
    bg = "#0f1117"  # canvas
    axbg = "#12151c"  # axes face
    fg = "#d0d4dc"  # text
    gridc = "#2a2f3a"  # grid/spines

    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(axbg)

    # Simple Australia outline when clipping
    if clip_aus:
        try:
            aus_outline = load_aus_polygon()
            aus_outline.boundary.plot(ax=ax, color="#2b313b", linewidth=0.6, zorder=1)
        except Exception:
            pass

    # Plot by category for legend control
    z = 10
    handles = []
    labels = []
    for cat, sub in gdf.groupby("_cat"):
        color = DEFAULT_COLOR_MAP.get(cat, "#4aa3ff")
        sc = ax.scatter(
            sub.geometry.x.values,
            sub.geometry.y.values,
            s=sub["_size"].values,
            c=color,
            edgecolors=("white" if size_scale <= 0.9 else "#d9d9d9"),
            linewidths=0.2,
            alpha=alpha,
            zorder=z,
            label=cat,
        )
        z += 1
        handles.append(sc)
        labels.append(cat)

    # Axes cosmetics
    for spine in ax.spines.values():
        spine.set_color(gridc)
    ax.tick_params(colors=fg, labelsize=9)
    ax.grid(True, color=gridc, linewidth=0.6, alpha=0.4)

    # Titles
    ax.set_title(title, color=fg, fontsize=18, pad=12, loc="left")
    ax.set_xlabel("Longitude", color=fg)
    ax.set_ylabel("Latitude", color=fg)

    # Category legend
    if source_col is not None:
        leg1 = ax.legend(
            handles=handles,
            labels=labels,
            title=f"Source ({source_col})",
            title_fontsize=11,
            fontsize=10,
            loc="lower left",
            frameon=True,
            facecolor=axbg,
            edgecolor=gridc,
        )
        for text in leg1.get_texts():
            text.set_color(fg)
        leg1.get_title().set_color(fg)
        ax.add_artist(leg1)

    # Size legend (emission tph)
    try:
        # Choose 3–4 nice reference values based on data
        vals = gdf[emission_col].astype(float).where(lambda x: x > 0).dropna()
        if len(vals):
            vmax = float(vals.max())
            ref = np.unique(
                np.round(
                    np.array(
                        [
                            vmax * 0.1,
                            vmax * 0.25,
                            vmax * 0.5,
                            vmax,
                        ]
                    ),
                    2,
                )
            )
            ref = ref[ref > 0]
            if len(ref) > 0:
                ref_sizes = size_from_emission(ref, min_size, max_size) * float(
                    size_scale
                )
                # Dummy scatter for legend
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
                for text in leg2.get_texts():
                    text.set_color(fg)
                leg2.get_title().set_color(fg)
    except Exception:
        pass

    # Tight layout and save
    fig.tight_layout(pad=1.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), dpi=dpi)
    print(f"Saved: {out_png}")


# CLI :'(
# -------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot methane plumes over Australia with dark theme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input file (GeoParquet/Parquet/GeoPackage/Shapefile)",
    )
    p.add_argument("--out-png", default=None, help="Output PNG path (auto if omitted)")
    p.add_argument(
        "--out-geojson",
        default=None,
        help="Optional: also export a web-ready GeoJSON here",
    )
    p.add_argument(
        "--geojson-precision",
        type=int,
        default=5,
        help="Coordinate decimal places for GeoJSON (if Shapely>=2)",
    )
    p.add_argument("--title", default="Methane Plumes – Australia (dark)")
    p.add_argument(
        "--emission-col",
        default="emission_tph",
        help="Field with emission in tons/hour",
    )
    p.add_argument(
        "--source-col",
        default="source_type",
        help="Categorical field for source coloring (set blank to disable)",
    )
    p.add_argument(
        "--min-size", type=float, default=6.0, help="Minimum marker size (points^2)"
    )
    p.add_argument(
        "--max-size", type=float, default=180.0, help="Maximum marker size (points^2)"
    )
    p.add_argument(
        "--size-scale",
        type=float,
        default=0.7,
        help="Multiply all marker sizes by this factor (<1 smaller)",
    )
    p.add_argument("--alpha", type=float, default=0.9, help="Point alpha (0–1)")
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    p.add_argument(
        "--clip-au", action="store_true", help="Clip features to Australia polygon"
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Read data
    gdf = read_any(in_path)

    # Normalize CRS
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    else:
        gdf = gdf.to_crs(4326)

    # Prepare output path
    if args.out_png:
        out_png = Path(args.out_png)
    else:
        out_dir = Path("plots")
        out_png = out_dir / f"{in_path.stem}_au_dark.png"

    # Allow disabling source coloring by passing empty string
    source_col = (
        args.source_col if (args.source_col or args.source_col is None) else None
    )

    # Optional GeoJSON export before plotting (mirror the plotting geometry/clipping)
    if args.out_geojson:
        keep_default = [
            "emission_tph",
            "source_type",
            "operator",
            "name",
            "facility",
            "plume_id",
            "obs_date",
        ]
        export_geojson(
            gdf=gdf,
            out_geojson=Path(args.out_geojson),
            keep_cols=keep_default,
            precision=int(args.geojson_precision),
            clip_aus=bool(args.clip_au),
        )

    if args.debug:
        print("=== DEBUG ===")
        print("Rows:", len(gdf))
        print("Columns:", list(gdf.columns))
        print("CRS:", gdf.crs)
        print("Clip AU:", args.clip_au)
        print("Emission column:", args.emission_col)
        print("Source column:", source_col)
        print("Out PNG:", out_png)
        print("============\n")

    plot_points(
        gdf=gdf,
        out_png=out_png,
        title=args.title,
        emission_col=args.emission_col,
        source_col=source_col,
        min_size=args.min_size,
        max_size=args.max_size,
        size_scale=args.size_scale,
        alpha=args.alpha,
        clip_aus=bool(args.clip_au),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
