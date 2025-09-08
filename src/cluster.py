#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global DBSCAN clustering and Australia subset.

Example:
  python src/cluster.py
    --in-gpkg data/processed/plumes_dedup.gpkg --layer plumes_dedup
    --engine haversine --eps-m 30000 --min-samples 5 --make_png
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Constants
EARTH_M = 6_371_008.8  # mean Earth radius (meters)

# Natural Earth mirrors and GADM AUS
NE_SOURCES = [
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip",
    # Legacy site
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip",
]
GADM_AUS = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_AUS_shp.zip"


# Boundary acquisition
def _download_zip(url: str, out_dir: Path) -> bool:
    import requests  # local import to avoid hard dependency until needed

    headers = {"User-Agent": "Mozilla/5.0 (cluster-script/1.0)"}
    r = requests.get(url, timeout=180, headers=headers)
    if r.status_code != 200:
        return False
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out_dir)
    return True


def ensure_aus_boundary(
    path_dir: Path = Path("data/raw/boundaries"),
) -> gpd.GeoDataFrame:
    """
    Return an EPSG:4326 GeoDataFrame for the Australia polygon via:
      1) Natural Earth countries (filter to Australia)
      2) GADM AUS (Australia only)
      3) GeoPandas built-in 'naturalearth_lowres' (fallback)
    """
    path_dir.mkdir(parents=True, exist_ok=True)

    # 1) Natural Earth countries
    ne_shp = path_dir / "ne_10m_admin_0_countries.shp"
    if not ne_shp.exists():
        for url in NE_SOURCES:
            try:
                if _download_zip(url, path_dir) and ne_shp.exists():
                    break
            except Exception:
                continue
    if ne_shp.exists():
        ne = gpd.read_file(ne_shp).to_crs("EPSG:4326")
        name_col = (
            "NAME_EN"
            if "NAME_EN" in ne.columns
            else ("ADMIN" if "ADMIN" in ne.columns else None)
        )
        if name_col:
            aus = ne[ne[name_col].str.contains("Australia", case=False, na=False)]
            if not aus.empty:
                return aus

    # 2) GADM AUS-only
    gadm_shp = path_dir / "gadm41_AUS_0.shp"
    if not gadm_shp.exists():
        try:
            _download_zip(GADM_AUS, path_dir)
        except Exception:
            pass
    if gadm_shp.exists():
        return gpd.read_file(gadm_shp).to_crs("EPSG:4326")

    # 3) Built-in lowres fallback
    lowres = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).to_crs(
        "EPSG:4326"
    )
    aus = lowres[lowres["name"].str.contains("Australia", case=False, na=False)]
    if aus.empty:
        raise RuntimeError("Failed to obtain Australia boundary from any source.")
    return aus


# IO helpers
def load_input(in_parquet: str, in_gpkg: str | None, layer: str) -> gpd.GeoDataFrame:
    p = Path(in_parquet)
    if p.exists():
        gdf = gpd.read_parquet(p)
    elif in_gpkg:
        gdf = gpd.read_file(in_gpkg, layer=layer)
    else:
        raise FileNotFoundError(f"Input not found: {p} (or provide --in-gpkg)")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf.to_crs("EPSG:4326").dropna(subset=["geometry"]).copy()


# DBSCAN engines
def dbscan_projected(
    gdf: gpd.GeoDataFrame, eps_m: float, min_samples: int, epsg_meters: int
) -> np.ndarray:
    g = gdf.to_crs(epsg=epsg_meters)
    X = np.c_[g.geometry.x, g.geometry.y]
    model = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    return model.fit_predict(X)


def dbscan_haversine(
    gdf: gpd.GeoDataFrame, eps_m: float, min_samples: int
) -> np.ndarray:
    eps_rad = eps_m / EARTH_M
    lat = np.radians(gdf.geometry.y.to_numpy())
    lon = np.radians(gdf.geometry.x.to_numpy())
    X = np.c_[lat, lon]  # sklearn expects [lat, lon] in radians for haversine
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", n_jobs=-1)
    return model.fit_predict(X)


# Cluster summaries
def summarize_clusters(
    gdf: gpd.GeoDataFrame, label_col: str = "cluster_id"
) -> gpd.GeoDataFrame:
    """
    For each cluster_id >= 0, compute:
      - size
      - median emission
      - provider mix (e.g., "KAYRROS:12;SRON:4")
      - convex hull geometry
    Returns a GeoDataFrame with geometry column properly named.
    """
    d = gdf[gdf[label_col] >= 0].copy()
    if d.empty:
        return gpd.GeoDataFrame(
            columns=[
                "cluster_id",
                "size",
                "median_emission_tph",
                "providers",
                "geometry",
            ],
            crs=gdf.crs,
        )

    stats = (
        d.groupby(label_col)
        .agg(
            size=("cluster_id", "size"), median_emission_tph=("emission_tph", "median")
        )
        .reset_index()
    )

    prov_mix = (
        d.groupby([label_col, "provider"])
        .size()
        .unstack(fill_value=0)
        .apply(
            lambda r: ";".join(f"{c}:{int(v)}" for c, v in r.items() if v > 0), axis=1
        )
        .reset_index()
        .rename(columns={0: "providers"})
    )
    stats = stats.merge(prov_mix, on=label_col, how="left")

    # Dissolve by cluster -> convex hulls (ensure geometry column name)
    diss = d.dissolve(by=label_col)  # index = cluster_id, geometry = union
    hull_series = diss.geometry.convex_hull.buffer(0)  # GeoSeries (no name)
    hulls = gpd.GeoDataFrame(
        {label_col: hull_series.index, "geometry": hull_series.values},
        geometry="geometry",
        crs=gdf.crs,
    )

    out = stats.merge(hulls, on=label_col, how="left")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


# Main
def main():
    ap = argparse.ArgumentParser(
        description="Global DBSCAN clustering + Australia subset."
    )
    ap.add_argument(
        "--in-parquet",
        default="data/processed/plumes_dedup.parquet",
        help="Input GeoParquet",
    )
    ap.add_argument("--in-gpkg", default=None, help="Alternative input GPKG")
    ap.add_argument(
        "--layer", default="plumes_dedup", help="Layer name when reading GPKG"
    )
    ap.add_argument(
        "--engine",
        choices=["project", "haversine"],
        default="project",
        help="DBSCAN coordinate mode",
    )
    ap.add_argument(
        "--eps-m",
        type=float,
        default=30000.0,
        help="DBSCAN eps in meters (per Schuit ~30 km)",
    )
    ap.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    ap.add_argument(
        "--epsg-meters",
        type=int,
        default=3857,
        help="Projected CRS (meters) if engine=project",
    )
    ap.add_argument(
        "--au-out-parquet",
        default="data/processed/au_plumes_clustered.parquet",
        help="AU-only points output",
    )
    ap.add_argument(
        "--au-out-gpkg",
        default="data/processed/clusters_au.gpkg",
        help="AU clusters GPKG (layer=clusters_au)",
    )
    ap.add_argument(
        "--global-out-parquet",
        default="data/processed/plumes_clustered_global.parquet",
        help="Global points+labels",
    )
    ap.add_argument(
        "--make-png",
        action="store_true",
        help="Write a quick PNG map for Australia clusters",
    )
    args = ap.parse_args()

    print(">>> cluster.py starting")
    print("argv:", __import__("sys").argv)

    # Load input
    gdf = load_input(args.in_parquet, args.in_gpkg, args.layer)

    # DBSCAN
    if args.engine == "project":
        labels = dbscan_projected(
            gdf,
            eps_m=args.eps_m,
            min_samples=args.min_samples,
            epsg_meters=args.epsg_meters,
        )
    else:
        labels = dbscan_haversine(gdf, eps_m=args.eps_m, min_samples=args.min_samples)
    gdf = gdf.copy()
    gdf["cluster_id"] = labels  # -1 = noise

    # Save global points + labels
    Path(args.global_out_parquet).parent.mkdir(parents=True, exist_ok=True)
    try:
        gdf.to_parquet(args.global_out_parquet, index=False)
    except Exception as e:
        print(f"[warn] couldn't write global parquet ({e})")

    # Australia boundary (robust sources + fallback)
    try:
        aus = ensure_aus_boundary()
    except Exception as e:
        print(f"[warn] boundary download failed ({e}); using built-in lowres.")
        lowres = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).to_crs(
            "EPSG:4326"
        )
        aus = lowres[lowres["name"].str.contains("Australia", case=False, na=False)]

    if aus.empty:
        raise RuntimeError("Australia boundary not found.")
    aus = aus.to_crs("EPSG:4326")

    # union polygon API compatibility (geopandas 1.0+ vs older)
    try:
        au_poly = aus.union_all()  # GeoPandas 1.0+
    except Exception:
        au_poly = aus.unary_union  # fallback for older versions

    # Clip to AU
    gdf_au = gdf[gdf.geometry.within(au_poly)].copy()

    # Summaries & hulls for AU
    clusters_au = summarize_clusters(gdf_au, label_col="cluster_id")

    # Write AU outputs (points + clusters)
    Path(args.au_out_parquet).parent.mkdir(parents=True, exist_ok=True)
    try:
        gdf_au.to_parquet(args.au_out_parquet, index=False)
    except Exception as e:
        print(f"[warn] couldn't write AU parquet ({e})")

    Path(args.au_out_gpkg).parent.mkdir(parents=True, exist_ok=True)
    clusters_au.to_file(args.au_out_gpkg, layer="clusters_au", driver="GPKG")

    # Research list output (AU points with attributes + lat/lon)
    research_csv = Path("data/processed/research_list.csv")
    gdf_au_out = gdf_au.copy()
    gdf_au_out["longitude"] = gdf_au_out.geometry.x
    gdf_au_out["latitude"] = gdf_au_out.geometry.y
    cols = [
        "plume_id_src",
        "obs_datetime_utc",
        "provider",
        "emission_tph",
        "emission_unc_tph",
        "cluster_id",
        "latitude",
        "longitude",
    ]
    cols = [c for c in cols if c in gdf_au_out.columns]
    gdf_au_out[cols].to_csv(research_csv, index=False)

    # Console summary
    print("=== CLUSTERING DONE ===")
    print(
        f"Points (global): {len(gdf)}  | clusters (labels incl. noise): {len(np.unique(labels))}"
    )
    print(f"Points (AU):     {len(gdf_au)}")
    print(f"Clusters (AU):   {clusters_au.shape[0]}")
    print(f"[OK] AU points parquet → {args.au_out_parquet}")
    print(f"[OK] AU clusters gpkg  → {args.au_out_gpkg}")
    print(f"[OK] Research list CSV → {research_csv}")

    # Optional PNG quicklook but sexy this time
    if args.make_png:
        try:
            out_png = Path("outputs/maps/clusters_au.png")
            out_png.parent.mkdir(parents=True, exist_ok=True)

            # Dark modern vibe
            plt.rcParams.update(
                {
                    "figure.facecolor": "#0b0b0e",
                    "axes.facecolor": "#0b0b0e",
                    "axes.edgecolor": "#2c2c33",
                    "axes.labelcolor": "#e9e9ef",
                    "xtick.color": "#b9b9c6",
                    "ytick.color": "#b9b9c6",
                    "text.color": "#e9e9ef",
                    "savefig.facecolor": "#0b0b0e",
                }
            )

            fig, ax = plt.subplots(figsize=(11, 9), dpi=150)

            # Australia outline
            aus.boundary.plot(ax=ax, linewidth=0.8, color="#3a3a45")

            # Split noise vs clusters
            pts_noise = gdf_au[gdf_au["cluster_id"] < 0]
            pts_cluster = gdf_au[gdf_au["cluster_id"] >= 0]

            # Build a modern colormap for clusters
            cluster_ids = (
                sorted(pts_cluster["cluster_id"].unique().tolist())
                if not pts_cluster.empty
                else []
            )
            n_clusters = max(1, len(cluster_ids))
            cmap = plt.cm.get_cmap("turbo", n_clusters)
            if cluster_ids:
                norm = mcolors.Normalize(vmin=min(cluster_ids), vmax=max(cluster_ids))
            else:
                norm = None

            # Plot clusters with color by cluster_id
            if not pts_cluster.empty:
                pts_cluster.plot(
                    ax=ax,
                    markersize=14,
                    alpha=0.9,
                    column="cluster_id",
                    cmap=cmap,
                    norm=norm,
                    linewidth=0,
                )

            # Noise in muted gray
            if not pts_noise.empty:
                pts_noise.plot(
                    ax=ax,
                    markersize=8,
                    alpha=0.35,
                    color="#8a8fa3",
                    linewidth=0,
                )

            # Cluster hulls with translucent neon edges
            if not clusters_au.empty:
                clusters_au.boundary.plot(
                    ax=ax, linewidth=1.2, color="#00e6a8", alpha=0.85
                )

            # Legend (simple categories)
            handles = []
            if not pts_cluster.empty:
                handles.append(
                    mpatches.Patch(
                        facecolor="#4c9cff",
                        edgecolor="none",
                        alpha=0.9,
                        label="Clusters",
                    )
                )
            if not pts_noise.empty:
                handles.append(
                    mpatches.Patch(
                        facecolor="#8a8fa3", edgecolor="none", alpha=0.35, label="Noise"
                    )
                )
            if handles:
                ax.legend(
                    handles=handles,
                    frameon=False,
                    loc="lower left",
                    fontsize=10,
                    labelcolor="#e9e9ef",
                )

            # Title labels grid
            ax.set_title(
                "Methane Super-Emitter Clusters — Australia", fontsize=14, pad=12
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, color="#2c2c33", linewidth=0.5, alpha=0.6)

            fig.tight_layout()
            fig.savefig(out_png)
            print(f"[OK] PNG map → {out_png}")
        except Exception as e:
            print(f"[warn] PNG generation failed: {e}")


if __name__ == "__main__":
    main()
