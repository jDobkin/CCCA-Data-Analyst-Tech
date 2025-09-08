#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export processed data to geojson for webmap applicaitons

Example:
    python src\export_geojson.py
"""

import geopandas as gpd
from pathlib import Path

pts_pq = Path("data/processed/au_plumes_clustered.parquet")
clu_gpkg = Path("data/processed/clusters_au.gpkg")  # layer=clusters_au
out_dir = Path("webmap-app/public/data")
out_dir.mkdir(parents=True, exist_ok=True)

# points
gdf_p = gpd.read_parquet(pts_pq).to_crs(4326)
gdf_p.to_file(out_dir / "au_plumes_clustered.geojson", driver="GeoJSON")
gdf_p.to_file(out_dir / "au_plumes_clustered.geojson", driver="GeoJSON")

# clusters (polygons)
gdf_c = gpd.read_file(clu_gpkg, layer="clusters_au").to_crs(4326)
gdf_c.to_file(out_dir / "clusters_au.geojson", driver="GeoJSON")
print(
    "Wrote:",
    out_dir / "au_plumes_clustered.geojson",
    "and",
    out_dir / "clusters_au.geojson",
)
