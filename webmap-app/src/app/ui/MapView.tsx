/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { LngLatBoundsLike, Map, Popup } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const DATA_POINTS = "/data/au_plumes_clustered.geojson";
const DATA_CLUSTERS = "/data/clusters_au.geojson";

// Australia bounds
const AUS_BOUNDS: LngLatBoundsLike = [112, -44, 154, -9];

type Provider = "SRON" | "KAYRROS";

export default function MapView() {
  const mapRef = useRef<Map | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [minEmission, setMinEmission] = useState<number>(0);
  const [providers, setProviders] = useState<Record<Provider, boolean>>({
    SRON: true,
    KAYRROS: true,
  });
  const [showPolys, setShowPolys] = useState(true);
  const [showPoints, setShowPoints] = useState(true);

  // Paint for circles (memoized)
  const paintCircle: any = useMemo(
    () => ({
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["zoom"],
        4, 2,
        6, 3,
        8, 5,
        10, 7,
        12, 9,
      ],
      "circle-color": [
        "case",
        ["<", ["get", "cluster_id"], 0],
        "#8a8fa3", // noise
        [
          "match",
          ["%", ["abs", ["get", "cluster_id"]], 8],
          0, "#4c9cff",
          1, "#00e6a8",
          2, "#f97d61",
          3, "#e6c229",
          4, "#ad73ff",
          5, "#56d2ef",
          6, "#ff70c2",
          7, "#87f26e",
          "#4c9cff",
        ],
      ],
      "circle-opacity": [
        "case",
        ["<", ["get", "cluster_id"], 0],
        0.35,
        0.9,
      ],
      "circle-stroke-color": "#0b0b0e",
      "circle-stroke-width": 0.5,
    }),
    []
  );

  // Build a filter expression from UI state (memoized)
  const filterExpression: any = useMemo(() => {
    const enabledProviders = (Object.keys(providers) as Provider[]).filter(
      (k) => providers[k]
    );

    const providerFilter =
      enabledProviders.length === 0
        ? ["in", ["get", "provider"], "___none___"] // match nothing
        : ["in", ["get", "provider"], ["literal", enabledProviders]];

    const emissionFilter = [
      ">=",
      ["coalesce", ["get", "emission_tph"], 0],
      minEmission,
    ];

    return ["all", providerFilter, emissionFilter];
  }, [providers, minEmission]);

  // Helper for popup HTML
  const makePopupHTML = (p: Record<string, unknown>, lng: number, lat: number) => {
    const fmt = (v: unknown) =>
      v === null || v === undefined || v === "" ? "—" : String(v);
    return `
      <div style="background:#1e1e24;color:#e9e9ef;border-radius:8px;border:1px solid #2c2c33;padding:8px 10px;font:13px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;">
        <div style="font-weight:600;margin-bottom:4px;">${fmt(p.provider)} · Cluster ${fmt(
      p.cluster_id
    )}</div>
        <div><b>Datetime (UTC):</b> ${fmt(p.obs_datetime_utc)}</div>
        <div><b>Emission (t/h):</b> ${fmt(
          p.emission_tph
        )} <span style="opacity:.75">± ${fmt(p.emission_unc_tph)}</span></div>
        <div><b>Sector:</b> ${fmt((p as { sector?: unknown }).sector)}</div>
        <div><b>Plume ID:</b> ${fmt(p.plume_id_src)}</div>
        <div style="opacity:.8;margin-top:4px;">[${lng.toFixed(5)}, ${lat.toFixed(
      5
    )}]</div>
      </div>`;
  };

  // Initialize map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        name: "Dark Minimal",
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm" }],
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
      },
      center: [134, -25],
      zoom: 3.5,
      minZoom: 2,
      maxZoom: 18,
    });

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.fitBounds(AUS_BOUNDS, { padding: 40 });

    map.on("load", () => {
      // Sources
      map.addSource("plumes", { type: "geojson", data: DATA_POINTS });
      map.addSource("clusters", { type: "geojson", data: DATA_CLUSTERS });

      // 1) Cluster polygons (FILL) — bottom
      map.addLayer({
        id: "cluster-fill",
        type: "fill",
        source: "clusters",
        paint: {
          "fill-color": "#00e6a8",
          "fill-opacity": 0.15,
        },
        layout: { visibility: "visible" },
      });

      // 2) Cluster polygons (OUTLINE)
      map.addLayer({
        id: "cluster-outline",
        type: "line",
        source: "clusters",
        paint: {
          "line-color": "#00e6a8",
          "line-width": 1.2,
          "line-opacity": 0.85,
        },
        layout: { visibility: "visible" },
      });

      // 3) Points — top
      map.addLayer({
        id: "plumes-circles",
        type: "circle",
        source: "plumes",
        paint: paintCircle,
        layout: { visibility: "visible" },
        filter: filterExpression,
      });

      // Popups for points only
      const popup = new Popup({ closeButton: true, closeOnClick: true, maxWidth: "320px" });
      map.on("click", "plumes-circles", (e) => {
        const f = e.features?.[0];
        if (!f) return;
        const p = (f.properties ?? {}) as Record<string, unknown>;
        popup
          .setLngLat(e.lngLat)
          .setHTML(makePopupHTML(p, e.lngLat.lng, e.lngLat.lat))
          .addTo(map);
      });

      map.on("mouseenter", "plumes-circles", () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", "plumes-circles", () => (map.getCanvas().style.cursor = ""));
    });

    mapRef.current = map;
    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
    // initialize once; dynamic updates handled below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to UI changes → update filters/visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (map.getLayer("plumes-circles")) {
      map.setFilter("plumes-circles", filterExpression as any);
      map.setLayoutProperty("plumes-circles", "visibility", showPoints ? "visible" : "none");
    }
    if (map.getLayer("cluster-fill")) {
      map.setLayoutProperty("cluster-fill", "visibility", showPolys ? "visible" : "none");
    }
    if (map.getLayer("cluster-outline")) {
      map.setLayoutProperty("cluster-outline", "visibility", showPolys ? "visible" : "none");
    }
  }, [filterExpression, showPoints, showPolys]);

  return (
    <div>
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={providers.KAYRROS}
            onChange={() => setProviders((p) => ({ ...p, KAYRROS: !p.KAYRROS }))}
          />
          <span className="opacity-85">Kayrros</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={providers.SRON}
            onChange={() => setProviders((p) => ({ ...p, SRON: !p.SRON }))}
          />
          <span className="opacity-85">SRON</span>
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={showPoints} onChange={() => setShowPoints((v) => !v)} />
          <span className="opacity-85">Points</span>
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={showPolys} onChange={() => setShowPolys((v) => !v)} />
          <span className="opacity-85">Cluster hulls</span>
        </label>
        <div className="flex items-center gap-2">
          <span className="opacity-70 text-sm">Min emission (t/h):</span>
          <input
            type="range"
            min={0}
            max={200}
            step={1}
            value={minEmission}
            onChange={(e) => setMinEmission(parseFloat(e.target.value))}
          />
          <span className="opacity-85 w-10 text-right">{minEmission}</span>
        </div>
      </div>

      {/* Map container */}
      <div
        ref={containerRef}
        style={{
          height: "70vh",
          width: "100%",
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 0 0 1px #2c2c33",
        }}
      />

      {/* Legend */}
      <div className="mt-3 text-sm opacity-85">
        <div className="inline-block mr-4">
          <span
            style={{
              display: "inline-block",
              width: 12,
              height: 12,
              background: "#4c9cff",
              borderRadius: 3,
              marginRight: 6,
            }}
          />
          Clusters
        </div>
        <div className="inline-block">
          <span
            style={{
              display: "inline-block",
              width: 12,
              height: 12,
              background: "#8a8fa3",
              borderRadius: 3,
              marginRight: 6,
            }}
          />
          Noise
        </div>
      </div>
      <p className="opacity-60 text-xs mt-2">
        Basemap © OpenStreetMap contributors · Data from your processed outputs
      </p>
    </div>
  );
}