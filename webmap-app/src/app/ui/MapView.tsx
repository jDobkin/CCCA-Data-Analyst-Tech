/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { LngLatBoundsLike, Map, Popup } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

/**
 * Data
 * - `DATA_POINTS` should include SRON/KAYRROS/CARBON_MAPPER points with `provider`, `emission_tph`, etc.
 * - `DATA_CLUSTERS` is polygons (optional).
 */
const DATA_POINTS = "/data/au_plumes.geojson";
const DATA_CLUSTERS = "/data/clusters_au.geojson";

/** Australia bounds */
const AUS_BOUNDS: LngLatBoundsLike = [112, -44, 154, -9];

type Provider = "SRON" | "KAYRROS" | "CARBON_MAPPER";

const PROVIDER_COLORS: Record<Provider | "NOISE", string> = {
  SRON: "#ad73ff",          // purple
  KAYRROS: "#4c9cff",       // blue
  CARBON_MAPPER: "#00e6a8", // teal
  NOISE: "#8a8fa3",
};

export default function MapView() {
  const mapRef = useRef<Map | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [minEmission, setMinEmission] = useState<number>(0);
  const [providers, setProviders] = useState<Record<Provider, boolean>>({
    SRON: true,
    KAYRROS: true,
    CARBON_MAPPER: true,
  });
  const [showPolys, setShowPolys] = useState(true);
  const [showPoints, setShowPoints] = useState(true);

  // Common circle color expression by provider
  const circleColorByProvider: any = [
    "match",
    ["upcase", ["get", "provider"]],
    "KAYRROS", PROVIDER_COLORS.KAYRROS,
    "SRON", PROVIDER_COLORS.SRON,
    "CARBON_MAPPER", PROVIDER_COLORS.CARBON_MAPPER,
    PROVIDER_COLORS.NOISE,
  ];

  // Larger point style (SRON/KAYRROS)
  const paintCircleLarge: any = useMemo(
    () => ({
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["zoom"],
        4, 2.5,
        6, 4,
        8, 6,
        10, 8.5,
        12, 11,
      ],
      "circle-color": circleColorByProvider,
      "circle-opacity": 0.9,
      "circle-stroke-color": "#0b0b0e",
      "circle-stroke-width": 0.6,
    }),
    []
  );

  // Smaller point style (Carbon Mapper)
  const paintCircleSmall: any = useMemo(
    () => ({
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["zoom"],
        4, 1.5,
        6, 2.2,
        8, 3.2,
        10, 4.2,
        12, 5.5,
      ],
      "circle-color": circleColorByProvider,
      "circle-opacity": 0.85,
      "circle-stroke-color": "#0b0b0e",
      "circle-stroke-width": 0.5,
    }),
    []
  );

  // Build filter expression from UI state
  const baseFilter: any = useMemo(() => {
    const enabled = (Object.keys(providers) as Provider[]).filter((k) => providers[k]);
    const providerFilter =
      enabled.length === 0
        ? ["in", ["get", "provider"], "___none___"] // match nothing
        : ["in", ["upcase", ["get", "provider"]], ["literal", enabled.map((p) => p.toUpperCase())]];
    const emissionFilter = [">=", ["coalesce", ["get", "emission_tph"], 0], minEmission];
    return ["all", providerFilter, emissionFilter];
  }, [providers, minEmission]);

  // Sub-filters so layers can separate CM vs others
  const filterCM: any = useMemo(
    () => ["all", baseFilter, ["==", ["upcase", ["get", "provider"]], "CARBON_MAPPER"]],
    [baseFilter]
  );
  const filterOthers: any = useMemo(
    () => [
      "all",
      baseFilter,
      ["!=", ["upcase", ["get", "provider"]], "CARBON_MAPPER"],
    ],
    [baseFilter]
  );

  // Popup
  const makePopupHTML = (p: Record<string, unknown>, lng: number, lat: number) => {
    const fmt = (v: unknown) => (v === null || v === undefined || v === "" ? "—" : String(v));
    return `
      <div style="background:#14141a;color:#e9e9ef;border-radius:10px;border:1px solid #2c2c33;padding:10px 12px;font:13px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <div style="font-weight:700;letter-spacing:.2px;">${fmt(p.provider)}</div>
          <span style="opacity:.75">Cluster ${fmt(p.cluster_id)}</span>
        </div>
        <div><b>Datetime (UTC):</b> ${fmt(p.obs_datetime_utc)}</div>
        <div><b>Emission (t/h):</b> ${fmt(p.emission_tph)} <span style="opacity:.75">± ${fmt(
      p.emission_unc_tph
    )}</span></div>
        <div><b>Sector:</b> ${fmt((p as { sector?: unknown }).sector)}</div>
        <div><b>Plume ID:</b> ${fmt(p.plume_id_src)}</div>
        <div style="opacity:.7;margin-top:6px;">[${lng.toFixed(5)}, ${lat.toFixed(5)}]</div>
      </div>`;
  };

  // Init map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        name: "Dark Minimal",
        sources: {
          // Dark raster tiles (no token)
          dark: {
            type: "raster",
            tiles: ["https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution:
              '© OpenStreetMap contributors © CARTO',
          },
        },
        layers: [{ id: "dark", type: "raster", source: "dark" }],
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
      },
      center: [134, -25],
      zoom: 3.6,
      minZoom: 2,
      maxZoom: 18,
      hash: true,
    });

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.fitBounds(AUS_BOUNDS, { padding: 40 });

    map.on("load", () => {
      // Sources
      map.addSource("plumes", { type: "geojson", data: DATA_POINTS });
      map.addSource("clusters", { type: "geojson", data: DATA_CLUSTERS });

      // 1) Cluster polygons — fill
      map.addLayer({
        id: "cluster-fill",
        type: "fill",
        source: "clusters",
        paint: { "fill-color": "#00e6a8", "fill-opacity": 0.14 },
        layout: { visibility: "visible" },
      });

      // 2) Cluster polygons — outline
      map.addLayer({
        id: "cluster-outline",
        type: "line",
        source: "clusters",
        paint: {
          "line-color": "#00e6a8",
          "line-width": 1.1,
          "line-opacity": 0.85,
        },
        layout: { visibility: "visible" },
      });

      // 3a) Carbon Mapper points (smaller)
      map.addLayer({
        id: "plumes-cm",
        type: "circle",
        source: "plumes",
        paint: paintCircleSmall,
        layout: { visibility: "visible" },
        filter: filterCM,
      });

      // 3b) Other providers (SRON/KAYRROS) (larger)
      map.addLayer({
        id: "plumes-others",
        type: "circle",
        source: "plumes",
        paint: paintCircleLarge,
        layout: { visibility: "visible" },
        filter: filterOthers,
      });

      const popup = new Popup({ closeButton: true, closeOnClick: true, maxWidth: "340px" });

      const clickHandler = (e: maplibregl.MapLayerMouseEvent) => {
        const f = e.features?.[0];
        if (!f) return;
        const p = (f.properties ?? {}) as Record<string, unknown>;
        popup.setLngLat(e.lngLat).setHTML(makePopupHTML(p, e.lngLat.lng, e.lngLat.lat)).addTo(map);
      };

      map.on("click", "plumes-cm", clickHandler);
      map.on("click", "plumes-others", clickHandler);

      const cursorEnter = () => (map.getCanvas().style.cursor = "pointer");
      const cursorLeave = () => (map.getCanvas().style.cursor = "");
      map.on("mouseenter", "plumes-cm", cursorEnter);
      map.on("mouseleave", "plumes-cm", cursorLeave);
      map.on("mouseenter", "plumes-others", cursorEnter);
      map.on("mouseleave", "plumes-others", cursorLeave);
    });

    mapRef.current = map;
    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to UI changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Update filters
    if (map.getLayer("plumes-cm")) {
      map.setFilter("plumes-cm", filterCM as any);
      map.setLayoutProperty("plumes-cm", "visibility", showPoints ? "visible" : "none");
    }
    if (map.getLayer("plumes-others")) {
      map.setFilter("plumes-others", filterOthers as any);
      map.setLayoutProperty("plumes-others", "visibility", showPoints ? "visible" : "none");
    }

    // Toggle clusters
    if (map.getLayer("cluster-fill")) {
      map.setLayoutProperty("cluster-fill", "visibility", showPolys ? "visible" : "none");
    }
    if (map.getLayer("cluster-outline")) {
      map.setLayoutProperty("cluster-outline", "visibility", showPolys ? "visible" : "none");
    }
  }, [filterCM, filterOthers, showPoints, showPolys]);

  return (
    <div className="fixed inset-0">
      {/* MAP */}
      <div ref={containerRef} className="h-full w-full" />

      {/* FLOATING UI */}
      <div className="absolute top-4 left-4 z-10">
        <div
          className="backdrop-blur bg-[rgba(16,16,22,.6)] border border-[#2c2c33] rounded-xl p-3 text-[13px] text-[#e9e9ef] shadow-[0_0_0_1px_rgba(44,44,51,.6)]"
          style={{ minWidth: 260 }}
        >
          <div className="font-semibold tracking-wide mb-2 opacity-90">Filters</div>

          <div className="grid grid-cols-2 gap-x-4 gap-y-2">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={providers.KAYRROS}
                onChange={() => setProviders((p) => ({ ...p, KAYRROS: !p.KAYRROS }))}
              />
              <span>Kayrros</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={providers.SRON}
                onChange={() => setProviders((p) => ({ ...p, SRON: !p.SRON }))}
              />
              <span>SRON</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={providers.CARBON_MAPPER}
                onChange={() =>
                  setProviders((p) => ({ ...p, CARBON_MAPPER: !p.CARBON_MAPPER }))
                }
              />
              <span>Carbon Mapper</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showPolys}
                onChange={() => setShowPolys((v) => !v)}
              />
              <span>Cluster hulls</span>
            </label>
            <label className="flex items-center gap-2 col-span-2">
              <input
                type="checkbox"
                checked={showPoints}
                onChange={() => setShowPoints((v) => !v)}
              />
              <span>Points</span>
            </label>
          </div>

          <div className="mt-3">
            <div className="opacity-70 text-xs mb-1">Min emission (t/h): {minEmission}</div>
            <input
              type="range"
              min={0}
              max={200}
              step={1}
              value={minEmission}
              onChange={(e) => setMinEmission(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="mt-3 flex items-center gap-3 text-xs opacity-85">
            <span className="inline-flex items-center gap-2">
              <i
                style={{
                  display: "inline-block",
                  width: 10,
                  height: 10,
                  background: PROVIDER_COLORS.KAYRROS,
                  borderRadius: 3,
                }}
              />
              Kayrros
            </span>
            <span className="inline-flex items-center gap-2">
              <i
                style={{
                  display: "inline-block",
                  width: 10,
                  height: 10,
                  background: PROVIDER_COLORS.SRON,
                  borderRadius: 3,
                }}
              />
              SRON
            </span>
            <span className="inline-flex items-center gap-2">
              <i
                style={{
                  display: "inline-block",
                  width: 10,
                  height: 10,
                  background: PROVIDER_COLORS.CARBON_MAPPER,
                  borderRadius: 3,
                }}
              />
              Carbon Mapper
            </span>
          </div>
        </div>
      </div>

      {/* FOOTER CAPTION */}
      <div className="absolute bottom-3 left-4 z-10 text-xs text-[#8a8fa3]">
        Basemap © OpenStreetMap & CARTO · Colors by provider · CM points smaller
      </div>
    </div>
  );
}
