// app/page.tsx

"use client";

import dynamic from "next/dynamic";


const MapView = dynamic(() => import("./ui/MapView"), { ssr: false });

export default function HomePage() {
  return (
    <main className="min-h-screen bg-[#0b0b0e] text-[#e9e9ef]">
      <div className="mx-auto max-w-7xl p-4">
        <h1 className="text-2xl font-semibold mb-2">Methane Super-Emitter Clusters — Australia</h1>
        <p className="opacity-80 mb-4">
          Explore clustered detections, filter by provider and emission, and click points for details.
        </p>
        <MapView />
      </div>
    </main>
  );
}
