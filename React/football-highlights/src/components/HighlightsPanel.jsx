import React from "react";
import { Download } from "lucide-react";

export default function HighlightsPanel({ highlights }) {
  const validHighlights = (highlights || []).filter(
    (h) => h && h.url && h.url.trim() !== ""
  );

  if (validHighlights.length === 0) {
    return <p className="text-gray-400 italic">No highlights yet…</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      {validHighlights.map((h) => (
        <div
          key={h.id || h.clip_name}
          className="bg-gray-700 rounded-lg p-3 shadow-md hover:bg-gray-600 transition"
        >
          <div className="relative mb-2">
            <video
              src={h.url}
              controls
              className="w-full rounded-md shadow-md"
            />
          </div>

          <div className="flex justify-between items-center">
            <h3 className="font-semibold text-lg text-white">{h.title}</h3>
            <button
              className="hover:text-blue-400"
              onClick={() => window.open(h.url, "_blank")}
              title="Download highlight"
            >
              <Download size={20} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
