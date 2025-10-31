import React, { useState, useEffect, useRef } from "react";
import UploadPanel from "./components/UploadPanel";
import HighlightsPanel from "./components/HighlightsPanel";

export default function App() {
  const [videoURL, setVideoURL] = useState(null);
  const [highlights, setHighlights] = useState([]);
  const [statusMessages, setStatusMessages] = useState([]);
  const socketRef = useRef(null);
  const logRef = useRef(null);

  // ✅ Smooth scroll to bottom when new logs appear
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [statusMessages]);

  // ✅ Smart WebSocket connection (works for local + Render)
  const startWebSocket = () => {
    if (socketRef.current) socketRef.current.close();

    // auto-detect correct backend address
    const backendWs =
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1"
        ? "ws://127.0.0.1:8000/ws/goals" // local backend
        : `wss://${window.location.host.replace("www.", "")}/ws/goals`; // Render deploy

    console.log("🔗 Connecting WebSocket to:", backendWs);
    const socket = new WebSocket(backendWs);
    socketRef.current = socket;

    socket.onopen = () => {
      console.log("✅ Connected to backend WebSocket");
      setStatusMessages((prev) => [...prev, "Connected to backend"]);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          setStatusMessages((prev) =>
            [...prev, data.message].slice(-100) // keep last 100
          );
        } else if (data.type === "goal") {
          console.log("🎯 New goal:", data);
          setHighlights((prev) => {
            if (prev.some((h) => h.url === data.url)) return prev;
            return [...prev, data];
          });
        }
      } catch (err) {
        console.error("Message parse error:", err);
      }
    };

    socket.onerror = (err) => {
      console.warn("⚠️ WebSocket error:", err);
      setStatusMessages((prev) => [...prev, "WebSocket error occurred"]);
    };

    socket.onclose = (e) => {
      console.warn("❌ WebSocket closed:", e.reason);
      setStatusMessages((prev) => [...prev, "Disconnected from backend"]);
      // 🔁 Auto-reconnect after 3s (for Render restarts)
      setTimeout(startWebSocket, 3000);
    };
  };

  // ✅ Trigger WebSocket 1s after upload
  const handleVideoSelect = (url) => {
    setVideoURL(url);
    setHighlights([]);
    setStatusMessages([
      "Video uploaded successfully. Starting backend pipeline...",
    ]);
    setTimeout(startWebSocket, 1000);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      <header className="text-center py-4 border-b border-gray-700">
        <h1 className="text-3xl font-bold text-green-400">
          Football Goal Highlights
        </h1>
      </header>

      <div className="flex flex-1 p-4 gap-4 overflow-hidden">
        {/* Left: Upload + Playback */}
        <div className="w-1/2 bg-gray-800 p-4 rounded-lg flex flex-col">
          <UploadPanel onVideoSelect={handleVideoSelect} />
          {videoURL && (
            <video
              src={videoURL}
              controls
              autoPlay
              className="mt-4 w-full rounded-lg shadow-lg"
            />
          )}
        </div>

        {/* Right: Highlights + Status */}
        <div className="w-1/2 bg-gray-800 p-4 rounded-lg border border-gray-700 overflow-y-auto">
          <h2 className="text-2xl font-semibold text-yellow-400 mb-3">
            Goal Highlights
          </h2>
          <HighlightsPanel highlights={highlights} />

          <div className="mt-4 border-t border-gray-600 pt-3">
            <h3 className="text-lg font-semibold text-blue-400 mb-2">
              Live Pipeline Status
            </h3>
            <div
              ref={logRef}
              className="text-sm text-gray-300 bg-gray-900 rounded-md p-3 h-60 overflow-y-auto font-mono"
            >
              {statusMessages.length === 0 ? (
                <p className="italic text-gray-500">Waiting for updates...</p>
              ) : (
                statusMessages.map((msg, i) => (
                  <p key={i} className="leading-snug">
                    {msg}
                  </p>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
