import React, { useState, useEffect, useRef } from "react";
import UploadPanel from "./components/UploadPanel";
import HighlightsPanel from "./components/HighlightsPanel";

export default function App() {
  const [videoURL, setVideoURL] = useState(null);
  const [highlights, setHighlights] = useState([]);
  const [statusMessages, setStatusMessages] = useState([]);
  const socketRef = useRef(null);
  const logRef = useRef(null);

  // ✅ Smooth scroll to bottom of logs when new updates arrive
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [statusMessages]);

  // ✅ WebSocket connection
  const startWebSocket = () => {
    if (socketRef.current) socketRef.current.close(); // cleanup old one

    const socket = new WebSocket("ws://127.0.0.1:8000/ws/goals");
    socketRef.current = socket;

    socket.onopen = () => {
      console.log("✅ Connected to backend WebSocket");
      setStatusMessages((prev) => [...prev, "Connected to backend"]);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "status") {
          // ✅ Keep only last 100 to avoid lag
          setStatusMessages((prev) => {
            const msgs = [...prev, data.message];
            return msgs.slice(-100);
          });
        } else if (data.type === "goal") {
          console.log("🎯 New goal:", data);
          
          setHighlights((prev) => {
            if (prev.some(h => h.url === data.url)) return prev;
            return [...prev, data]
        });
        }
      } catch (err) {
        console.error("Message parse error:", err);
      }
    };

    socket.onclose = () => {
      console.log("❌ WebSocket disconnected");
      setStatusMessages((prev) => [...prev, "Disconnected from backend"]);
    };

    socket.onerror = (err) => {
      console.warn("⚠️ WebSocket error:", err);
      setStatusMessages((prev) => [...prev, "WebSocket error occurred"]);
    };
  };

  // ✅ Start WebSocket 1 second after upload (backend ready)
  const handleVideoSelect = (url) => {
    setVideoURL(url);
    setHighlights([]);
    setStatusMessages(["Video uploaded successfully. Starting backend pipeline..."]);
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
        {/* Left panel: Upload + playback */}
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

        {/* Right panel: Highlights + Live status */}
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
