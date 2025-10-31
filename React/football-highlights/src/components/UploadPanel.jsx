import React, { useState } from "react";

export default function UploadPanel({ onVideoSelect }) {
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [videoName, setVideoName] = useState("");

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setVideoName(file.name);

    // ✅ Use backend-served path for stable playback
    const backendBase =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : `https://${window.location.host.replace("www.", "")}`;
const backendUrl = `${backendBase}/outputs/uploads/${file.name}`;
onVideoSelect(backendUrl);

const formData = new FormData();
formData.append("file", file);

try {
  console.log("Backend base URL:", backendBase);

  const res = await fetch(`${backendBase}/process_video`, {
    method: "POST",
    body: formData,
  });


      const result = await res.json();
      console.log("Backend response:", result);

      if (result.status === "ok") {
        setProcessing(true);
      } else {
        alert("Backend error: " + (result.message || "Unknown error"));
      }
    } catch (err) {
      console.error("Upload error:", err);
      alert("Upload failed. Check backend connection.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-600 rounded-lg p-6 text-center">
      <p className="mb-2 text-gray-300">
        {uploading
          ? "Uploading video..."
          : processing
          ? "Processing highlights..."
          : "Upload your football match video"}
      </p>

      <input
        type="file"
        accept="video/*"
        onChange={handleUpload}
        disabled={uploading}
        className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 
                   file:rounded-md file:border-0 file:text-sm file:font-semibold
                   file:bg-green-500 file:text-white hover:file:bg-green-600"
      />

      {videoName && (
        <p className="text-gray-400 text-xs mt-2">
          Selected: <span className="text-white">{videoName}</span>
        </p>
      )}
    </div>
  );
}
