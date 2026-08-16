import { useState } from "react";

interface AudioUploadProps {
  onResult: (result: any) => void;
  onFileSelected: (file: File) => void;
}

export default function AudioUpload({ onResult, onFileSelected }: AudioUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] || null;
    setFile(selected);
    if (selected) {
      onFileSelected(selected);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/api/v1/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Upload failed with status ${res.status}`);
      }
      const data = await res.json();
      onResult(data);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-2 border-dashed border-gray-700 p-6 rounded-lg mb-4 text-center">
      <input
        type="file"
        accept=".wav,.mp3,.flac,.ogg,.m4a"
        onChange={handleFileChange}
        className="mb-4 text-sm"
      />
      <br />
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-white font-semibold disabled:opacity-50 text-sm"
      >
        {loading ? "Analyzing..." : "Analyze Audio"}
      </button>
      {error && (
        <div className="bg-red-900/40 border border-red-500 text-red-200 p-3 rounded mt-4 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
