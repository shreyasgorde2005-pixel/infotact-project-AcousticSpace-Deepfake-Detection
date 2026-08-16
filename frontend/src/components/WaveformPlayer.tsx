import { useEffect, useState } from "react";

interface FlaggedSegment {
  start: number;
  end: number;
}

interface WaveformPlayerProps {
  file: File | null;
  flaggedSegments?: FlaggedSegment[];
}

export default function WaveformPlayer({ file, flaggedSegments = [] }: WaveformPlayerProps) {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setAudioUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  if (!file) return null;

  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700 mb-4">
      <h3 className="text-md font-semibold mb-2">Audio Preview</h3>
      {audioUrl && (
        <audio id="audio-player-element" controls src={audioUrl} className="w-full mb-2" />
      )}
      {flaggedSegments.length > 0 && (
        <div className="mt-2 text-xs text-red-400">
          <strong>Flagged Segments:</strong>
          <ul className="list-disc list-inside mt-1">
            {flaggedSegments.map((seg, idx) => (
              <li key={idx}>
                {seg.start.toFixed(2)}s - {seg.end.toFixed(2)}s
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
