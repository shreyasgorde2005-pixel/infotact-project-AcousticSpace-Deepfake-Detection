import { useState } from "react";
import AudioUpload from "../components/AudioUpload";
import WaveformPlayer from "../components/WaveformPlayer";
import ResultsPanel from "../components/ResultsPanel";
import { useWebSocketPredict } from "../hooks/useWebSocketPredict";

export default function Upload() {
  const [result, setResult] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [useLive, setUseLive] = useState(false);
  const { stage, result: wsResult, error: wsError, analyze } = useWebSocketPredict();

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    if (useLive) analyze(file);
  };
  const displayedResult = useLive ? wsResult : result;

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Upload Audio for Analysis</h1>
      <label className="flex items-center gap-2 text-sm mb-3">
        <input 
          type="checkbox" 
          checked={useLive} 
          onChange={(e) => setUseLive(e.target.checked)} 
        />
        Use live progress (WebSocket)
      </label>
      <AudioUpload onResult={setResult} onFileSelected={handleFileSelected} />
      {useLive && stage && <p className="text-sm text-blue-400 mt-2">Stage: {stage}</p>}
      {useLive && wsError && <p className="text-red-400 text-sm mt-2">{wsError}</p>}
      <WaveformPlayer file={selectedFile} flaggedSegments={displayedResult?.flagged_segments || []} />
      <ResultsPanel result={displayedResult} />
    </div>
  );
}
