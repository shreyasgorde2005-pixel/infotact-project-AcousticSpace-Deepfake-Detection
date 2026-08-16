interface AcousticComparison {
  voice_rt60?: number | null;
  voice_drr?: number | null;
  bg_rt60?: number | null;
  bg_drr?: number | null;
}

interface ResultsPanelProps {
  result?: {
    filename?: string;
    is_fake?: boolean;
    confidence?: number;
    rir_mismatch_score?: number | null;
    breathing_score?: number | null;
    inference_time_ms?: number | null;
    acoustic_comparison?: AcousticComparison | null;
  } | null;
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const percentage = Math.round(value * 100);
  return (
    <div className="my-2">
      <div className="flex justify-between text-sm font-medium mb-1">
        <span>{label}</span>
        <span>{percentage}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2.5">
        <div
          className="bg-blue-500 h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
        />
      </div>
    </div>
  );
}

function ComparisonBar({ label, voiceValue, bgValue, unit }: {
  label: string; voiceValue: number;
  bgValue: number; unit: string
}) {
  const max = Math.max(Math.abs(voiceValue), Math.abs(bgValue), 1);
  return (
    <div className="mb-3">
      <div className="text-sm mb-1">{label}</div>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs w-20">Voice</span>
        <div className="flex-1 bg-gray-700 rounded h-2">
          <div className="h-2 rounded bg-blue-500" style={{
            width: `${(Math.abs(voiceValue) / max) * 100}%`
          }} />
        </div>
        <span className="text-xs w-16 text-right">{voiceValue.toFixed(2)}{unit}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs w-20">Background</span>
        <div className="flex-1 bg-gray-700 rounded h-2">
          <div className="h-2 rounded bg-orange-500" style={{
            width: `${(Math.abs(bgValue) / max) * 100}%`
          }} />
        </div>
        <span className="text-xs w-16 text-right">{bgValue.toFixed(2)}{unit}</span>
      </div>
    </div>
  );
}

export default function ResultsPanel({ result }: ResultsPanelProps) {
  if (!result) return null;

  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700">
      <h2 className="text-xl font-semibold mb-2">Analysis Result</h2>
      <p className="text-sm mb-1">
        <strong>File:</strong> {result.filename || "Uploaded File"}
      </p>
      <p className="text-sm mb-1">
        <strong>Prediction:</strong>{" "}
        <span className={result.is_fake ? "text-red-400 font-bold" : "text-green-400 font-bold"}>
          {result.is_fake ? "Fake (Synthetic)" : "Real (Authentic)"}
        </span>
      </p>
      {result.inference_time_ms != null && (
        <p className="text-xs text-gray-500 mb-3">
          Analyzed in {result.inference_time_ms.toFixed(0)} ms
        </p>
      )}
      {result.confidence !== undefined && (
        <p className="text-sm mb-2">
          <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
        </p>
      )}

      {result.rir_mismatch_score !== undefined && result.rir_mismatch_score !== null && (
        <ScoreBar label="RIR mismatch score" value={result.rir_mismatch_score} />
      )}
      {result.breathing_score !== undefined && result.breathing_score !== null && (
        <ScoreBar label="Breathing pattern score" value={result.breathing_score} />
      )}
      <p className="text-xs text-gray-500 -mt-1 mb-2">
        Higher means the voice's room echo looks more artificial.
      </p>

      {result.acoustic_comparison && (
        <div className="mt-4 border-t border-gray-700 pt-3">
          <h3 className="text-sm font-semibold mb-2">Acoustic comparison (voice vs. background)</h3>
          {result.acoustic_comparison.voice_rt60 != null && result.acoustic_comparison.bg_rt60 != null && (
            <ComparisonBar label="RT60 (echo decay time)" voiceValue={result.acoustic_comparison.voice_rt60}
              bgValue={result.acoustic_comparison.bg_rt60} unit="s" />
          )}
          {result.acoustic_comparison.voice_drr != null && result.acoustic_comparison.bg_drr != null && (
            <ComparisonBar label="DRR (direct-to-reverberant ratio)" voiceValue={result.acoustic_comparison.voice_drr} bgValue={result.acoustic_comparison.bg_drr} unit="dB" />
          )}
        </div>
      )}
    </div>
  );
}
