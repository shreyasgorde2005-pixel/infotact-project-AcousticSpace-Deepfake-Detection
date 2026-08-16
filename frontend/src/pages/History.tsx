import { useEffect, useState } from "react";
import LoadingSpinner from "../components/LoadingSpinner";

interface HistoryEntry {
  timestamp: string;
  filename: string;
  is_fake: boolean;
  confidence: number;
}

export default function History() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/history")
      .then((res) => res.json())
      .then((data) => setEntries(data.history))
      .catch(() => setEntries([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading history..." />;

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Analysis History</h1>
      {entries.length === 0 ? (
        <p className="text-gray-400">No analyses yet - upload a file to get started.</p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b border-gray-700">
              <th className="py-2">Time</th>
              <th>File</th>
              <th>Result</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, i) => (
              <tr key={i} className="border-b border-gray-800">
                <td className="py-2">{new Date(entry.timestamp).toLocaleString()}</td>
                <td>{entry.filename}</td>
                <td className={entry.is_fake ? "text-red-400" : "text-green-400"}>
                  {entry.is_fake ? "Fake" : "Real"}
                </td>
                <td>{(entry.confidence * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
