import { useState, useCallback } from "react";
export function useWebSocketPredict() {
    const [stage, setStage] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const analyze = useCallback((file: File) => {
        setError(null);
        setResult(null);
        const ws = new WebSocket("ws://localhost:8000/ws/predict");

        ws.onopen = () => {
            file.arrayBuffer().then((buffer) => ws.send(buffer));
        };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setStage(data.stage);
            if (data.stage === "error") setError(data.message);
            if (data.stage === "done") {
                setResult(data.result);
                ws.close();
            }
        };
        ws.onerror = () => setError("WebSocket connection failed");
    }, []);
    return { stage, result, error, analyze };
}