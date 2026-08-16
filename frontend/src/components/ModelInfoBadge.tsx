import { useEffect, useState } from "react";
export default function ModelInfoBadge() {
    const [info, setInfo] = useState<{ model_version: string } | null>(null);
    useEffect(() => {
        fetch("http://localhost:8000/model-info")
            .then((res) => res.json())
            .then(setInfo)
            .catch(() => setInfo(null));
    }, []);
    if (!info) return null;
    return (
        <span className="inline-block px-2 py-1 rounded text-xs bg-gray-700 text-gray-200">
            Model: {info.model_version}
        </span>
    );
}