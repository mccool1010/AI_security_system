import { useEffect, useState } from "react";

export default function CamerasPage() {
    const [cameras, setCameras] = useState([]);
    const [showAdd, setShowAdd] = useState(false);
    const [newName, setNewName] = useState("");
    const [newSource, setNewSource] = useState("");
    const [loading, setLoading] = useState(false);
    const [testResult, setTestResult] = useState({});

    const fetchCameras = async () => {
        try {
            const res = await fetch("/api/cameras");
            const data = await res.json();
            setCameras(data);
        } catch (err) {
            console.error("Failed to fetch cameras", err);
        }
    };

    useEffect(() => {
        fetchCameras();
        const id = setInterval(fetchCameras, 3000);
        return () => clearInterval(id);
    }, []);

    const addCamera = async () => {
        if (!newName.trim() || !newSource.trim()) return;
        setLoading(true);
        try {
            const res = await fetch("/api/cameras", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: newName.trim(), source: newSource.trim() }),
            });
            if (res.ok) {
                setNewName("");
                setNewSource("");
                setShowAdd(false);
                fetchCameras();
            }
        } catch (err) {
            console.error(err);
        }
        setLoading(false);
    };

    const removeCamera = async (id) => {
        try {
            await fetch(`/api/cameras/${id}`, { method: "DELETE" });
            fetchCameras();
        } catch (err) {
            console.error(err);
        }
    };

    const testCamera = async (id) => {
        setTestResult((prev) => ({ ...prev, [id]: "testing..." }));
        try {
            const res = await fetch(`/api/cameras/${id}/test`, { method: "POST" });
            const data = await res.json();
            setTestResult((prev) => ({
                ...prev,
                [id]: data.ok ? "✅ Connected" : `❌ ${data.status || "Failed"}`,
            }));
        } catch (err) {
            setTestResult((prev) => ({ ...prev, [id]: "❌ Error" }));
        }
        setTimeout(() => {
            setTestResult((prev) => {
                const copy = { ...prev };
                delete copy[id];
                return copy;
            });
        }, 3000);
    };

    const statusColor = (s) => {
        if (s === "online") return "bg-emerald-500";
        if (s === "paused") return "bg-amber-500";
        return "bg-red-500";
    };

    return (
        <div>
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Camera Management</h2>
                <button
                    onClick={() => setShowAdd((s) => !s)}
                    className="px-4 py-2 rounded-lg font-medium text-sm transition-all"
                    style={{
                        background: showAdd
                            ? "rgba(239,68,68,0.2)"
                            : "linear-gradient(135deg, #6366f1, #8b5cf6)",
                        color: showAdd ? "#f87171" : "#fff",
                        border: showAdd ? "1px solid rgba(239,68,68,0.3)" : "none",
                    }}
                >
                    {showAdd ? "✕ Cancel" : "+ Add Camera"}
                </button>
            </div>

            {/* Add Camera Form */}
            {showAdd && (
                <div className="bg-gray-800/80 backdrop-blur border border-gray-700 rounded-xl p-5 mb-6 animate-in">
                    <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
                        Add IP / Network Camera
                    </h3>
                    <p className="text-xs text-gray-500 mb-4">
                        Use an MJPEG stream URL from an IP camera app (e.g. IP Webcam for
                        Android: <code>http://192.168.x.x:8080/video</code>) or an RTSP
                        URL.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                        <div>
                            <label className="text-xs text-gray-400 block mb-1">
                                Camera Name
                            </label>
                            <input
                                type="text"
                                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                                placeholder="e.g. Front Door"
                                value={newName}
                                onChange={(e) => setNewName(e.target.value)}
                            />
                        </div>
                        <div>
                            <label className="text-xs text-gray-400 block mb-1">
                                Stream URL or Device Index
                            </label>
                            <input
                                type="text"
                                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                                placeholder="e.g. http://192.168.1.100:8080/video"
                                value={newSource}
                                onChange={(e) => setNewSource(e.target.value)}
                            />
                        </div>
                    </div>
                    <button
                        onClick={addCamera}
                        disabled={loading || !newName.trim() || !newSource.trim()}
                        className="px-5 py-2 rounded-lg text-sm font-medium text-white disabled:opacity-40"
                        style={{
                            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                        }}
                    >
                        {loading ? "Adding..." : "Add Camera"}
                    </button>
                </div>
            )}

            {/* Camera List */}
            <div className="space-y-3">
                {cameras.length === 0 && (
                    <p className="text-gray-500 text-sm">No cameras registered.</p>
                )}
                {cameras.map((cam) => (
                    <div
                        key={cam.id}
                        className="bg-gray-800/60 border border-gray-700 rounded-xl p-4 flex items-center gap-4 hover:border-gray-600 transition-colors"
                    >
                        {/* Preview thumbnail */}
                        <div className="w-32 h-20 bg-black rounded-lg overflow-hidden flex-shrink-0 relative">
                            {cam.status === "online" ? (
                                <img
                                    src={`/api/cameras/${cam.id}/feed`}
                                    alt={cam.name}
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs">
                                    {cam.status === "paused" ? "⏸ Paused" : "📷 Offline"}
                                </div>
                            )}
                        </div>

                        {/* Info */}
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                                <span
                                    className={`w-2 h-2 rounded-full ${statusColor(cam.status)}`}
                                />
                                <h4 className="font-medium text-sm truncate">{cam.name}</h4>
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-400 uppercase">
                                    {cam.type}
                                </span>
                            </div>
                            <p className="text-xs text-gray-500 truncate">{cam.source}</p>
                            {testResult[cam.id] && (
                                <p className="text-xs mt-1">{testResult[cam.id]}</p>
                            )}
                        </div>

                        {/* Actions */}
                        <div className="flex gap-2 flex-shrink-0">
                            <button
                                onClick={() => testCamera(cam.id)}
                                className="px-3 py-1.5 text-xs rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
                            >
                                Test
                            </button>
                            {cam.id !== "cam_local" && (
                                <button
                                    onClick={() => removeCamera(cam.id)}
                                    className="px-3 py-1.5 text-xs rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 transition-colors"
                                >
                                    Remove
                                </button>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
