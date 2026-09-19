import { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line, Doughnut, Bar, Pie } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler
);

export default function MetricsPage() {
  const [stats, setStats] = useState({ events_last_hour: 0, events_today: 0 });
  const [metrics, setMetrics] = useState(null);
  const [perf, setPerf] = useState(null);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [sRes, mRes, pRes] = await Promise.all([
          fetch("/api/stats"),
          fetch("/api/metrics"),
          fetch("/api/perf"),
        ]);
        if (sRes.ok) setStats(await sRes.json());
        if (mRes.ok) setMetrics(await mRes.json());
        if (pRes.ok) setPerf(await pRes.json());
      } catch (err) {
        console.error("Metrics fetch error", err);
      }
    };
    fetchAll();
    const id = setInterval(fetchAll, 8000);
    return () => clearInterval(id);
  }, []);

  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#9ca3af", font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#6b7280" }, grid: { color: "#1f2937" } },
      y: {
        ticks: { color: "#6b7280", stepSize: 1 },
        grid: { color: "#1f2937" },
        beginAtZero: true,
      },
    },
  };

  const doughnutOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: "bottom", labels: { color: "#9ca3af", font: { size: 11 } } } },
  };

  // ── Chart data ─────────────────────────────────────
  const eventsLineData = metrics
    ? {
      labels: metrics.events_by_hour.map((b) => `-${24 - b.hour}h`),
      datasets: [
        {
          label: "Events",
          data: metrics.events_by_hour.map((b) => b.count),
          borderColor: "#818cf8",
          backgroundColor: "rgba(129,140,248,0.15)",
          fill: true,
          tension: 0.4,
          pointRadius: 2,
        },
      ],
    }
    : null;

  const actionDoughnutData = metrics
    ? {
      labels: metrics.detection_distribution.map((d) => d.action),
      datasets: [
        {
          data: metrics.detection_distribution.map((d) => d.count),
          backgroundColor: ["#34d399", "#60a5fa", "#fbbf24", "#f87171", "#a78bfa", "#f472b6", "#22d3ee", "#94a3b8"],
          borderWidth: 0,
        },
      ],
    }
    : null;

  const confBarData = metrics
    ? {
      labels: metrics.confidence_histogram.map((b) => b.range),
      datasets: [
        {
          label: "Detections",
          data: metrics.confidence_histogram.map((b) => b.count),
          backgroundColor: "#818cf8",
          borderRadius: 4,
        },
      ],
    }
    : null;

  const riskPieData = metrics
    ? {
      labels: metrics.risk_distribution.map((r) => r.level),
      datasets: [
        {
          data: metrics.risk_distribution.map((r) => r.count),
          backgroundColor: ["#34d399", "#fbbf24", "#f87171"],
          borderWidth: 0,
        },
      ],
    }
    : null;

  const card =
    "bg-gray-800/60 backdrop-blur border border-gray-700 rounded-xl p-4";

  return (
    <div className="space-y-6">
      {/* ── Header ─────────────────────────── */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          Performance Metrics
        </h2>
        <span className="text-xs text-gray-500">Auto-refreshes every 8s</span>
      </div>

      {/* ── Stat Cards ─────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className={card}>
          <div className="text-xs text-gray-400 mb-1">Events Last Hour</div>
          <div className="text-2xl font-bold text-white">
            {stats.events_last_hour}
          </div>
        </div>
        <div className={card}>
          <div className="text-xs text-gray-400 mb-1">Events Today</div>
          <div className="text-2xl font-bold text-white">
            {stats.events_today}
          </div>
        </div>
        <div className={card}>
          <div className="text-xs text-gray-400 mb-1">Events (24h)</div>
          <div className="text-2xl font-bold text-white">
            {metrics?.total_events_24h ?? "—"}
          </div>
        </div>
        <div className={card}>
          <div className="text-xs text-gray-400 mb-1" title={metrics?.face_match_rate_note}>
            Faces matched (24h)
          </div>
          <div className="text-2xl font-bold text-emerald-400">
            {metrics?.face_observations ? `${metrics.face_match_rate}%` : "—"}
          </div>
          <div className="text-[10px] text-gray-500">
            {metrics?.face_observations ?? 0} visible faces · not an accuracy figure
          </div>
        </div>
      </div>

      {/* ── Measured performance ───────────── */}
      {perf && <PerfPanel perf={perf} card={card} />}

      {/* ── Charts Grid ────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Events over time */}
        <div className={card}>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
            📈 Events Over Time (24h)
          </h3>
          <div style={{ height: 220 }}>
            {eventsLineData ? (
              <Line data={eventsLineData} options={chartOpts} />
            ) : (
              <p className="text-gray-500 text-sm">Loading…</p>
            )}
          </div>
        </div>

        {/* Detection distribution */}
        <div className={card}>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
            🧠 Event Triggers
          </h3>
          <div style={{ height: 220 }}>
            {actionDoughnutData ? (
              <Doughnut data={actionDoughnutData} options={doughnutOpts} />
            ) : (
              <p className="text-gray-500 text-sm">Loading…</p>
            )}
          </div>
        </div>

        {/* Confidence histogram */}
        <div className={card}>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
            📊 Confidence Histogram
          </h3>
          <div style={{ height: 220 }}>
            {confBarData ? (
              <Bar data={confBarData} options={chartOpts} />
            ) : (
              <p className="text-gray-500 text-sm">Loading…</p>
            )}
          </div>
        </div>

        {/* Risk distribution */}
        <div className={card}>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
            🔒 Risk Level Distribution
          </h3>
          <div style={{ height: 220 }}>
            {riskPieData ? (
              <Pie data={riskPieData} options={doughnutOpts} />
            ) : (
              <p className="text-gray-500 text-sm">Loading…</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ms(v) {
  return v == null ? "—" : `${v.toFixed(v < 10 ? 1 : 0)} ms`;
}

function PerfPanel({ perf, card }) {
  const cams = Object.entries(perf.cameras || {});
  const models = [
    ["Pose (YOLOv8n)", perf.device, true],
    ["Activity (SlowFast)", perf.activity?.device, perf.activity?.ready, perf.activity?.last_inference_ms],
    ["Depth (MiDaS)", perf.depth?.device, perf.depth?.ready, perf.depth?.last_inference_ms],
    ["Faces", perf.face?.backend, perf.face?.available],
  ];
  return (
    <div className={card}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">⚙ Measured performance</h3>
        <span className="text-xs text-gray-500">
          {perf.device === "cuda" ? `GPU · ${perf.gpu}` : "CPU"} · torch {perf.torch}
        </span>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <table className="kv-table">
          <thead>
            <tr className="text-gray-500">
              <td>Camera</td><td>Capture</td><td>Processed</td><td>Frame → result (p50 / p95)</td>
            </tr>
          </thead>
          <tbody>
            {cams.map(([id, c]) => (
              <tr key={id}>
                <td>{id}</td>
                <td>{c.capture_fps ?? "—"} fps</td>
                <td>{c.pipeline_fps ?? "—"} fps</td>
                <td>{ms(c.latency?.frame_to_result?.p50_ms)} / {ms(c.latency?.frame_to_result?.p95_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <table className="kv-table">
          <tbody>
            {models.map(([name, dev, ready, last]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>
                  {ready ? (dev || "on") : "off"}
                  {last != null ? ` · ${ms(last)}` : ""}
                </td>
              </tr>
            ))}
            {cams[0] && Object.entries(cams[0][1].stages || {}).filter(([, v]) => v).map(([k, v]) => (
              <tr key={k}>
                <td className="text-gray-500">stage: {k}</td>
                <td>{ms(v.mean_ms)} mean</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
