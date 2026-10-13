import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const EMPTY_SERIES: Array<{ epoch: number; accuracy: number }> = [];

export function MetricsPlaceholder() {
  return (
    <section className="rounded-3xl border border-stone-300 bg-white p-6 shadow-sm">
      <h2 className="font-serif text-xl text-stone-900">Training metrics</h2>
      <p className="mt-1 text-sm text-stone-600">
        Charts will read `artifacts/metrics/history.json` after a real training run. This
        plot is empty on purpose.
      </p>
      <div className="mt-4 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={EMPTY_SERIES}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Line type="monotone" dataKey="accuracy" stroke="#1f4d3a" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
