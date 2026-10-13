import { AlertTriangle, CheckCircle2, LoaderCircle } from "lucide-react";
import type { HealthResponse } from "../api/health";

type Props = {
  health: HealthResponse | null;
  error: string | null;
  loading: boolean;
};

export function StatusBanner({ health, error, loading }: Props) {
  if (loading) {
    return (
      <div
        className="flex items-center gap-3 rounded-2xl border border-stone-300 bg-white px-4 py-3 text-stone-700"
        role="status"
      >
        <LoaderCircle className="animate-spin" size={18} aria-hidden="true" />
        Checking model status…
      </div>
    );
  }

  if (error) {
    return (
      <div
        className="flex items-start gap-3 rounded-2xl border border-red-300 bg-red-50 px-4 py-3 text-red-950"
        role="alert"
      >
        <AlertTriangle size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
        <div>
          <p className="font-semibold">Model unavailable</p>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    );
  }

  const available = health?.model.available === true;
  const message = health?.model.message ?? "Model unavailable";

  if (available) {
    return (
      <div className="flex items-start gap-3 rounded-2xl border border-emerald-300 bg-emerald-50 px-4 py-3 text-emerald-950">
        <CheckCircle2 size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
        <div>
          <p className="font-semibold">Model ready</p>
          <p className="text-sm">The API is running and {health?.model.model_version} is loaded.</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex items-start gap-3 rounded-2xl border border-amber-400 bg-amber-50 px-4 py-3 text-amber-950"
      role="status"
    >
      <AlertTriangle size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <div>
        <p className="font-semibold">{message}</p>
        <p className="text-sm">
          The API can be running while the model is unavailable. Predictions stay disabled until the
          deployment bundle loads.
        </p>
      </div>
    </div>
  );
}
