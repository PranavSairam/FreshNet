import type { HealthResponse } from "../api/health";

type Props = {
  health: HealthResponse | null;
  error: string | null;
  loading: boolean;
};

export function StatusBanner({ health, error, loading }: Props) {
  if (loading) {
    return (
      <p className="flex items-center gap-3 text-[var(--muted)]" role="status">
        <span className="dot-live h-2.5 w-2.5 rounded-full bg-[var(--muted)]" aria-hidden="true" />
        Checking model status…
      </p>
    );
  }

  if (error) {
    return (
      <div className="bg-[#e8c9bd] px-4 py-3 text-[#4a1d14]" role="alert">
        <p className="font-medium">Model unavailable</p>
        <p className="text-[length:var(--step-small)]">{error}</p>
      </div>
    );
  }

  if (health?.model.available === true) {
    return (
      <p className="flex items-start gap-3" role="status">
        <span className="dot-live mt-2 h-2.5 w-2.5 shrink-0 rounded-full bg-[var(--accent)]" aria-hidden="true" />
        <span>
          <span className="font-medium">Model ready.</span>{" "}
          <span className="text-[var(--muted)]">
            The API is running and{" "}
            <span className="[overflow-wrap:break-word]">{health.model.model_version}</span> is loaded.
          </span>
        </span>
      </p>
    );
  }

  return (
    <div className="bg-[#e6d3a8] px-4 py-3" role="status">
      <p className="font-medium">{health?.model.message ?? "Model unavailable"}</p>
      <p className="text-[length:var(--step-small)]">
        The API can be running while the model is unavailable. Predictions stay disabled until the
        deployment bundle loads.
      </p>
    </div>
  );
}
