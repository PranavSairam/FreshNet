const DEFAULT_BASE = "";

export type ModelHealth = {
  available: boolean;
  status: string;
  message: string;
  architecture: string;
  model_version: string;
};

export type ProduceModelHealth = {
  available: boolean;
  status: string;
  model_version: string;
  confirmation_threshold: number | null;
};

export type HealthResponse = {
  status: string;
  service: string;
  version: string;
  model: ModelHealth;
  /** Separate produce-type model. Absent from older API versions. */
  produce_model?: ProduceModelHealth;
  database: {
    ok: boolean;
    path: string;
    message: string;
  };
};

export function apiUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? DEFAULT_BASE;
  return `${base}${path}`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(apiUrl("/api/health"));
  if (!response.ok) {
    throw new Error(`Health request failed (${response.status})`);
  }
  return (await response.json()) as HealthResponse;
}
