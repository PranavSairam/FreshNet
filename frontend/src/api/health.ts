const DEFAULT_BASE = "";

export type ModelHealth = {
  available: boolean;
  status: string;
  message: string;
  architecture: string;
  model_version: string;
};

export type HealthResponse = {
  status: string;
  service: string;
  version: string;
  model: ModelHealth;
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
