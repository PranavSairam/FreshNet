import { apiUrl } from "./health";

export type Prediction = {
  label: "Fresh" | "Spoiled";
  label_index: number;
  probabilities: { Fresh: number; Spoiled: number };
  model_version: string;
  elapsed_ms: number;
  timing_scope: string;
};

export class PredictError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function predictImage(file: File): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(apiUrl("/api/predict"), { method: "POST", body });
  const payload = (await response.json()) as { message?: string; code?: string };
  if (!response.ok) {
    throw new PredictError(payload.message ?? "The image could not be classified.", response.status);
  }
  return payload as Prediction;
}
