import { apiUrl } from "./health";

export type GuidancePoint = { text: string; sources: string[] };

export type GuidanceSource = {
  id: string;
  title: string;
  publisher: string;
  url: string;
  scope: string;
  reviewed_on: string;
};

export type Guidance = {
  catalogue_version: string;
  general_advice: GuidancePoint[];
  storage: GuidancePoint[];
  checks: GuidancePoint[];
  review_status: "reviewed" | "awaiting_review" | "generic";
  safety_note: string;
  disclaimer: string;
  sources: GuidanceSource[];
};

export type ProduceIdentity = {
  id: string;
  label: string;
  source: "user_selected" | "model_predicted" | "unknown";
  model_version: string | null;
  probabilities: Record<string, number> | null;
};

export type Severity = {
  status: "not_assessed";
  label: null;
  explanation: string;
};

/** Everything that depends on the produce type and freshness class, but not on pixels. */
export type Described = {
  combined_label: string;
  produce: ProduceIdentity;
  severity: Severity;
  guidance: Guidance;
};

export type Prediction = Described & {
  label: "Fresh" | "Spoiled";
  label_index: number;
  probabilities: { Fresh: number; Spoiled: number };
  model_version: string;
  elapsed_ms: number;
  timing_scope: string;
  contract_version: string;
  freshness: {
    label: "Fresh" | "Spoiled";
    class_index: number;
    probabilities: { Fresh: number; Spoiled: number };
    model_version: string;
  };
};

export type ProduceOption = { id: string; label: string };

export type ProduceCatalogue = {
  catalogue_version: string;
  options: ProduceOption[];
  identification: { mode: "user_selected"; available: boolean; model_version: string | null };
};

export class PredictError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function readJson<T>(response: Response, fallback: string): Promise<T> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const message = (payload as { message?: string } | null)?.message;
    throw new PredictError(message ?? fallback, response.status);
  }
  return payload as T;
}

export async function predictImage(file: File, produce: string): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);
  body.append("produce", produce);
  const response = await fetch(apiUrl("/api/predict"), { method: "POST", body });
  return readJson<Prediction>(response, "The image could not be classified.");
}

export async function fetchProduceCatalogue(): Promise<ProduceCatalogue> {
  const response = await fetch(apiUrl("/api/produce"));
  return readJson<ProduceCatalogue>(response, "The produce list could not be loaded.");
}

/** Catalogue lookup only. The image is not sent and the model is not run. */
export async function fetchGuidance(produce: string, freshness: "Fresh" | "Spoiled"): Promise<Described> {
  const query = new URLSearchParams({ produce, freshness });
  const response = await fetch(apiUrl(`/api/guidance?${query.toString()}`));
  return readJson<Described>(response, "Guidance could not be loaded.");
}
