import { useEffect, useRef, useState } from "react";
import { ImagePlus, LoaderCircle, Trash2 } from "lucide-react";
import { PredictError, predictImage, type Prediction } from "../api/predict";

const MAX_BYTES = 8 * 1024 * 1024;

type Props = {
  modelReady: boolean;
  modelMessage: string;
};

export function AnalyzePanel({ modelReady, modelMessage }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const requestId = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [clientError, setClientError] = useState<string | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  function choose(next: File | null) {
    requestId.current += 1;
    setResult(null);
    setServerError(null);
    setBusy(false);
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    if (!next) {
      setFile(null);
      setPreview(null);
      setClientError(null);
      return;
    }
    const name = next.type.toLowerCase();
    if (name !== "image/jpeg" && name !== "image/png") {
      setFile(null);
      setPreview(null);
      setClientError("Use a static JPEG or PNG.");
      return;
    }
    if (next.size > MAX_BYTES) {
      setFile(null);
      setPreview(null);
      setClientError("That file is larger than 8 MB.");
      return;
    }
    setClientError(null);
    setFile(next);
    setPreview(URL.createObjectURL(next));
  }

  async function analyze() {
    if (!file || busy || !modelReady) {
      return;
    }
    const id = requestId.current;
    setBusy(true);
    setServerError(null);
    setResult(null);
    try {
      const prediction = await predictImage(file);
      if (id !== requestId.current) {
        return;
      }
      setResult(prediction);
    } catch (error) {
      if (id !== requestId.current) {
        return;
      }
      setServerError(error instanceof PredictError ? error.message : "The image could not be classified.");
    } finally {
      if (id === requestId.current) {
        setBusy(false);
      }
    }
  }

  return (
    <section id="analyze" className="scroll-mt-24 rounded-3xl border border-stone-200 bg-white p-6 shadow-sm">
      <h2 className="font-serif text-3xl text-stone-900">Analyze an image</h2>
      <p className="mt-2 max-w-2xl text-stone-700">
        JPEG or PNG, up to 8 MB. The file must be a still image without transparency, and the pixels
        themselves should already be upright. Animated files and EXIF-only rotation are rejected.
      </p>
      {!modelReady && (
        <p className="mt-4 rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-amber-950" role="status">
          {modelMessage || "The model is not available. Predictions stay disabled."}
        </p>
      )}
      <div
        className="mt-5 rounded-2xl border-2 border-dashed border-[#1f4d3a]/40 bg-[#f6f5f2] p-4"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          choose(event.dataTransfer.files?.[0] ?? null);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          className="sr-only"
          id="food-image"
          onChange={(event) => choose(event.target.files?.[0] ?? null)}
        />
        {preview ? (
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
            <img src={preview} alt="Selected upload preview" className="max-h-72 w-full rounded-xl object-contain sm:w-72" />
            <div className="flex flex-col gap-3">
              <p className="text-sm text-stone-600">{file?.name}</p>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className="rounded-full bg-[#1f4d3a] px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={analyze}
                  disabled={!modelReady || busy}
                >
                  {busy ? "Analyzing…" : "Analyze"}
                </button>
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-full border border-stone-300 px-4 py-2 text-sm"
                  onClick={() => {
                    choose(null);
                    if (inputRef.current) {
                      inputRef.current.value = "";
                    }
                  }}
                >
                  <Trash2 size={16} aria-hidden="true" />
                  Remove
                </button>
              </div>
            </div>
          </div>
        ) : (
          <label htmlFor="food-image" className="flex min-h-40 cursor-pointer flex-col items-center justify-center gap-2 text-center">
            <ImagePlus className="text-[#1f4d3a]" aria-hidden="true" />
            <span className="font-medium">Drop a fruit or vegetable image here, or choose a file</span>
            <span className="text-sm text-stone-600">JPEG or PNG</span>
          </label>
        )}
      </div>
      {clientError && (
        <p className="mt-3 text-sm text-red-800" role="alert">
          {clientError}
        </p>
      )}
      {serverError && (
        <p className="mt-3 text-sm text-red-800" role="alert">
          {serverError}
        </p>
      )}
      {busy && (
        <p className="mt-4 flex items-center gap-2 text-sm text-stone-700" role="status">
          <LoaderCircle className="animate-spin" size={16} aria-hidden="true" />
          Classifying the visible condition…
        </p>
      )}
      {result && (
        <div className="mt-6 rounded-2xl border border-stone-200 bg-[#f6f5f2] p-5">
          <p className="text-sm uppercase tracking-[0.14em] text-[#1f4d3a]">Visible condition</p>
          <p className="mt-1 font-serif text-5xl text-stone-900">{result.label}</p>
          <dl className="mt-4 grid gap-3 sm:grid-cols-2">
            <div>
              <dt className="text-sm text-stone-600">Fresh probability</dt>
              <dd className="text-lg">{formatProbability(result.probabilities.Fresh)}</dd>
            </div>
            <div>
              <dt className="text-sm text-stone-600">Spoiled probability</dt>
              <dd className="text-lg">{formatProbability(result.probabilities.Spoiled)}</dd>
            </div>
          </dl>
          <p className="mt-4 text-sm text-stone-700">
            These probabilities are model outputs, not a freshness percentage or a shelf-life estimate.
            Visual classification does not certify food safety.
          </p>
          <p className="mt-3 text-xs text-stone-500">
            Model {result.model_version}. Time {result.elapsed_ms.toFixed(0)} ms for decode, validation,
            preprocessing, and the model forward.
          </p>
        </div>
      )}
    </section>
  );
}

function formatProbability(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
