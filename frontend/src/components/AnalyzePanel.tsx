import { useEffect, useRef, useState } from "react";
import {
  fetchGuidance,
  fetchProduceCatalogue,
  PredictError,
  predictImage,
  type Described,
  type Prediction,
  type ProduceOption,
} from "../api/predict";

const UNKNOWN_OPTION: ProduceOption = { id: "unknown", label: "Unknown / other" };

const MAX_BYTES = 8 * 1024 * 1024;

/**
 * The detail branch reads five 160 x 160 patches from the 384 x 384 resize:
 * the four corners and the centre. These boxes are those regions as shares of
 * the square. The preview is drawn at the same square shape the model sees.
 */
const PATCHES = [
  { left: 0, top: 0 },
  { left: 58.3333, top: 0 },
  { left: 0, top: 58.3333 },
  { left: 58.3333, top: 58.3333 },
  { left: 29.1667, top: 29.1667 },
] as const;
const PATCH_SIZE = 41.6667;

type Props = {
  modelReady: boolean;
  modelMessage: string;
  checking?: boolean;
};

export function AnalyzePanel({ modelReady, modelMessage, checking = false }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const requestId = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [clientError, setClientError] = useState<string | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [dragging, setDragging] = useState(false);
  const [options, setOptions] = useState<ProduceOption[]>([UNKNOWN_OPTION]);
  const [produceId, setProduceId] = useState(UNKNOWN_OPTION.id);
  const [described, setDescribed] = useState<Described | null>(null);
  const [guidanceBusy, setGuidanceBusy] = useState(false);
  const [guidanceError, setGuidanceError] = useState<string | null>(null);
  const [automatic, setAutomatic] = useState(false);
  const [supportedIds, setSupportedIds] = useState<string[]>([]);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  useEffect(() => {
    let live = true;
    fetchProduceCatalogue()
      .then((catalogue) => {
        if (!live) {
          return;
        }
        if (catalogue.options.length > 0) {
          setOptions(catalogue.options);
        }
        setAutomatic(catalogue.identification.available);
        setSupportedIds(catalogue.identification.supported_ids ?? []);
      })
      .catch(() => {
        // Without the list only "Unknown / other" is offered. Freshness still works.
      });
    return () => {
      live = false;
    };
  }, []);

  function choose(next: File | null) {
    // A new image invalidates everything tied to the old one: the freshness
    // result, the guidance, any in-flight request, and the produce selection.
    requestId.current += 1;
    setResult(null);
    setDescribed(null);
    setGuidanceBusy(false);
    setGuidanceError(null);
    setProduceId(UNKNOWN_OPTION.id);
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
    setGuidanceError(null);
    setResult(null);
    setDescribed(null);
    if (automatic) {
      setProduceId(UNKNOWN_OPTION.id);
    }
    try {
      // With automatic identification the type is never sent up front: the produce
      // model reads the photo, and the user corrects afterwards if needed.
      const prediction = await predictImage(file, automatic ? "unknown" : produceId);
      if (id !== requestId.current) {
        return;
      }
      setResult(prediction);
      setDescribed(prediction);
      // The control starts from what the produce model said. Low confidence stays Unknown.
      setProduceId(prediction.produce.id);
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

  /**
   * Changing the produce type after a result only swaps the label and advice.
   * It reads the guidance catalogue; the image is not sent again, the model is
   * not run, and the freshness numbers stay exactly as they were.
   */
  async function changeProduce(next: string) {
    const previous = produceId;
    setProduceId(next);
    setGuidanceError(null);
    if (!result) {
      return;
    }
    const id = requestId.current;
    setGuidanceBusy(true);
    try {
      const updated = await fetchGuidance(next, result.label);
      if (id !== requestId.current) {
        return;
      }
      setDescribed(updated);
    } catch {
      if (id !== requestId.current) {
        return;
      }
      setProduceId(previous);
      setGuidanceError("Guidance for that produce type could not be loaded. The previous selection is kept.");
    } finally {
      if (id === requestId.current) {
        setGuidanceBusy(false);
      }
    }
  }

  const fresh = result?.label === "Fresh";
  const selectorLocked = busy || guidanceBusy;

  return (
    <div
      data-ui
      className="on-field w-full max-w-[560px] bg-[#e7e7e2] p-5 text-[var(--ink)] sm:p-6"
      style={{ borderRadius: 6 }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-[length:var(--step-lead)]">Analyze a photo</h2>
        <span className="text-[length:var(--step-small)] text-[var(--muted)]">JPEG or PNG · 8 MB max</span>
      </div>

      {!modelReady && !checking && (
        <p className="mt-3 bg-[#e6d3a8] px-3 py-2 text-[length:var(--step-small)]" role="status">
          {modelMessage || "The model is not available. Predictions stay disabled."}
        </p>
      )}

      <div
        className={`mt-4 transition-colors duration-200 ${dragging ? "bg-[#dfe8cc]" : "bg-[#dcdcd3]"}`}
        style={{ border: "1px dashed #7d7a6c", borderRadius: 4 }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
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
          <div className="p-3">
            <div className="relative mx-auto aspect-square w-full max-w-[340px] overflow-hidden bg-[#cfc7b3]">
              <img
                src={preview}
                alt="Your upload, resized to the square the model sees"
                className="absolute inset-0 h-full w-full"
                style={{ objectFit: "fill" }}
              />
              {PATCHES.map((patch, index) => (
                <span
                  key={index}
                  className={`patch absolute ${busy ? "is-busy" : ""}`}
                  style={{
                    left: `${patch.left}%`,
                    top: `${patch.top}%`,
                    width: `${PATCH_SIZE}%`,
                    height: `${PATCH_SIZE}%`,
                    border: `2px solid rgba(232,242,196,${busy || result ? 0.95 : 0.55})`,
                    ["--p" as string]: index,
                  }}
                />
              ))}
              {busy && (
                <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
                  <div className="scan h-full" style={{ borderBottom: "3px solid #e8f2c4" }} />
                </div>
              )}
            </div>
            <p className="mt-2 text-center text-[length:var(--step-small)] text-[var(--muted)]">
              Squared to 384 × 384. The boxes mark the five 160 × 160 patches the detail branch reads.
            </p>
          </div>
        ) : (
          <label
            htmlFor="food-image"
            className="flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-2 px-4 py-8 text-center"
          >
            <span className="display text-[length:var(--step-lead)]">Drop a photo here</span>
            <span className="text-[length:var(--step-small)] text-[var(--muted)]">
              or choose a fruit or vegetable image from your device
            </span>
          </label>
        )}
      </div>

      {preview && automatic && !result && (
        <div className="mt-4 text-[length:var(--step-small)] text-[var(--muted)]" data-testid="auto-note">
          <p>
            FreshNet will also try to recognise the produce type from the photo, using a separate model
            from the freshness one. Upload one clearly visible fruit or vegetable.
          </p>
          <details className="mt-1">
            <summary className="cursor-pointer font-medium text-[var(--ink)]">Supported produce types</summary>
            <p className="mt-1">
              {supportedLabels(options, supportedIds)}. The produce model has no &ldquo;not produce&rdquo;
              option, so something else in the photo can still get a confident wrong guess. You can
              correct the type after the result.
            </p>
          </details>
        </div>
      )}

      {preview && (!automatic || result) && (
        <div className="mt-4">
          <label htmlFor="produce-type" className="block text-[length:var(--step-small)] font-medium">
            {automatic ? "Correct produce type" : "Produce type"}
          </label>
          <select
            id="produce-type"
            aria-describedby="produce-type-hint"
            className="mt-1 w-full cursor-pointer bg-[#dcdcd3] px-3 py-2 disabled:cursor-not-allowed disabled:opacity-60"
            style={{ border: "1px solid #7d7a6c", borderRadius: 4 }}
            value={produceId}
            disabled={selectorLocked}
            onChange={(event) => void changeProduce(event.target.value)}
          >
            {options.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          <p id="produce-type-hint" className="mt-1 text-[length:var(--step-small)] text-[var(--muted)]">
            {automatic
              ? "Changing this only updates the label and the advice. The photo is not analysed again and the freshness result stays the same."
              : "Selected by you. Automatic produce recognition is not available right now, so this only sets the label and the advice."}
          </p>
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-4">
        {preview ? (
          <>
            <button type="button" className="button" onClick={analyze} disabled={!modelReady || busy || checking}>
              {busy ? "Looking…" : "Analyze"}
            </button>
            <button
              type="button"
              className="link cursor-pointer bg-transparent text-[length:var(--step-small)]"
              onClick={() => {
                choose(null);
                if (inputRef.current) {
                  inputRef.current.value = "";
                }
              }}
            >
              Remove
            </button>
            <span className="truncate text-[length:var(--step-small)] text-[var(--muted)]">{file?.name}</span>
          </>
        ) : (
          <label htmlFor="food-image" className="button">
            Choose a photo
          </label>
        )}
      </div>

      <p className="mt-3 text-[length:var(--step-small)] text-[var(--muted)]">
        Still images only. No transparency, and the pixels should already be upright: animated files
        and EXIF-only rotation are turned back.
      </p>

      {clientError && (
        <p className="mt-3 font-medium text-[#8a2d22]" role="alert">
          {clientError}
        </p>
      )}
      {serverError && (
        <p className="mt-3 font-medium text-[#8a2d22]" role="alert">
          {serverError}
        </p>
      )}
      {busy && (
        <p className="mt-3 text-[length:var(--step-small)]" role="status">
          Reading the whole picture, then the five close patches…
        </p>
      )}

      {guidanceError && (
        <p className="mt-3 font-medium text-[#8a2d22]" role="alert">
          {guidanceError}
        </p>
      )}

      {result && described && (
        <div aria-busy={guidanceBusy} className={guidanceBusy ? "opacity-60 transition-opacity" : "transition-opacity"}>
          <div
            className={`pop mt-5 p-5 text-white ${fresh ? "bg-[var(--field)]" : "bg-[var(--spoiled)]"}`}
            style={{ borderRadius: 4 }}
            role="region"
            aria-label="Result"
          >
            <h3 className="display text-4xl" data-testid="combined-label">
              Predicted: {described.combined_label}
            </h3>
            <IdentityNote
              described={described}
              result={result}
              options={options}
              disabled={selectorLocked}
              onUse={(id) => void changeProduce(id)}
            />

            <p className="mt-5 text-[length:var(--step-small)] opacity-80">
              Freshness from the model: <strong className="font-semibold">{result.label}</strong>
            </p>
            <div className="mt-2 space-y-3">
              <Meter label="Fresh" value={result.probabilities.Fresh} />
              <Meter label="Spoiled" value={result.probabilities.Spoiled} />
            </div>
            <p className="mt-3 text-[length:var(--step-small)] opacity-90">
              These probabilities are model outputs, not a freshness percentage or a shelf-life estimate.
            </p>
            <p className="mt-1 text-[length:var(--step-small)] opacity-70">
              Freshness model {result.model_version}. {result.elapsed_ms.toFixed(0)} ms for decode,
              validation, preprocessing, and the model forward.
            </p>

            <p className="mt-5 text-[length:var(--step-small)] opacity-80">
              Spoilage severity: <strong className="font-semibold">Not assessed</strong>
            </p>
            <p className="mt-1 text-[length:var(--step-small)] opacity-90" data-testid="severity-note">
              {described.severity.explanation}
            </p>
          </div>

          <GuidanceBlock guidance={described.guidance} unidentified={described.produce.id === "unknown"} />
        </div>
      )}
    </div>
  );
}

function labelFor(options: ProduceOption[], id: string): string {
  return options.find((option) => option.id === id)?.label ?? id;
}

function supportedLabels(options: ProduceOption[], ids: string[]): string {
  const labels = ids.map((id) => labelFor(options, id));
  return labels.length > 0 ? labels.join(", ") : "the trained produce types";
}

function topMatches(probabilities: Record<string, number> | null, options: ProduceOption[]): string {
  if (!probabilities) {
    return "";
  }
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([id, value]) => `${labelFor(options, id)} ${(value * 100).toFixed(1)}%`)
    .join(" · ");
}

/**
 * Says where the produce type came from. The produce model is separate from the
 * freshness model, so its numbers are labelled as its own and are never combined
 * with the freshness probabilities.
 */
function IdentityNote({
  described,
  result,
  options,
  disabled,
  onUse,
}: {
  described: Described;
  result: Prediction;
  options: ProduceOption[];
  disabled: boolean;
  onUse: (id: string) => void;
}) {
  const { produce } = described;
  const original = result.identification;
  const matches = topMatches(original.probabilities, options);
  const modelSaid =
    original.status === "model_predicted"
      ? result.produce.label
      : original.status === "needs_confirmation" && original.suggestion
        ? original.suggestion.label
        : null;

  let line: string;
  if (produce.source === "user_selected") {
    line =
      modelSaid && original.status !== "user_selected"
        ? produce.id === result.produce.id || produce.id === original.suggestion?.id
          ? `Produce type: ${produce.label}. Confirmed by you.`
          : `Produce type: ${produce.label}. Corrected by you; the produce model had said ${modelSaid}.`
        : `Produce type: ${produce.label}. Selected by you, not recognised from the image.`;
  } else if (produce.source === "model_predicted") {
    line = `Produce type: ${produce.label}. Predicted by the produce model, which is separate from the freshness model.`;
  } else if (original.status === "needs_confirmation") {
    line = "Please confirm produce type.";
  } else if (original.status === "failed") {
    line = "Produce type: not identified. Automatic identification failed for this photo; use the produce type selector. The freshness result is not affected.";
  } else if (original.status === "unavailable") {
    line = "Produce type: not identified. Automatic identification is not available; use the produce type selector for produce-specific advice.";
  } else {
    line = "Produce type: not identified. Use the produce type selector for produce-specific advice.";
  }

  const showSuggestion =
    produce.source !== "user_selected" && original.status === "needs_confirmation" && original.suggestion;
  const showMatches = matches !== "" && (original.status === "model_predicted" || original.status === "needs_confirmation");

  return (
    <div className="mt-2 text-[length:var(--step-small)]" data-testid="identity-note">
      <p className="opacity-90">{line}</p>
      {showSuggestion && original.suggestion && (
        <p className="mt-1 opacity-90">
          The produce model&rsquo;s best guess is {original.suggestion.label}
          {original.suggestion.confidence !== null ? ` (${(original.suggestion.confidence * 100).toFixed(0)}%)` : ""},
          but that is not confident enough to use without your check.{" "}
          <button
            type="button"
            className="link cursor-pointer bg-transparent font-semibold text-white disabled:opacity-60"
            style={{ textDecoration: "underline", textUnderlineOffset: 3 }}
            disabled={disabled}
            onClick={() => onUse(original.suggestion!.id)}
          >
            Use {original.suggestion.label}
          </button>
        </p>
      )}
      {showMatches && (
        <p className="mt-1 opacity-75" data-testid="produce-matches">
          Produce model top matches (its own scores, not the freshness result): {matches}
        </p>
      )}
    </div>
  );
}

function GuidanceBlock({ guidance, unidentified }: { guidance: Described["guidance"]; unidentified: boolean }) {
  const groups = [
    { title: "Suggested next steps", points: guidance.general_advice },
    { title: "Storage guidance", points: guidance.storage },
    { title: "What to check yourself", points: guidance.checks },
  ];
  return (
    <section className="mt-4" aria-label="Practical guidance" data-testid="guidance">
      {groups.map((group) => (
        <div key={group.title} className="mt-3 first:mt-0">
          <h4 className="text-[length:var(--step-small)] font-semibold">{group.title}</h4>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-[length:var(--step-small)]">
            {group.points.map((point) => (
              <li key={point.text}>{point.text}</li>
            ))}
          </ul>
        </div>
      ))}
      {guidance.review_status === "awaiting_review" && (
        <p className="mt-3 text-[length:var(--step-small)] text-[var(--muted)]">
          Storage advice specific to this produce type is awaiting review, so general advice is shown.
        </p>
      )}
      {unidentified && guidance.review_status === "generic" && (
        <p className="mt-3 text-[length:var(--step-small)] text-[var(--muted)]">
          This is general advice. Use the produce type selector for advice specific to it.
        </p>
      )}
      <p className="mt-3 text-[length:var(--step-small)] font-medium">{guidance.safety_note}</p>
      <p className="mt-1 text-[length:var(--step-small)] text-[var(--muted)]">{guidance.disclaimer}</p>
      <details className="mt-2 text-[length:var(--step-small)]">
        <summary className="cursor-pointer font-medium">Sources and review</summary>
        <ul className="mt-2 space-y-2">
          {guidance.sources.map((source) => (
            <li key={source.id}>
              <a className="link" href={source.url} target="_blank" rel="noreferrer">
                {source.title}
              </a>
              <span className="block text-[var(--muted)]">
                {source.publisher}. Covers: {source.scope} Reviewed {source.reviewed_on}.
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-[var(--muted)]">
          Guidance catalogue {guidance.catalogue_version}. Wording was checked against these pages by the
          FreshNet developers, not by a food-safety professional.
        </p>
      </details>
    </section>
  );
}

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-[length:var(--step-small)]">
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-white/20">
        <div
          className="h-full origin-left bg-[var(--field-soft)]"
          style={{
            width: `${Math.max(value * 100, 1)}%`,
            animation: "bar-in 1100ms var(--ease-out) both",
          }}
        />
      </div>
    </div>
  );
}
