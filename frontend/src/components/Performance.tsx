import { benchmarkScope, fusedTest, validationMacroF1 } from "../data/benchmark";

const bars = [
  { label: "Context, validation macro F1", short: "Context", value: validationMacroF1[0].macroF1, tag: "Validation" },
  { label: "Detail, validation macro F1", short: "Detail", value: validationMacroF1[1].macroF1, tag: "Validation" },
  { label: "Fused, validation macro F1", short: "Fused", value: validationMacroF1[2].macroF1, tag: "Validation", accent: true },
  { label: "Fused, test macro F1", short: "Macro F1", value: fusedTest.macroF1, tag: "Test" },
  { label: "Fused, test balanced accuracy", short: "Balanced", value: fusedTest.balancedAccuracy, tag: "Test" },
  { label: "Fused, test Fresh recall", short: "Fresh recall", value: fusedTest.freshRecall, tag: "Test" },
  { label: "Fused, test Spoiled recall", short: "Spoiled recall", value: fusedTest.spoiledRecall, tag: "Test", accent: true },
] as const;

export function Performance() {
  return (
    <section
      id="performance"
      aria-labelledby="perf-title"
      className="reveal-bars on-field scroll-mt-0 bg-[var(--field)] py-[clamp(72px,9vw,140px)] text-[var(--field-ink)]"
    >
      <div className="mx-auto max-w-[1440px] px-[var(--margin)]">
        <h2 id="perf-title" className="sr-only">
          Model Performance
        </h2>
        <p
          className="display text-[clamp(5.5rem,10vw,10.5rem)] leading-none tabular-nums"
          data-reveal
          aria-label={`${(fusedTest.accuracy * 100).toFixed(1)} percent test accuracy`}
        >
          {(fusedTest.accuracy * 100).toFixed(1)}%
        </p>
        <p className="mt-2 text-[length:var(--step-lead)] leading-tight" data-reveal style={{ ["--i" as string]: 1 }}>
          Accuracy on {fusedTest.images.toLocaleString()} test images
        </p>
        <p className="mt-6 max-w-2xl text-[length:var(--step-body)] text-white/85" data-reveal style={{ ["--i" as string]: 2 }}>
          Validation chose the model. The test set was scored once afterward. These figures belong to
          the internal v3 benchmark of {benchmarkScope.included.toLocaleString()} images.{" "}
          {benchmarkScope.quarantined.toLocaleString()} of {benchmarkScope.original.toLocaleString()} original
          images were quarantined. Confirmed same-capture links that stayed in do not cross splits, and
          the duplicate search was not exhaustive.
        </p>
      </div>

      <div className="bars-wrap mx-auto mt-14 max-w-[1440px] px-[var(--margin)]" data-reveal data-bars>
        <ul className="bars" aria-label="Recorded scores, bars drawn from zero to one">
          {bars.map((bar, index) => (
            <li key={bar.label} className="bar-col flex flex-col justify-end gap-2" style={{ ["--i" as string]: index }}>
              <span className="tabular-nums text-[length:var(--step-lead)] leading-none">
                {bar.value.toFixed(3)}
              </span>
              <span className="bar accent-bar" style={{ ["--v" as string]: bar.value, ["--i" as string]: index }} data-accent={"accent" in bar || undefined} aria-hidden="true" />
              <span className="text-[length:var(--step-small)] text-white/80">
                {bar.short}
                <span className="block text-white/60">{bar.tag}</span>
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 max-w-2xl text-[length:var(--step-small)] text-white/70">
          Bars start at zero and end at 1.0, so they look alike: the printed numbers carry the
          difference. Context and Detail were scored on validation only. The test set was scored for
          the selected fused model alone.
        </p>
      </div>

      <div className="mx-auto mt-16 grid max-w-[1440px] gap-10 px-[var(--margin)] lg:grid-cols-[1fr_1fr]">
        <div data-reveal>
          <h3 className="text-[length:var(--step-lead)]">What it got wrong</h3>
          <p className="mt-3 max-w-md text-white/85">
            Spoiled classified as Fresh: {fusedTest.spoiledAsFresh} of {fusedTest.spoiledImages.toLocaleString()}{" "}
            Spoiled test images. {(fusedTest.accuracy * 100).toFixed(2)}% accuracy is a result on that
            held-out benchmark. It is not the chance that the next photo you upload will be right.
          </p>
        </div>
        <div data-reveal style={{ ["--i" as string]: 1 }} className="overflow-x-auto">
          <table className="w-full min-w-[320px] text-left tabular-nums" data-ui>
            <caption className="pb-2 text-left text-[length:var(--step-small)] text-white/70">
              Confusion matrix on the test set. Rows are actual, columns are predicted.
            </caption>
            <thead>
              <tr>
                <th className="py-2 pr-4 font-normal" />
                <th className="py-2 pr-4 font-medium">Predicted Fresh</th>
                <th className="py-2 font-medium">Predicted Spoiled</th>
              </tr>
            </thead>
            <tbody>
              {["Actual Fresh", "Actual Spoiled"].map((label, row) => (
                <tr key={label} style={{ borderTop: "1px solid rgba(255,255,255,0.25)" }}>
                  <th className="py-3 pr-4 font-medium">{label}</th>
                  {fusedTest.confusion[row].map((count, column) => (
                    <td
                      key={`${label}-${column}`}
                      className={`py-3 pr-4 text-[length:var(--step-lead)] ${row === column ? "" : "text-[var(--field-soft)]"}`}
                    >
                      {count.toLocaleString()}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
