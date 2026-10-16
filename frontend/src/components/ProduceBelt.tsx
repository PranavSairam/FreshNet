import { useState } from "react";
import { Produce, produceNames, type ProduceKind, type ProduceState } from "./Produce";

const lineup: { kind: ProduceKind; state: ProduceState }[] = [
  { kind: "apple", state: "fresh" },
  { kind: "tomato", state: "spoiled" },
  { kind: "lemon", state: "fresh" },
  { kind: "carrot", state: "fresh" },
  { kind: "banana", state: "spoiled" },
  { kind: "broccoli", state: "fresh" },
  { kind: "apple", state: "spoiled" },
  { kind: "carrot", state: "spoiled" },
  { kind: "lemon", state: "spoiled" },
  { kind: "tomato", state: "fresh" },
  { kind: "broccoli", state: "spoiled" },
  { kind: "banana", state: "fresh" },
];

export function ProduceBelt() {
  // Flips are local to this drawing. They are not model output.
  const [flipped, setFlipped] = useState<Record<string, boolean>>({});

  function toggle(key: string) {
    setFlipped((current) => ({ ...current, [key]: !current[key] }));
  }

  const row = (copy: number) =>
    lineup.map((item, index) => {
      const key = `${copy}-${index}`;
      const state: ProduceState = flipped[key] ? (item.state === "fresh" ? "spoiled" : "fresh") : item.state;
      const duplicate = copy === 1;
      return (
        <li key={key} className="shrink-0 px-3" aria-hidden={duplicate || undefined}>
          <button
            type="button"
            onClick={() => toggle(key)}
            tabIndex={duplicate ? -1 : 0}
            className="flex w-40 cursor-pointer flex-col items-center gap-3 bg-transparent p-3 text-center text-[var(--ink)] transition-transform duration-200 hover:-translate-y-2"
            aria-label={`${produceNames[item.kind]} drawing, currently ${state}. Press to flip it.`}
          >
            <span className="bob block" style={{ ["--i" as string]: index }}>
              <Produce kind={item.kind} state={state} size={110} />
            </span>
            <span className="text-[length:var(--step-small)] text-[var(--muted)]">
              {produceNames[item.kind]}, {state}
            </span>
          </button>
        </li>
      );
    });

  return (
    <section aria-labelledby="belt-title" className="belt overflow-hidden bg-[var(--surface)] py-[clamp(72px,10vw,140px)]">
      <div className="mx-auto max-w-[1280px] px-[var(--margin)]">
        <h2 id="belt-title" className="max-w-3xl text-[length:var(--step-head)]" data-reveal>
          Every piece starts fresh. Time has other plans.
        </h2>
        <p
          className="mt-5 max-w-xl text-[length:var(--step-lead)] leading-snug text-[var(--ink)]"
          data-reveal
          style={{ ["--i" as string]: 1 }}
        >
          Press any drawing to see it a week later. These are illustrations of the two classes, not
          model output.
        </p>
      </div>
      <ul className="belt-track mt-14 flex w-max" aria-label="Drawn fruits and vegetables, fresh and spoiled">
        {row(0)}
        {row(1)}
      </ul>
    </section>
  );
}
