import { useEffect, useRef, useState } from "react";

const PATCHES = [
  [0, 0],
  [58.3333, 0],
  [0, 58.3333],
  [58.3333, 58.3333],
  [29.1667, 29.1667],
] as const;

const captions = [
  "The photo passes the checks: still JPEG or PNG, upright, no transparency, within the size limits.",
  "One resize to 384 × 384. The picture is squared, not cropped or rotated.",
  "The whole view goes to the context branch. The five patches go to the detail branch.",
  "Both summaries are joined into one score each for Fresh and Spoiled. The larger is the label.",
];

function Stage({ step }: { step: number }) {
  return (
    <div className="on-field bg-[var(--field)] p-5 text-white sm:p-7">
      <div className="relative mx-auto aspect-square w-full max-w-[460px] overflow-hidden bg-[#2d4220]">
        <img
          src="/pictures/roxbury-russet.jpg"
          alt="Roxbury Russet apples in a wooden crate, squared the way the model sees a photo"
          width={1400}
          height={1311}
          loading="lazy"
          className="absolute inset-0 h-full w-full"
          style={{ objectFit: "fill" }}
        />
        <span
          aria-hidden="true"
          className="absolute inset-0"
          style={{
            outline: `${step === 0 ? 5 : 0}px solid #e8f2c4`,
            outlineOffset: "-5px",
            background: step === 2 ? "rgba(232,242,196,0.16)" : "transparent",
            transition: "outline-width 400ms var(--ease-out), background-color 500ms ease",
          }}
        />
        {PATCHES.map(([left, top], index) => (
          <span
            key={index}
            aria-hidden="true"
            className={`patch absolute ${step === 2 ? "is-busy" : ""}`}
            style={{
              left: `${left}%`,
              top: `${top}%`,
              width: "41.6667%",
              height: "41.6667%",
              border: "2px solid #e8f2c4",
              opacity: step >= 1 ? 1 : 0,
              transition: "opacity 500ms var(--ease-out)",
              ["--p" as string]: index,
            }}
          />
        ))}
      </div>
      <p className="mt-4 min-h-[4.5em] text-[length:var(--step-small)] text-white/90" aria-live="off">
        {captions[step]}
      </p>
    </div>
  );
}

function Chips() {
  const items = [
    ["Context branch", "EfficientNet-B0", "224 × 224, whole picture"],
    ["Detail branch", "Small CNN, trained from scratch", "5 patches, averaged"],
    ["Fusion", "Joined, then two scores", "Fresh and Spoiled"],
  ];
  return (
    <div className="flex flex-wrap gap-3" data-ui>
      {items.map(([title, body, note]) => (
        <div key={title} className="min-w-[180px] flex-1 bg-[#e7e7e2] p-3" style={{ borderRadius: 6 }}>
          <p className="font-medium text-[var(--ink)]">{title}</p>
          <p className="text-[length:var(--step-small)] text-[var(--ink)]">{body}</p>
          <p className="text-[length:var(--step-small)] text-[var(--muted)]">{note}</p>
        </div>
      ))}
    </div>
  );
}

export function HowItWorks() {
  const [step, setStep] = useState(0);
  const items = useRef<(HTMLLIElement | null)[]>([]);

  useEffect(() => {
    if (!("IntersectionObserver" in window)) {
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const index = items.current.indexOf(entry.target as HTMLLIElement);
            if (index >= 0) {
              setStep(index);
            }
          }
        }
      },
      { rootMargin: "-40% 0px -50% 0px" },
    );
    items.current.forEach((element) => element && observer.observe(element));
    return () => observer.disconnect();
  }, []);

  const text = (lead: string, rest: string) => (
    <p className="leading-[1.6] text-[var(--ink)]">
      <strong className="font-medium text-[var(--ink)]">{lead}</strong> {rest}
    </p>
  );

  return (
    <section
      id="how-it-works"
      aria-labelledby="how-title"
      className="scroll-mt-16 py-[clamp(80px,11vw,180px)]"
    >
      <div className="mx-auto grid max-w-[1440px] gap-12 px-[var(--margin)] lg:grid-cols-[5fr_6fr] lg:gap-[8%]">
        <div className="lg:sticky lg:top-24 lg:self-start">
          <h2 id="how-title" className="mb-8 text-[length:var(--step-head)]" data-reveal>
            How it reads a photo
          </h2>
          <div data-reveal style={{ ["--i" as string]: 1 }}>
            <Stage step={step} />
          </div>
        </div>

        <ol className="m-0 flex list-none flex-col gap-[clamp(72px,12vw,200px)] p-0 pt-4 lg:pt-[18vh]">
          <li ref={(el) => { items.current[0] = el; }} data-reveal>
            {text(
              "First, the photo is checked.",
              "Static JPEG or PNG only, no transparency, upright pixels, up to 8 MB and 16 million pixels. Anything else is turned back with a reason instead of being quietly changed.",
            )}
            <div className="shutter drift relative mt-5 aspect-video w-full overflow-hidden">
              <img
                src="/pictures/vegetable-stall.jpg"
                alt="A market stall with crates of cabbage, aubergines, bananas, herbs and cucumbers"
                width={1500}
                height={1125}
                loading="lazy"
                className="h-full w-full object-cover"
              />
            </div>
          </li>
          <li ref={(el) => { items.current[1] = el; }} data-reveal>
            {text(
              "Then it is resized once.",
              "One 384 × 384 square, with no cropping or rotating. A 224 view carries the whole picture. Five 160 × 160 patches, the four corners and the centre, carry the close look.",
            )}
          </li>
          <li ref={(el) => { items.current[2] = el; }} data-reveal>
            {text(
              "Next, two networks read it.",
              "EfficientNet-B0 reads the whole view. A smaller network, trained from scratch, reads each patch and averages what it saw.",
            )}
            <div className="mt-5">
              <Chips />
            </div>
          </li>
          <li ref={(el) => { items.current[3] = el; }} data-reveal>
            {text(
              "At the end, the larger score wins.",
              "The two summaries are joined and turned into one score each for Fresh and Spoiled. The larger is the label. There is no Medium class.",
            )}
          </li>
        </ol>
      </div>
    </section>
  );
}
