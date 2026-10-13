const steps = [
  ["Upload", "Still JPEG or PNG"],
  ["Preprocess", "384 resize, then 224 view and five patches"],
  ["Context", "EfficientNet-B0 on the whole view"],
  ["Detail", "Shared CNN on five patches"],
  ["Fusion", "Join both features, then Fresh or Spoiled"],
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="scroll-mt-24">
      <h2 className="font-serif text-3xl">How it works</h2>
      <p className="mt-2 max-w-2xl text-stone-700">
        The uploaded image is validated, then prepared once. Two branches read that preparation in
        parallel. Fusion combines their representations and the higher logit is the label.
      </p>
      <div className="mt-6 flex flex-col gap-3 md:flex-row md:items-stretch" aria-label="Processing diagram">
        {steps.map(([title, copy], index) => (
          <div key={title} className="flex flex-1 items-stretch gap-3">
            <article className="flex-1 rounded-2xl border border-[#1f4d3a]/30 bg-white p-4">
              <p className="text-sm font-medium text-[#1f4d3a]">{title}</p>
              <p className="mt-2 text-sm text-stone-700">{copy}</p>
            </article>
            {index < steps.length - 1 && (
              <span className="hidden items-center text-[#1f4d3a] md:flex" aria-hidden="true">
                →
              </span>
            )}
          </div>
        ))}
      </div>
      <p className="mt-4 text-sm text-stone-600">
        Context is the whole-image branch. Detail averages five patch embeddings from a shared CNN.
        Fusion concatenates those two vectors and classifies Fresh versus Spoiled. There is no Medium class.
      </p>
    </section>
  );
}
