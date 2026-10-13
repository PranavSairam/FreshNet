import { benchmarkScope, fusedTest, validationMacroF1 } from "../data/benchmark";

export function Performance() {
  return (
    <section id="performance" className="scroll-mt-24">
      <h2 className="font-serif text-3xl">Model performance</h2>
      <p className="mt-2 max-w-2xl text-stone-700">
        Validation and test are separate. These numbers describe the internal v3 benchmark of{" "}
        {benchmarkScope.included.toLocaleString()} images. {benchmarkScope.quarantined.toLocaleString()} of{" "}
        {benchmarkScope.original.toLocaleString()} original images were quarantined. Known confirmed
        same-capture links do not cross the splits, and the duplicate search was not exhaustive.
      </p>
      <h3 className="mt-6 text-lg font-medium">Validation macro F1</h3>
      <div className="mt-3 overflow-x-auto rounded-2xl border border-stone-200 bg-white">
        <table className="w-full text-left text-sm">
          <caption className="sr-only">Best validation macro F1 for each architecture</caption>
          <thead className="bg-[#f6f5f2] text-stone-600">
            <tr>
              <th className="px-4 py-3 font-medium">Model</th>
              <th className="px-4 py-3 font-medium">Epoch</th>
              <th className="px-4 py-3 font-medium">Macro F1</th>
            </tr>
          </thead>
          <tbody>
            {validationMacroF1.map((row) => (
              <tr key={row.model} className="border-t border-stone-200">
                <td className="px-4 py-3">
                  {row.model}
                  {"selected" in row ? " (selected)" : ""}
                </td>
                <td className="px-4 py-3">{row.epoch}</td>
                <td className="px-4 py-3">{row.macroF1.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3 className="mt-8 text-lg font-medium">Selected fused model on the test set</h3>
      <p className="mt-1 text-sm text-stone-600">{fusedTest.images.toLocaleString()} images. Rounded for display.</p>
      <dl className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Stat label="Accuracy" value={fusedTest.accuracy} />
        <Stat label="Balanced accuracy" value={fusedTest.balancedAccuracy} />
        <Stat label="Macro F1" value={fusedTest.macroF1} />
        <Stat label="Spoiled recall" value={fusedTest.spoiledRecall} />
        <div className="rounded-2xl border border-stone-200 bg-white p-4 sm:col-span-2">
          <dt className="text-sm text-stone-600">Spoiled classified as Fresh</dt>
          <dd className="mt-1 text-2xl text-stone-900">
            {fusedTest.spoiledAsFresh} of {fusedTest.spoiledImages.toLocaleString()}
          </dd>
        </div>
      </dl>
      <div className="mt-4 overflow-x-auto rounded-2xl border border-stone-200 bg-white p-4">
        <p className="text-sm text-stone-600">Confusion matrix. Rows are actual labels. Columns are predictions.</p>
        <table className="mt-3 text-sm">
          <thead>
            <tr>
              <th className="px-3 py-2" />
              <th className="px-3 py-2 font-medium">Pred Fresh</th>
              <th className="px-3 py-2 font-medium">Pred Spoiled</th>
            </tr>
          </thead>
          <tbody>
            {["Actual Fresh", "Actual Spoiled"].map((label, row) => (
              <tr key={label}>
                <th className="px-3 py-2 text-left font-medium">{label}</th>
                {fusedTest.confusion[row].map((count) => (
                  <td key={`${label}-${count}`} className="px-3 py-2">
                    {count.toLocaleString()}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-sm text-stone-600">
        A test accuracy of {(fusedTest.accuracy * 100).toFixed(2)}% is a result on that benchmark. It is not the
        chance that your next upload is classified correctly.
      </p>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-4">
      <dt className="text-sm text-stone-600">{label}</dt>
      <dd className="mt-1 text-2xl text-stone-900">{value.toFixed(3)}</dd>
    </div>
  );
}
