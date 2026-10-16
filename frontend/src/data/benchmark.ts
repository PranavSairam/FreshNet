/**
 * Recorded v3 metrics. Validation figures are the selected checkpoints' best
 * validation macro F1 values. Test figures are copied from
 * artifacts/metrics/v3/eval/fused/test/metrics.json.
 * They are not recomputed in the browser.
 */
export const validationMacroF1 = [
  { model: "Context", epoch: 19, macroF1: 0.984695 },
  { model: "Detail", epoch: 26, macroF1: 0.932566 },
  { model: "Fused", epoch: 29, macroF1: 0.985118, selected: true },
] as const;

export const fusedTest = {
  images: 8035,
  accuracy: 0.983323,
  balancedAccuracy: 0.979706,
  macroF1: 0.981337,
  spoiledRecall: 0.968452,
  freshRecall: 0.990959,
  spoiledPrecision: 0.982143,
  spoiledAsFresh: 86,
  spoiledImages: 2726,
  confusion: [
    [5261, 48],
    [86, 2640],
  ],
} as const;

export const benchmarkScope = {
  included: 53052,
  quarantined: 18266,
  original: 71318,
} as const;
