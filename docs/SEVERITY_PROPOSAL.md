# Proposal: genuine spoilage-severity prediction

Documentation only. Nothing here is implemented. No severity labels exist in the data, and no severity training has been run.

## Where things stand

The current model separates Fresh from Spoiled. It outputs two softmax scores and takes the larger. Those scores are not a measure of how spoiled something is. A confident "Spoiled" can be a lightly bruised item or a heavily rotten one, and a model can be unsure about an item that is clearly one or the other. For that reason the product shows "Not assessed" for severity and never turns the probability into a percent, a mild / moderate / severe label, a shelf life, a number of days or a safe-to-eat score. Cutting the probability into ranges would give those labels the look of measurement without any.

## What genuine severity prediction would need

1. **Explicit severity labels.** Written down by people looking at images, not inferred from folder names or from model output. A fixed scale such as 0 none, 1 slight, 2 clear, 3 extensive, with written criteria and example images for each step.
2. **Produce-specific definitions.** What counts as slight or extensive differs for a banana, a tomato and a potato. The guide must keep apart:
   - **ripeness** (for example spots on a ripe banana, which are not spoilage),
   - **bruising or surface damage** (cosmetic or early damage),
   - **spoilage** (mold, rot, collapse).
   A single scale that mixes these will teach the model the wrong thing.
3. **Consistent annotation.** At least two independent annotators per image, written guidelines, a pilot round, and agreement statistics (for example weighted kappa for ordinal labels). Items they disagree on are reviewed, and the disagreement rate is reported.
4. **Source and specimen grouping.** All images of one physical item, and near-duplicates and augmented copies of one source photo, must stay in the same split. The current dataset has no specimen ids, so new data should record them from the start. The existing source-group split can be reused for old images only after their labels are checked.
5. **Independent validation and test sets.** Model and threshold choices are made on validation only. The test set is scored once. It must be separate from the Fresh/Spoiled test set unless that one is annotated and its single-use rule is respected.
6. **Ordinal and multiclass metrics.** Per-class precision and recall, a confusion matrix, quadratic weighted kappa, mean absolute error on the ordinal scale, and the rate of errors two or more steps apart. Report per produce type, because pooled numbers hide weak types.
7. **Separate model and version.** Severity would be its own model with its own bundle, version, training data description and metrics. It would not be folded into the freshness model and would not reuse the 98.33 % accuracy figure.
8. **Calibration and abstention.** If severity is shown, check that stated confidence matches observed accuracy on held-out data, and allow "cannot assess".
9. **Interface.** `severity.status` would change from `not_assessed` to `assessed` only when a verified severity model is loaded. `severity.label` would then carry the label and the response would include that model's version and probabilities. Until then the contract stays as it is.

## Limits that will remain

A photo shows surface appearance. It cannot show smell, internal rot, storage history, toxins or the bacteria that may grow alongside mold. Even a good visual severity model would estimate how damaged the surface looks. It could not say whether the food is microbiologically safe, how long it will keep, or whether it can be salvaged. The interface would keep the line "Visual classification does not certify food safety".

## Suggested order of work

1. Write and test the severity guide for two or three produce types.
2. Annotate a pilot of a few hundred images, measure agreement, revise the guide.
3. Collect specimen ids and enough images per level and produce type.
4. Annotate in full and freeze the splits.
5. Train and select on validation only, then score the test set once.
6. Review the error analysis before any change to the interface.
