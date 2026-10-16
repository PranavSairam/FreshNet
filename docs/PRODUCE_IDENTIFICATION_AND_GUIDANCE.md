# Produce identity, combined labels and guidance

Status as of 2026-10-16. This describes what exists today and what would be needed to go further.

## What works now

- **Freshness** is unchanged: the verified v3 fused model (`freshnet-v3-fused-epoch29-b962c650ed80`), same preprocessing, argmax rule and weights. Nothing was retrained or rescored.
- **Produce identity is chosen by the user.** No produce classifier exists in this repository, so the website offers a "Produce type" selector (12 types plus "Unknown / other"). The identity is labelled "Selected by you" and is never presented as recognition. The filename is never used.
- **Combined label**: "Predicted: Fresh banana", "Predicted: Spoiled apple". Unknown identity gives "Predicted: Fresh produce" or "Predicted: Spoiled produce". Every word of the combined label after "Predicted:" is composed on the server from the freshness class and the catalogue noun.
- **Severity is always "Not assessed".** The response carries `severity: {status: "not_assessed", label: null, explanation}`. No probability is converted into a percent, a grade, a shelf life or a safety score.
- **Guidance** comes from a versioned catalogue (`backend/app/data/guidance_catalogue.json`), keyed by produce type and predicted freshness class. It is deterministic, reviewed text. No generative service is called, and text typed by a user is never put into advice.

## Not claimed

- Automatic produce recognition does not exist.
- There is no unknown / non-produce detector. The dataset has no out-of-scope images, so one could not be trained or evaluated.
- Spoilage severity is not estimated (see `SEVERITY_PROPOSAL.md`).

## API (additive)

`POST /api/predict` keeps every legacy field (`label`, `label_index`, `probabilities`, `model_version`, `elapsed_ms`, `timing_scope`). An optional multipart field `produce` takes a catalogue id. An id outside the catalogue returns 400 `unsupported_produce` (the submitted text is not echoed). New fields:

| Field | Meaning |
| --- | --- |
| `contract_version` | `freshnet.predict.v2` |
| `combined_label` | e.g. `Fresh banana`, `Spoiled produce` |
| `produce` | `{id, label, source: user_selected / model_predicted / unknown, model_version, probabilities}`. The last two are `null` unless a separate model produced the identity |
| `freshness` | `{label, class_index, probabilities, model_version}`, the same numbers as the legacy fields |
| `severity` | `{status: "not_assessed", label: null, explanation}` |
| `guidance` | `{catalogue_version, general_advice, storage, checks, review_status, safety_note, disclaimer, sources}` |

Other endpoints: `GET /api/produce` (selector options, `identification.mode = user_selected`, `available = false`) and `GET /api/guidance?produce=&freshness=` (catalogue lookup only, used when the user changes the produce type after a result; the image is not re-sent and the model is not run).

Order of checks on `/api/predict`: produce id, then image policy, then 503 if the model is missing. Identity and guidance are added after the timed freshness path, so `elapsed_ms` still measures decode, validation, preprocessing and the forward pass.

## Guidance catalogue

Version `2026-10-16.1`, reviewed 2026-10-16. Each point lists its source ids. Sources were fetched and read on the review date:

| Id | Source | Used for |
| --- | --- | --- |
| `fda_selecting_serving` | FDA, Selecting and Serving Produce Safely | washing under running water, refrigerating perishable produce, cutting away damage, throwing away rotten produce |
| `usda_fsis_molds` | USDA FSIS, Molds on Food: Are They Dangerous? | do not sniff mold, discard if covered, soft produce, clean the storage spot, check nearby items |
| `ucd_postharvest_storing` | UC Davis Postharvest / UC ANR Publication 21590 | room temperature or refrigerator storage for apple, banana, carrot, cucumber, mango, orange, pepper, potato, strawberry, tomato |
| `uc_master_gardener_storing` | UC Master Gardener, Harvesting and Storing | okra storage, ripe fruit near carrots, tomato storage |

The review checked wording against the pages. It was done by the developers, not by a food-safety professional, and the interface says so.

Review state:

- **Reviewed storage guidance:** apple, banana, bell pepper / capsicum, carrot, cucumber, mango, okra, orange, potato, strawberry, tomato.
- **Awaiting review:** bitter gourd. None of the fetched sources covers it. The result shows the general refrigerator note and says produce-specific advice is awaiting review.
- **Unknown / other:** general advice only.
- Spoiled results use only shared, reviewed text (throw away what looks rotten, discard if covered in mold, do not sniff it, clean the storage spot). Trimming advice that USDA gives for firm produce is deliberately left out of Spoiled results, because this app cannot judge how much of an item is affected.

Rules the tests enforce: every point has a source in the catalogue; every source has an https URL, scope and review date; a catalogue marked reviewed has storage points and one awaiting review has none; the words "shelf life", "expire", "safe to eat", "mild/moderate/severe", "mold detected", "cooking" and "peel" never appear in guidance; Spoiled guidance never suggests cutting, trimming or washing.

To change guidance: edit the JSON, bump `catalogue_version`, re-read the cited page, update `reviewed_on`, and run `pytest tests/backend/test_produce_guidance.py`.

## Produce labels in the data

See `PRODUCE_LABEL_INVENTORY.md` (and the JSON beside it). In short: produce type exists only as the image folder name (`FreshBanana`, `RottenBanana`, ...). There are no specimen ids. Counts range from 529 (okra) to 11,431 (tomato). One source group mixes a bell pepper and a carrot image. Carrot, orange and tomato are mostly Fresh, while apple, potato and strawberry lean Spoiled.

## What an automatic produce classifier would need

1. **Decide the label set.** The audit finds 13 produce ids, because `Bellpepper` and `Capsicum` are separate folders. The selector offers 12 types because it joins them as "Bell pepper / capsicum" (a bell pepper is a capsicum). A classifier would need the same rule, and the merge should be checked against the images first. `Bittergroud` is also assumed to mean bitter gourd.
2. **Fix label problems first.** Resolve the cross-produce group (RottenBellpepper / RottenCarrot, train) and recheck the spelling variants.
3. **Keep the existing source-group split boundaries** from `manifests_v3_source`. Train on train, select on validation. The test split is for one final evaluation and must not be used for development, thresholds or model choice.
4. **Guard against shortcuts.** Produce folders differ in background, camera and file style, and some are mostly one freshness class, so the classifier can learn those cues. Report per-class metrics for Fresh and Spoiled separately, and prefer macro averages because classes are very uneven.
5. **Evaluate on its own.** Per-class precision and recall, a confusion matrix, macro F1, and a separate model version and bundle. Its accuracy is its own number and must not borrow the 98.33 % freshness figure.
6. **Unknown handling.** The dataset has no out-of-scope images. Either collect and label some, or ship the classifier only with an abstain rule tuned on validation and described as "low confidence", not as an unknown detector.
7. **Serve it as a separate component.** Implement the `ProduceIdentifier` interface in `backend/app/services/produce.py` and register it with `set_identifier`. The result then reports `source: model_predicted`, its own `model_version` and `probabilities`. A user's selection still wins. If the identifier fails, returns nothing or returns an id outside the catalogue, identity becomes unknown and freshness is unaffected. This path is covered by tests with stand-in identifiers only.
8. **Guidance for bitter gourd** still needs a source before it is marked reviewed.
