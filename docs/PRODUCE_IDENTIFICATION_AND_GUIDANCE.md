# Produce identity, combined labels and guidance

Status as of 2026-10-16. Produce type is now recognised by a separate, trained classifier (`PRODUCE_MODEL.md`); this page covers how identity, combined labels and guidance fit together.

## What works now

- **Freshness** is unchanged: the verified v3 fused model (`freshnet-v3-fused-epoch29-b962c650ed80`), same preprocessing, argmax rule and weights. Nothing was retrained or rescored.
- **Produce identity comes from a separate model, with the user in control.** A 12-class produce classifier (own bundle, preprocessing and version) runs next to the freshness model on every upload. The website shows who set the identity: the produce model, or "Selected by you" after a correction. A user's choice always wins. If the produce bundle is missing, fails, or is not confident, identity is Unknown and the website offers manual selection; freshness is unaffected. The filename is never used.
- **Combined label**: "Predicted: Fresh banana", "Predicted: Spoiled apple". Unknown identity gives "Predicted: Fresh produce" or "Predicted: Spoiled produce". Every word of the combined label after "Predicted:" is composed on the server from the freshness class and the catalogue noun.
- **Severity is always "Not assessed".** The response carries `severity: {status: "not_assessed", label: null, explanation}`. No probability is converted into a percent, a grade, a shelf life or a safety score.
- **Guidance** comes from a versioned catalogue (`backend/app/data/guidance_catalogue.json`), keyed by produce type and predicted freshness class. It is deterministic, reviewed text. No generative service is called, and text typed by a user is never put into advice.

## Not claimed

- The produce model recognises only its 12 trained types and has no "not produce" class. It is not an unknown / non-produce detector: the dataset has no out-of-scope images, so one could not be trained or evaluated, and unsupported or multi-item photos can still get confident guesses.
- Its accuracy (test 0.9873, macro F1 0.9781 on the existing holdout, a separate-task evaluation) is its own number, not the freshness model's 98.33 %.
- Spoilage severity is not estimated (see `SEVERITY_PROPOSAL.md`).

## API (additive)

`POST /api/predict` keeps every legacy field (`label`, `label_index`, `probabilities`, `model_version`, `elapsed_ms`, `timing_scope`). An optional multipart field `produce` takes a catalogue id. An id outside the catalogue returns 400 `unsupported_produce` (the submitted text is not echoed). New fields:

| Field | Meaning |
| --- | --- |
| `contract_version` | `freshnet.predict.v2` |
| `combined_label` | e.g. `Fresh banana`, `Spoiled produce` |
| `produce` | `{id, label, source: user_selected / model_predicted / unknown, model_version, probabilities}`. The last two are `null` unless the produce model produced the identity. Probabilities here are the produce model's own scores over the 12 types and are never combined with the freshness probabilities |
| `identification` | How the type was obtained: `status` (`user_selected`, `model_predicted`, `needs_confirmation`, `unavailable`, `failed`, `no_prediction`), `confirmation_required`, `suggestion` (`{id, label, confidence}` when confirmation is needed), `model_version`, `probabilities`, `confidence`, `confirmation_threshold`. When confirmation is needed, `produce` stays Unknown and the best guess is only a suggestion |
| `freshness` | `{label, class_index, probabilities, model_version}`, the same numbers as the legacy fields |
| `severity` | `{status: "not_assessed", label: null, explanation}` |
| `guidance` | `{catalogue_version, general_advice, storage, checks, review_status, safety_note, disclaimer, sources}` |

Other endpoints: `GET /api/produce` (selector options; `identification.mode` is `automatic_with_correction` when the produce bundle is loaded, otherwise `user_selected`; also `available`, `model_version`, `confirmation_threshold`, `supported_ids`), `GET /api/health` (adds a separate `produce_model` status) and `GET /api/guidance?produce=&freshness=` (catalogue lookup only, used when the user changes the produce type after a result; the image is not re-sent and the model is not run).

Order of checks on `/api/predict`: produce id, then image policy, then 503 if the freshness model is missing. The produce model runs after the timed freshness path, so `elapsed_ms` still measures only the freshness decode, validation, preprocessing and forward pass. If a user selected a type, the produce model is not run.

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

## How the automatic classifier was built (and what remains open)

Done, in `PRODUCE_MODEL.md` and `PRODUCE_DATASET_V1.md`:

1. **Label set decided and audited.** Bellpepper, Capciscum and Capsicum are merged as capsicum (bell pepper); Okara joins okra; Bittergroud is read as bitter gourd. Each merge was checked against sample images. 12 classes.
2. **Label problems fixed first.** The one cross-produce source group (RottenBellpepper / RottenCarrot, train) was excluded (2 images).
3. **v3 source-group split boundaries kept.** Trained on train, selected on validation. The test split was evaluated once after the checkpoint was frozen.
4. **Shortcut risk reported, not hidden.** Fresh and Spoiled subgroup results are reported separately (Fresh 0.9930, Spoiled 0.9762), alongside per-class metrics and macro averages.
5. **Evaluated on its own.** Per-class precision, recall and F1, a confusion matrix, macro F1, balanced accuracy, and a separate bundle and model version.
6. **Unknown handling.** No non-produce data exists, so the model ships with a confirmation rule (max-softmax threshold chosen on validation only), described as "please confirm", not as an unknown detector.
7. **Served as a separate component** through the `ProduceIdentifier` interface (`backend/app/services/produce_model.py`). A user's selection still wins, and failure or a missing bundle leaves identity unknown with freshness unaffected.

Still open:

- Guidance for bitter gourd still needs a source before it is marked reviewed.
- A genuine unknown / non-produce detector would need labelled out-of-scope images.
- An independent external test set (photos from outside this dataset) has not been collected; current numbers come from the existing holdout.
