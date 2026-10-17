# Produce classifier evaluation

Task: 12-class produce type. This is **not** the freshness model and its numbers must not be mixed with the freshness results.

- Checkpoint SHA-256: `e389c317c8e5df8f8434dd9025737afe4fb48b39e17d4ee63e87df78cfc759fa` (best epoch 10, chosen by validation macro F1)
- Produce manifest fingerprint: `9e3a27855e6e01d97a32fee61273dde59b1df7e3166ef25e764e03b97c872c66`
- Test subset: 8035 images. produce test subset of the existing v3 holdout; a separate-task evaluation, not a new independent external test.

| Split | Accuracy | Macro F1 | Balanced accuracy |
|---|---|---|---|
| Validation (used for selection) | 0.9929 | 0.9877 | 0.9898 |
| Test (evaluated once) | 0.9873 | 0.9781 | 0.9856 |

## Per-class test results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| apple | 0.9949 | 0.9790 | 0.9869 | 1000 |
| banana | 1.0000 | 0.9986 | 0.9993 | 710 |
| bitter_gourd | 1.0000 | 1.0000 | 1.0000 | 119 |
| capsicum | 0.9318 | 0.9927 | 0.9613 | 413 |
| carrot | 0.9964 | 0.9952 | 0.9958 | 1666 |
| cucumber | 0.9610 | 0.9471 | 0.9540 | 208 |
| mango | 0.9427 | 0.9737 | 0.9579 | 152 |
| okra | 0.9136 | 1.0000 | 0.9548 | 74 |
| orange | 0.9942 | 0.9840 | 0.9891 | 1565 |
| potato | 0.9307 | 0.9691 | 0.9495 | 291 |
| strawberry | 0.9919 | 1.0000 | 0.9959 | 122 |
| tomato | 0.9965 | 0.9883 | 0.9924 | 1715 |

## Confusion matrix (rows = true, columns = predicted)

| true \ pred | apple | banana | bitter_gourd | capsicum | carrot | cucumber | mango | okra | orange | potato | strawberry | tomato |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple | 979 | 0 | 0 | 6 | 0 | 2 | 3 | 0 | 1 | 8 | 0 | 1 |
| banana | 0 | 709 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bitter_gourd | 0 | 0 | 119 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| capsicum | 0 | 0 | 0 | 410 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| carrot | 0 | 0 | 0 | 2 | 1658 | 3 | 0 | 0 | 2 | 1 | 0 | 0 |
| cucumber | 1 | 0 | 0 | 0 | 2 | 197 | 0 | 7 | 0 | 1 | 0 | 0 |
| mango | 0 | 0 | 0 | 0 | 0 | 0 | 148 | 0 | 2 | 2 | 0 | 0 |
| okra | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 74 | 0 | 0 | 0 | 0 |
| orange | 3 | 0 | 0 | 4 | 2 | 0 | 5 | 0 | 1540 | 8 | 0 | 3 |
| potato | 0 | 0 | 0 | 4 | 1 | 2 | 0 | 0 | 0 | 282 | 1 | 1 |
| strawberry | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 122 | 0 |
| tomato | 1 | 0 | 0 | 14 | 0 | 1 | 0 | 0 | 4 | 0 | 0 | 1695 |

## Fresh / Spoiled subgroup accuracy (produce-type correctness)

- Fresh: n=5309, produce accuracy=0.993031
- Spoiled: n=2726, produce accuracy=0.976156

## Confirmation rule (chosen on validation only)

Threshold 0.5 (smallest grid value with selective accuracy >= 0.99).
- Validation: coverage 0.999491, selective accuracy 0.993248
- Test: coverage 0.999253, selective accuracy 0.987919, 6 images asked for confirmation

Limits: 12 supported classes only, no non-produce class, no out-of-distribution detection. Source images may be near-related across the original dataset in ways the group split cannot fully rule out.
