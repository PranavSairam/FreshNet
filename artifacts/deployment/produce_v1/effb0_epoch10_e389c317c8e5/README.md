# FreshNet produce bundle `freshnet-produce-v1-effb0-e389c317c8e5`

Separate 12-class produce-type classifier (EfficientNet-B0). It is independent of the freshness bundle.

- Validation accuracy / macro F1: 0.992869 / 0.987662
- Test accuracy / macro F1 (existing holdout, separate-task evaluation): 0.987306 / 0.978079
- Confirmation threshold (validation-chosen): 0.5

Serve with `FRESHNET_PRODUCE_BUNDLE_DIR=<this directory>`. If it cannot be loaded, the API falls back to manual produce selection.
