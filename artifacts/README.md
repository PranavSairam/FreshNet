# Artifacts

Runtime and training outputs live here. **Checkpoints, logs, and metric files are gitignored.**

## Layout after a real training run

```text
artifacts/
  checkpoints/freshnet.pt    # written only by an actual training job
  metrics/history.json       # written from the trainer's real history
  logs/                      # optional training logs
```

This repository does not ship trained weights. Until a valid checkpoint exists at the configured path, the API and website must report **Model not trained** or **Model unavailable**.

Do not copy random `.pt` files into this folder and treat them as FreshNet results.
