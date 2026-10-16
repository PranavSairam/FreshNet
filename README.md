# FreshNet

Local academic project: **visual food freshness classification** with a dual-path neural network.

The website asks for a fruit or vegetable image and classifies its visible condition as Fresh or Spoiled. That is a visual estimate. It does not certify food safety, and class probability is not a freshness percentage or a shelf-life estimate.

The API loads `artifacts/deployment/v3/fused_epoch29_b962c650ed80` when `FRESHNET_BUNDLE_DIR` points there. If that bundle is missing, the API still starts and predictions return unavailable. There is no mock model.

## Requirements

- Node.js 20.19+ (Vite 8). This repo was scaffolded with Node 25.
- Python **3.10+** (FastAPI 0.142.x and current PyTorch require it). macOS system Python 3.9 is not sufficient.
- Windows 10/11 with PowerShell, or macOS/Linux with bash.

## Repository

```text
frontend/    React + Vite + Tailwind UI
backend/     FastAPI + SQLite
ml/          DualPathFreshNet + train/eval
configs/     default.yaml
notebooks/   Colab/local training notebook
tests/       pytest
docs/        specification and status
```

## Start the API (Windows PowerShell)

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
cd ..
$env:PYTHONPATH = "$pwd\backend"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If `py` is missing, use `python` from a 3.10+ install.

Health check: http://127.0.0.1:8000/api/health

## Start the API (macOS / Linux)

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
cd ..
export PYTHONPATH="$PWD/backend:$PWD/ml"
export FRESHNET_BUNDLE_DIR="artifacts/deployment/v3/fused_epoch29_b962c650ed80"
export FRESHNET_DEVICE="cpu"
backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`FRESHNET_DEVICE=mps` is optional. CPU remains supported. A convenience script that starts the API and the website only when ports 8000 and 5173 are free:

```bash
bash scripts/start_local.sh
```

## Start the frontend (PowerShell or bash)

```powershell
cd frontend
npm install
Copy-Item .env.example .env   # PowerShell
npm run dev
```

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

UI: http://localhost:5173  
The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

`POST /api/predict` accepts one multipart field named `file`. Static JPEG and PNG only, at most 8 MB and 16 million pixels. Animated images, images with an alpha channel, and images whose EXIF orientation is not upright are rejected. EXIF is not applied, and transparency is not composited. Health can report the API as up while `model.available` is false. A loaded model reports `model.status` of `ready` and the bundle version. `elapsed_ms` covers decode, validation, preprocessing, and the model forward.

`POST /api/predict` also accepts an optional multipart field `produce` (a catalogue id such as `banana`, or `unknown`). FreshNet does not recognise produce from the photo: the identity is chosen by the user and shown as "Selected by you". The response keeps the fields below and adds `combined_label` (for example `Fresh banana`, or `Fresh produce` when unknown), `produce`, `freshness`, `severity` (always `not_assessed`) and `guidance` (reviewed, sourced, general advice). An unsupported `produce` value returns 400 `unsupported_produce`. `GET /api/produce` lists the supported types and `GET /api/guidance?produce=&freshness=` looks up guidance without running the model. See `docs/PRODUCE_IDENTIFICATION_AND_GUIDANCE.md` and `docs/SEVERITY_PROPOSAL.md`.

Example from a real validation JPEG/PNG request against the loaded bundle:

```json
{
  "label": "Fresh",
  "label_index": 0,
  "probabilities": {"Fresh": 0.9999998807907104, "Spoiled": 7.611431840359728e-08},
  "model_version": "freshnet-v3-fused-epoch29-b962c650ed80",
  "elapsed_ms": 104.229,
  "timing_scope": "decode_validation_preprocessing_and_forward"
}
```

The probabilities above are from `data/Dataset/Fresh/FreshBanana/b_f856.png` through the bundle loader. `elapsed_ms` varies by machine; the test checks that it is non-negative rather than a fixed duration.

## Tests

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
cd ..
$env:PYTHONPATH = "$pwd\backend;$pwd\ml"
python -m pytest
```

```bash
source backend/.venv/bin/activate
PYTHONPATH="$PWD/backend:$PWD/ml" python -m pytest
```

```powershell
cd frontend
npm run build
npm run lint
```

## Dataset preparation / training

See [docs/DATASET.md](docs/DATASET.md) and [docs/TRAINING.md](docs/TRAINING.md).

```powershell
$env:PYTHONPATH = "$pwd\ml"
python ml\audit_integrity.py --config configs\dataset.yaml --repair-splits
python ml\train_model.py --architecture fused --config configs\train.yaml --smoke
```

## Training (full runs)

Do not install dummy weights. When a labeled `ImageFolder` dataset exists:

```powershell
python ml/train.py --data-dir data/raw/your-dataset --num-classes 3
```

`--num-classes` must match the class folders. See `docs/PROJECT_SPEC.md`.

## License / academic use

This is a student project scaffold. Dataset licenses are your responsibility. Do not commit images or checkpoints.
