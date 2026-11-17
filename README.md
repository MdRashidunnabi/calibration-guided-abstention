# Calibration-Guided Abstention for Aerial–Ground Person Re-Identification

Official implementation for the AG-ReID and AG-ReID.v2 experiments. The pipeline trains a ResNet-50 model, estimates confidence, calibrates an abstention threshold, and evaluates selective retrieval.

## Requirements

- Python 3.12
- CUDA-capable GPU
- AG-ReID and AG-ReID.v2 datasets

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Dataset setup

Set `AGREID_DATA` to a directory containing:

```text
AG-ReID/
AG-ReID.v2/
```

Linux or macOS:

```bash
export AGREID_DATA=/path/to/DATASET
```

Windows PowerShell:

```powershell
$env:AGREID_DATA = "C:\path\to\DATASET"
```

The datasets must follow their official directory and protocol-file structure. Dataset files are not included in this repository.

## Reproduce the results

Run the three reported experiments from the repository root:

```bash
python src/reproduce.py --dataset agreid
python src/reproduce.py --dataset agreid --all-ids
python src/reproduce.py --dataset agreid_v2
```

The default configuration uses seed 2026 and 80 epochs. Final metrics are written to `results/`. Generated training artifacts remain in the ignored `cache/` directory and are not part of the repository.

Expected Rank-1 results:

| Experiment | Protocol | Rank-1 |
|---|---|---:|
| AG-ReID | Aerial → ground | 0.681 |
| AG-ReID | Ground → aerial | 0.675 |
| AG-ReID, all training identities | Aerial → ground | 0.726 |
| AG-ReID, all training identities | Ground → aerial | 0.737 |
| AG-ReID.v2 | Aerial → CCTV | 0.676 |
| AG-ReID.v2 | Aerial → wearable | 0.675 |
| AG-ReID.v2 | CCTV → aerial | 0.695 |
| AG-ReID.v2 | Wearable → aerial | 0.614 |

Small numerical differences can occur across GPU and CUDA versions. The identity splits in `splits/` are fixed and should not be regenerated when reproducing the reported results.
