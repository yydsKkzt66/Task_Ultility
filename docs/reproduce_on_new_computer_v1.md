# Reproduce the final V25 experiments on a new computer

## What is now stored in Git

The repository contains the final source, configurations, tests, frozen
ordinary-V25 scenario bundles, calibrated two-node thermal parameters, public
accounting parameters, final plotting scripts, and the formerly external
frozen runtime snapshots under `vendor/frozen_runtime_v1`.

Raw SARFish imagery is not required by the final synthetic V25 main,
ablation, or sensitivity experiments and is not stored in Git.

## Reference environment

- Windows 10/11 x64;
- Python 3.9.13;
- packages pinned in `requirements-reproduce.txt`;
- at least 8 GB RAM; 16 GB is recommended for parallel runs;
- sufficient free disk space for newly generated run traces.  Keep at least
  20 GB free if running every experiment.

The plots prefer Times New Roman when installed and fall back to DejaVu Serif.
The data and geometry are reproducible with the fallback, but exact text
metrics can differ slightly between fonts and operating systems.

The final workflow does not import pandas.  It is intentionally omitted from
the pinned reproduction environment; some older exploratory scripts may still
use the broader, non-frozen `requirements.txt`.

## First-time setup

```powershell
git clone https://github.com/yydsKkzt66/Task_Ultility.git
cd Task_Ultility
git checkout release/final-v25-dual-thermal-v1
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduce.txt
python scripts/verify_portable_reproduction_v1.py --strict-versions
```

The strict preflight verifies every vendored file against the committed SHA256
manifest.  It does not run an experiment.

## Recreate the final figures quickly

Run without arguments, including from an IDE Run button:

```powershell
python scripts/reproduce_on_new_computer_v1.py
```

Outputs:

- `scripts/combined_primary_auxiliary_metrics_boxed_titles/`;
- `scripts/sensitivity_ablation_2x2_zoom_legends_v1/`.

The two plotting scripts contain the reviewed frozen numeric arrays used for
the submitted figures.  Therefore figure regeneration does not require the
multi-hour simulation traces.

## Run simulations

```powershell
# Fast mechanism check: five methods, seed 2026
python scripts/reproduce_on_new_computer_v1.py --scope single

# Main experiment: 5 methods x 5 seeds, all 25 runs fresh
python scripts/reproduce_on_new_computer_v1.py --scope main

# Five ablations x five seeds
python scripts/reproduce_on_new_computer_v1.py --scope ablation

# V10/V25/V50 and capacity sensitivity
python scripts/reproduce_on_new_computer_v1.py --scope sensitivity

# Everything followed by both figure sets
python scripts/reproduce_on_new_computer_v1.py --scope all
```

Every run creates a new numbered `attempt_XXX` directory and does not overwrite
the committed compact results.  The main portable mode deliberately reruns
seed 2026 instead of depending on the old workstation's complete run folder.

## Storage policy

Git contains compact tables, audits, figures, and all inputs needed for the
final V25 workflow.  Newly generated full traces remain ignored by `.gitignore`.
Archive those separately if raw trace preservation is required; they are not
needed to clone, rerun, or redraw the final figures.
