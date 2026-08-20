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
- at least 8 GB RAM; 16 GB is recommended for parallel runs. Canonical batch
  entry points cap concurrency at two processes to avoid memory-pressure
  failures without changing any scenario, method, or metric;
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
git -c core.longpaths=true clone --branch comsoc-apb-2026-v1 --single-branch https://github.com/yydsKkzt66/Task_Ultility.git
cd Task_Ultility
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduce.txt
python scripts/verify_portable_reproduction_v1.py --strict-versions
```

The strict preflight verifies every vendored file against the committed SHA256
manifest.  It does not run an experiment.

On Linux/macOS, replace the virtual-environment commands with:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduce.txt
python scripts/verify_portable_reproduction_v1.py --strict-versions
```

## Recreate the final figures quickly

Run without arguments, including from an IDE Run button:

```powershell
python scripts/reproduce_on_new_computer_v1.py
```

Outputs:

- `artifacts/competition_release_v1/figures/main_four_panel.pdf` and `.png`
  (submitted four-panel main figure: overall utility, high-value utility,
  high-value task closures, and utility–resource tradeoff);
- `artifacts/competition_release_v1/figures/sensitivity_ablation_2x2.pdf`
  and `.png`;
- `artifacts/competition_release_v1/figures/sensitivity_ablation_normalized_values.csv`.

Older figure directories under `scripts/` are retained as research history;
the default competition entry neither reads nor overwrites them.

The main plotting script reads the Git-tracked five-seed simulation summary;
the sensitivity/ablation script reads the reviewed frozen plotting values.
Figure regeneration therefore does not require multi-hour full traces. The
old high-value-layer-delivery and overall-closure-latency panels are not part
of the default or submitted main figure.

The default entry also runs:

```bash
python scripts/verify_competition_results_v1.py
```

This independently reads the compact simulation table, verifies the five
headline PROP values, and recomputes all four PROP-vs-TUT percentages.

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

## Optional SARFishSample adapter

The competition V25 experiments are synthetic and do not read SAR imagery.
Only adapter development or validation needs the external dataset. The public
configuration contains no machine-specific drive path and reads
`SARFISH_DATASET_ROOT` when supplied.

Windows PowerShell (the migrated local dataset is now at `E:\6.22Dataset`):

```powershell
$env:SARFISH_DATASET_ROOT = 'E:\6.22Dataset'
python -m pytest -q tests/test_sarfishsample_dataset.py
```

Linux/macOS:

```bash
export SARFISH_DATASET_ROOT=/data/6.22Dataset
python -m pytest -q tests/test_sarfishsample_dataset.py
```

Do not commit raw SAR imagery or a generated local scene index. If the
environment variable is absent, the optional SAR adapter requires an explicit
dataset path; the default competition workflow remains unaffected.

## Runtime path policy

The final entry points prefer `vendor/frozen_runtime_v1` inside the clone.
Environment-variable overrides and historical `D:/wt/...` paths in the
resolver are retained only for audited legacy workstations; they are never
needed by the public default workflow. Older exploratory scripts are research
history and are not canonical reproduction entry points.

## Storage policy

Git contains compact tables, audits, figures, and all inputs needed for the
final V25 workflow.  Newly generated full traces remain ignored by `.gitignore`.
Archive those separately if raw trace preservation is required; they are not
needed to clone, rerun, or redraw the final figures.
