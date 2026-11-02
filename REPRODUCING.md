# Reproduction instructions

## Environment and evidence

Use Python 3.12 in an isolated environment. The delivered dependency pins record the original run; this preparation does not claim that every pin is available for every operating system. Install the requirements from the relevant subdirectory. Vendored scikit-fem is retained with its original license.

For example, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r dynamic/requirements-reproduction.txt
```

On POSIX systems activate with `source .venv/bin/activate` instead.
The original Python build was 3.12.14. Exact static-run locks include OS, machine, Python and library provenance. On a different environment, strict verification may refuse to run. Do not edit the original locks. Fresh runs must use new output directories and be compared numerically within appropriate tolerances; a local successful test suite alone does not prove a full cross-platform reproduction.

## Dynamic results

First make a working copy of the `dynamic/` directory, because the supplied runners regenerate their default result directories. Commands below are run from that working copy.

PowerShell environment:

```powershell
$env:PYTHONPATH = "code;vendor;audit"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
```

POSIX environment:

```sh
export PYTHONPATH=code:vendor:audit
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Then follow `dynamic/README_REPRO.md`. Start with `python -m pytest -q tests`. The archived guide lists forward benchmarks, harmonic response, pulse histories, layer checks and plotting commands. No complete numerical rerun was performed during repository packaging.

Important data locations:

- `inputs/static_fields.json`: the 192 frozen static fields and their inherited residual status.
- `inputs/layer_calibration.json`: the two static intact-layer calibrations.
- `results/beam/beam_responses.npz`: 1,536 cases by 61 gauges, including complex predicted/reference strains and row keys.
- `results/beam/`: accompanying tables and pulse arrays.
- `results/published/`: published benchmark frequencies in rad/s, not Hz.
- `results/layer/` and `results/layer_refined/`: original and additionally refined soil-layer cases.

## Static results

Extract `static-results.zip` into the repository root before running data-backed checks or plotting. Work from `static/` and configure `PYTHONPATH` to include `code` and `vendor` (use `;` on Windows, `:` on POSIX).

Original-record verification commands:

```sh
python code/run_load_information.py verify --output results_load_information
python code/run_continuum_study.py verify --output results_continuum
python code/check_regularization_endpoint.py verify --source results_load_information --output results_regularization_endpoint
```

These commands retain the original environment checks. For new calculations, use the frozen designs and fresh outputs rather than overwriting archived data:

```sh
python code/run_load_information.py freeze --output rerun_load_information
python code/run_load_information.py run --output rerun_load_information
python code/run_load_information.py verify --output rerun_load_information
python code/run_continuum_study.py check-mesh --output rerun_continuum
python code/run_continuum_study.py calibrate --output rerun_continuum
python code/run_continuum_study.py run --output rerun_continuum
python code/run_continuum_study.py verify --output rerun_continuum
python code/check_regularization_endpoint.py freeze --source rerun_load_information --output rerun_endpoint
python code/check_regularization_endpoint.py run --source rerun_load_information --output rerun_endpoint
python code/check_regularization_endpoint.py verify --source rerun_load_information --output rerun_endpoint
```

Check each exit status before proceeding. Original serial budgets were two hours for the main inverse campaign, one hour for the continuum campaign and thirty minutes for the endpoint diagnostic. A timeout is an incomplete run, not a result.

Plot from the archived static results:

```sh
python code/make_load_information_figures.py --results results_load_information --out-dir regenerated_figures --summary-dir regenerated_load_summary
python code/make_continuum_figures.py --results results_continuum --out-dir regenerated_figures --summary-dir regenerated_continuum_summary
```

The static plotting driver generates some historical diagnostic figures beyond those in the current paper. The old static manuscript is intentionally not distributed here.

## Integrity

`docs/SOURCE_MANIFEST.csv` maps retained files to their original archive members and exact hashes. Provenance JSON files may retain historical machine/path identifiers; these are not installation paths. Old audit records report their original checks, not checks repeated during this packaging operation. See the preparation report outside the repository for the checks actually run now.
