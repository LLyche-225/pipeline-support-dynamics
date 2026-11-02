# Reproducing the dynamic pipe-strain study

The data are numerical. Unpack the dynamic research archive into a **new directory** and run commands from its root. The 192 static support fields and two intact layer calibrations are included under inputs/. No dynamic observation was used to fit them. The static research archive is separate and contains the full earlier inverse campaigns, inputs, raw outputs and static reproduction instructions. Its historical manuscript is not the current submission.

Use Python 3.12, install the pinned numerical/plotting packages from requirements-reproduction.txt into an isolated environment, and use the included scikit-fem 12.0.2 with its third-party license. On the original machine the workspace-managed scientific Python was used. The following examples assume the environment's interpreter is named python. Rerunning scripts regenerates results; work in a copy to retain the distributed evidence.

```sh
export PYTHONPATH=code:vendor:audit
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m pytest -q tests
python code/published_modes.py
python code/dynamic_beam_verify.py
python code/dynamic_beam_run.py
python code/dynamic_layer_run.py --budget-s 2400
python code/verify_layer_refinement.py
python code/plot_vibration_transfer.py
python audit/replay_intact_exact.py
python audit/replay_layer_mass.py
python audit/replay_layer_comparison.py
python code/plot_layer_transfer.py
```

The primary frequency data are in results/beam/beam_responses.npz: 1,536 rows by 61 gauge locations, with complex predicted and reference strains and a row key. The paired CSV tables and the pulse arrays are stored beside it. Join each key to inputs/static_fields.json to recover the original static residual status (inherited_compatible). The label selected means the optimizer returned a field, not that the field passed the discrepancy test. The zero endpoint means zero loss, not zero stiffness. results/published/ contains angular frequencies in rad/s, while the harmonic campaigns use frequency in Hz and angular frequency internally.

The production beam and scikit-fem reference have distinct stiffness, load and gauge assembly. The matrix-exponential check uses neither finite-element assembly. The soil-layer mass replay uses exact tensor-product element mass matrices with the verified static stiffness. Mesh refinement and changes to the remote lateral boundaries answer different questions: the latter is a change to the physical fixture. See DYNAMIC_PROTOCOL.md for excitations, units and reference conditions. Results do not constitute a measured earthquake or field validation.

The archive retains the original grid study and the six additional refinements. Where a 0.25 m production case is flagged unconverged, use the corresponding 1/12 m result in results/layer_refined/cases.csv for manuscript panel (a). For the boundary check the manuscript and audit/layer_replay_metrics.csv normalize the 50-to-60 m difference by the norm for the longer domain; results/layer/boundary_sensitivity.csv normalizes the same difference by the shorter-domain norm. These are different ratios, not inconsistent simulations. Both end clamps and soil side boundaries move in this check.
