"""Driver for the dynamic elastic-layer vs local-spring benchmark.

Runs protocol step 4 end to end: verification, bounded mesh study,
production comparison, finite-domain boundary sensitivity, figure, and the
compact audit report.  See audit/task_layer.md for the bounded scope.

Exact run command (from the repository root):

    PYTHONPATH=code:vendor OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-layer \
    python code/dynamic_layer_run.py

Test command:

    PYTHONPATH=code:vendor OMP_NUM_THREADS=1 \
    python -m pytest tests/test_dynamic_layer.py -q
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "vendor"))

import continuum_reference as cr  # noqa: E402
import dynamic_layer as dl  # noqa: E402
from shape_factory import asymmetric_shape, intact_shape  # noqa: E402
from skfem_reference import mesh_with_load_edges  # noqa: E402

V3_ROOT = ROOT.parent / "revision_v3_load_information_20261102"
V3_CALIBRATION = V3_ROOT / "results_continuum" / "calibration"
RESULTS = ROOT / "results" / "layer"
AUDIT = ROOT / "audit"
FIGURES = ROOT / "manuscript" / "figures_v5"

CONVERGENCE_TOL = 0.01  # v3 last_strain_relative_L2_change_max
SHAPES = ("intact", "asymmetric")
CENTERS = np.array([dl.LOAD_CENTER_M])

BC_CAVEAT = (
    "Beam clamped at both domain ends; soil bottom fixed in both components; "
    "soil sides are vertical rollers (horizontal displacement fixed, vertical "
    "free). Finite fixture, not a half-space impedance model. The +/-60 m "
    "extent check also lengthens the clamped beam from 100 m to 120 m."
)
TRANSFER_ASSUMPTION = (
    "Comparator spring k(x) = k0_calib(E) * (1 - d(x)): the same dimensionless "
    "reduction as the layer interface loss applied to the intact-calibrated "
    "uniform equivalent stiffness. Equality of physical interface fields is "
    "not claimed or tested."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_calibration() -> dict:
    """v3 intact-state equivalent stiffness per modulus (immutable inputs).

    Primary: inputs/layer_calibration.json (schema dynamic-layer-calibration-input-v1,
    copied by the input-export worker).  Each entry's source_sha256 is checked
    against the sibling v3 calibration file.  Fallback: read the sibling v3
    files directly.
    """
    copied = ROOT / "inputs" / "layer_calibration.json"
    if copied.exists():
        data = json.loads(copied.read_text())
        records = {}
        for entry in data["calibrations"]:
            modulus = float(entry["soil_E_Pa"])
            sibling = V3_CALIBRATION / Path(entry["source_file"]).name
            verified = bool(sibling.exists()) and _sha256(sibling) == entry["source_sha256"]
            records[str(int(modulus))] = {
                "k0_N_m2": float(entry["k0_N_m2"]),
                "source": entry["source_file"],
                "source_sha256": entry["source_sha256"],
                "sibling_hash_verified": verified,
            }
        return {"source": copied.relative_to(ROOT).as_posix(), "sha256": _sha256(copied), "records": records}
    records = {}
    for modulus in dl.SOIL_MODULI_PA:
        path = V3_CALIBRATION / f"E{int(modulus / 1e6)}MPa.json"
        data = json.loads(path.read_text())
        records[str(int(modulus))] = {
            "k0_N_m2": data["k0_N_m2"],
            "passed": data["passed"],
            "source": path.relative_to(ROOT.parent).as_posix(),
            "source_sha256": _sha256(path),
            "sibling_hash_verified": True,
        }
    return {"source": V3_CALIBRATION.relative_to(ROOT.parent).as_posix(), "records": records}


def k0_for(calibration: dict, modulus: float) -> float:
    return float(calibration["records"][str(int(modulus))]["k0_N_m2"])


def make_mesh(h: float, half_domain: float, shape: str) -> np.ndarray:
    centres = np.unique(np.concatenate([CENTERS, -CENTERS]))
    return mesh_with_load_edges(
        -half_domain,
        half_domain,
        h,
        centres,
        load_width=dl.FOOTPRINT_M,
        extra_edges=np.array([1.2]) if shape == "asymmetric" else None,
    )


def make_depth(dz: float) -> np.ndarray:
    return np.linspace(0.0, dl.LAYER_DEPTH_M, round(dl.LAYER_DEPTH_M / dz) + 1)


class Budget:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.started = time.perf_counter()
        self.exhausted = False

    def check(self) -> bool:
        if time.perf_counter() - self.started > self.seconds:
            self.exhausted = True
        return self.exhausted

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def layer_case_strains(system, gauge_operator, rhs, freqs, alpha, beta):
    """Solve one assembled layer system at all frequencies (one LU each)."""
    out = {}
    for f_hz in freqs:
        omega = 2.0 * np.pi * f_hz
        started = time.perf_counter()
        u, diag = dl.solve_harmonic(
            system["K"], system["M"], rhs, system["free"], omega, alpha, beta
        )
        strain = gauge_operator @ u[: system["beam_basis"].N]
        out[f_hz] = {"strain": strain, "diag": diag, "seconds": time.perf_counter() - started}
    return out


def local_case_strains(system, gauge_operator, rhs, freqs, alpha, beta):
    out = {}
    for f_hz in freqs:
        omega = 2.0 * np.pi * f_hz
        u, diag = dl.solve_harmonic(
            system["K"], system["M"], rhs, system["free"], omega, alpha, beta
        )
        strain = gauge_operator @ u
        out[f_hz] = {"strain": strain, "diag": diag}
    return out


def assemble_pair(points, depth, shape, modulus, k0):
    support = asymmetric_shape if shape == "asymmetric" else intact_shape
    layer = dl.assemble_layer_system(
        points,
        depth,
        support,
        soil_E=modulus,
        support_discontinuity_points=np.array([1.2]) if shape == "asymmetric" else None,
    )
    local = dl.assemble_local_beam(points, support, k0=k0)
    return layer, local


def pair_strains(points, depth, shape, modulus, k0, freqs, alpha, beta):
    """Gauge strains of both model families for one (mesh, modulus, shape)."""
    layer, local = assemble_pair(points, depth, shape, modulus, k0)
    n_beam = layer["beam_basis"].N
    rhs_layer = np.zeros(layer["K"].shape[0])
    rhs_layer[:n_beam] = dl.beam_load_vector(
        layer["beam_basis"], CENTERS, dl.FOOTPRINT_M, [dl.FORCE_N]
    )[:, 0]
    rhs_local = dl.beam_load_vector(local["beam_basis"], CENTERS, dl.FOOTPRINT_M, [dl.FORCE_N])[:, 0]
    gauge_layer = dl.gauge_strain_matrix(layer["beam_basis"], dl.GAUGE_POINTS, dl.GAUGE_LENGTH_M)
    gauge_local = dl.gauge_strain_matrix(local["beam_basis"], dl.GAUGE_POINTS, dl.GAUGE_LENGTH_M)
    ref = layer_case_strains(layer, gauge_layer, rhs_layer, freqs, alpha, beta)
    pred = local_case_strains(local, gauge_local, rhs_local, freqs, alpha, beta)
    return ref, pred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-s", type=float, default=2700.0,
                        help="wall-clock budget for solves; remaining cases are "
                             "recorded as skipped_budget, never silently dropped")
    args = parser.parse_args()
    budget = Budget(args.budget_s)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    alpha, beta = dl.rayleigh_coefficients()
    calibration = load_calibration()
    k0 = {modulus: k0_for(calibration, modulus) for modulus in dl.SOIL_MODULI_PA}
    freqs = list(dl.FREQUENCIES_HZ)

    verification = {"rayleigh": {"alpha": alpha, "beta": beta, "zeta_target": dl.ZETA_TARGET,
                                 "f1_hz": dl.F1_HZ, "f2_hz": dl.F2_HZ}}

    # --- Verification at production resolution -----------------------------
    points_v = make_mesh(0.25, dl.HALF_DOMAIN_M, "asymmetric")
    depth_v = make_depth(0.25)
    zero_freq = {}
    for shape, modulus in (("intact", 50.0e6), ("asymmetric", 50.0e6), ("intact", 200.0e6)):
        if budget.check():
            break
        support = asymmetric_shape if shape == "asymmetric" else intact_shape
        layer, _ = assemble_pair(points_v, depth_v, shape, modulus, k0[modulus])
        static = cr.continuum_forward(
            points_v, depth_v, CENTERS, support, dl.GAUGE_POINTS, dl.GAUGE_LENGTH_M,
            dl.EI_N_M2, dl.INTERFACE_STIFFNESS_N_M2, modulus, dl.SOIL_NU, dl.WIDTH_M,
            dl.FORCE_N, dl.FOOTPRINT_M, dl.RADIUS_M,
            support_discontinuity_points=np.array([1.2]) if shape == "asymmetric" else None,
        )
        rhs = np.zeros(layer["K"].shape[0])
        rhs[: layer["beam_basis"].N] = dl.beam_load_vector(
            layer["beam_basis"], CENTERS, dl.FOOTPRINT_M, [dl.FORCE_N]
        )[:, 0]
        u, diag = dl.solve_harmonic(layer["K"], layer["M"], rhs, layer["free"], 0.0, 0.0, 0.0)
        gauge = dl.gauge_strain_matrix(layer["beam_basis"], dl.GAUGE_POINTS, dl.GAUGE_LENGTH_M)
        strain_dynamic = gauge @ u[: layer["beam_basis"].N]
        strain_static = np.asarray(static["strain"]).reshape(-1)
        err = dl.complex_l2_relative(strain_dynamic, strain_static)
        # Consistent-mass rigid translation totals on the production mesh.
        u_beam = np.zeros(layer["M_beam"].shape[0])
        u_beam[0::2] = 1.0
        beam_mass_total = float(np.ones(u_beam.size) @ (layer["M_beam"] @ u_beam))
        beam_mass_expected = dl.M_LINE_KG_M * (points_v[-1] - points_v[0])
        u_soil = np.zeros(layer["M_soil"].shape[0])
        u_soil[1::2] = 1.0
        soil_mass_total = float(np.ones(u_soil.size) @ (layer["M_soil"] @ u_soil))
        soil_mass_expected = (
            dl.RHO_SOIL_KG_M3 * dl.WIDTH_M * (points_v[-1] - points_v[0]) * dl.LAYER_DEPTH_M
        )
        zero_freq[f"E{int(modulus / 1e6)}MPa_{shape}"] = {
            "strain_relative_l2_vs_continuum_forward": err,
            "residual_relative": diag["residual_relative"],
            "beam_mass_relative_error": abs(beam_mass_total - beam_mass_expected) / beam_mass_expected,
            "soil_mass_relative_error": abs(soil_mass_total - soil_mass_expected) / soil_mass_expected,
        }
        print(f"[verify] omega=0 E={modulus/1e6:.0f}MPa {shape}: strain rel L2 = {err:.3e}", flush=True)
    verification["zero_frequency"] = zero_freq
    verification["budget_exhausted"] = budget.exhausted

    # --- Bounded mesh study h=0.5 -> 0.25 (-> 0.125 where needed) ----------
    mesh_rows = []
    mesh_cache: dict[tuple, dict] = {}
    for h in (0.5, 0.25):
        for modulus in dl.SOIL_MODULI_PA:
            for shape in SHAPES:
                if budget.check():
                    mesh_rows.append({"h_m": h, "soil_E_Pa": modulus, "shape": shape,
                                      "status": "skipped_budget"})
                    continue
                points = make_mesh(h, dl.HALF_DOMAIN_M, shape)
                started = time.perf_counter()
                ref, pred = pair_strains(points, make_depth(h), shape, modulus,
                                         k0[modulus], freqs, alpha, beta)
                mesh_cache[(h, modulus, shape)] = (ref, pred)
                print(f"[mesh] h={h} E={modulus/1e6:.0f}MPa {shape} "
                      f"({time.perf_counter() - started:.1f}s)", flush=True)
    production: dict[tuple, dict] = {}
    for modulus in dl.SOIL_MODULI_PA:
        for shape in SHAPES:
            key = (0.25, modulus, shape)
            if key not in mesh_cache:
                continue
            ref05, _p1 = mesh_cache.get((0.5, modulus, shape), (None, None))
            ref25, pred25 = mesh_cache[key]
            production[(modulus, shape)] = {"ref": ref25, "pred": pred25, "h": 0.25}
            for f_hz in freqs:
                row = {"h_m": 0.25, "soil_E_Pa": modulus, "shape": shape, "freq_hz": f_hz}
                if ref05 is None:
                    row["status"] = "skipped_budget"
                else:
                    row["delta_layer_0.5_to_0.25"] = dl.complex_l2_relative(
                        ref05[f_hz]["strain"], ref25[f_hz]["strain"])
                    row["delta_local_0.5_to_0.25"] = dl.complex_l2_relative(
                        _p1[f_hz]["strain"], pred25[f_hz]["strain"])
                    row["status"] = "ok"
                mesh_rows.append(row)
    # Refine conditions whose layer change exceeds the v3 acceptance level.
    for modulus in dl.SOIL_MODULI_PA:
        for shape in SHAPES:
            for f_hz in freqs:
                match = [r for r in mesh_rows
                         if r.get("soil_E_Pa") == modulus and r.get("shape") == shape
                         and r.get("freq_hz") == f_hz and r.get("h_m") == 0.25]
                if not match or match[0].get("status") != "ok":
                    continue
                if match[0]["delta_layer_0.5_to_0.25"] <= CONVERGENCE_TOL:
                    match[0]["converged_at_production"] = True
                    continue
                if budget.check():
                    match[0]["status"] = "refinement_skipped_budget"
                    match[0]["converged_at_production"] = False
                    continue
                points = make_mesh(0.125, dl.HALF_DOMAIN_M, shape)
                started = time.perf_counter()
                ref12, pred12 = pair_strains(points, make_depth(0.125), shape, modulus,
                                             k0[modulus], [f_hz], alpha, beta)
                ref25 = production[(modulus, shape)]["ref"]
                match[0]["delta_layer_0.25_to_0.125"] = dl.complex_l2_relative(
                    ref25[f_hz]["strain"], ref12[f_hz]["strain"])
                match[0]["delta_local_0.25_to_0.125"] = dl.complex_l2_relative(
                    production[(modulus, shape)]["pred"][f_hz]["strain"], pred12[f_hz]["strain"])
                match[0]["converged_at_production"] = bool(
                    match[0]["delta_layer_0.25_to_0.125"] <= CONVERGENCE_TOL)
                match[0]["refinement_seconds"] = time.perf_counter() - started
                print(f"[refine] E={modulus/1e6:.0f}MPa {shape} f={f_hz}Hz "
                      f"delta025->0125={match[0]['delta_layer_0.25_to_0.125']:.3e}", flush=True)

    # --- Production metrics --------------------------------------------------
    case_rows = []
    fields: dict[str, np.ndarray] = {"x": dl.GAUGE_POINTS}
    case_keys = []
    for modulus in dl.SOIL_MODULI_PA:
        for shape in SHAPES:
            if (modulus, shape) not in production:
                for f_hz in freqs:
                    case_rows.append({"soil_E_Pa": modulus, "shape": shape, "freq_hz": f_hz,
                                      "status": "skipped_budget"})
                continue
            ref = production[(modulus, shape)]["ref"]
            pred = production[(modulus, shape)]["pred"]
            for f_hz in freqs:
                key = f"E{int(modulus / 1e6)}MPa_{shape}_f{f_hz:g}Hz"
                case_keys.append(key)
                s_ref = ref[f_hz]["strain"]
                s_pred = pred[f_hz]["strain"]
                ratio = dl.amplitude_ratio(s_pred, s_ref)
                fields[f"ref_{key}"] = s_ref
                fields[f"pred_{key}"] = s_pred
                fields[f"ampratio_{key}"] = ratio
                conv = [r for r in mesh_rows
                        if r.get("soil_E_Pa") == modulus and r.get("shape") == shape
                        and r.get("freq_hz") == f_hz and r.get("h_m") == 0.25]
                case_rows.append({
                    "case_id": key,
                    "soil_E_Pa": modulus,
                    "shape": shape,
                    "freq_hz": f_hz,
                    "complex_L2": dl.complex_l2_relative(s_pred, s_ref),
                    "amp_norm_ratio": float(np.linalg.norm(s_pred) / np.linalg.norm(s_ref)),
                    "amp_ratio_max": float(np.max(ratio)),
                    "amp_ratio_min": float(np.min(ratio)),
                    "mesh_h_m": 0.25,
                    "converged": conv[0].get("converged_at_production") if conv else None,
                    "layer_residual_relative": ref[f_hz]["diag"]["residual_relative"],
                    "layer_power_balance_relative_error": ref[f_hz]["diag"]["power_balance_relative_error"],
                    "layer_external_power_W": ref[f_hz]["diag"]["external_power_W"],
                    "local_residual_relative": pred[f_hz]["diag"]["residual_relative"],
                    "local_power_balance_relative_error": pred[f_hz]["diag"]["power_balance_relative_error"],
                    "status": "ok",
                })
                print(f"[case] {key}: E_L2={case_rows[-1]['complex_L2']:.4f} "
                      f"ratio={case_rows[-1]['amp_norm_ratio']:.4f}", flush=True)
    fields["case_keys"] = np.array(case_keys)
    for shape in SHAPES:
        support = asymmetric_shape if shape == "asymmetric" else intact_shape
        fields[f"loss_{shape}"] = np.asarray(support(dl.GAUGE_POINTS), dtype=float)

    # --- Boundary sensitivity +/-50 vs +/-60 (intact, both moduli) ---------
    boundary_rows = []
    for modulus in dl.SOIL_MODULI_PA:
        for f_hz in freqs:
            if budget.check():
                boundary_rows.append({"soil_E_Pa": modulus, "freq_hz": f_hz,
                                      "status": "skipped_budget"})
                continue
            points = make_mesh(0.25, 60.0, "intact")
            ref60, pred60 = pair_strains(points, make_depth(0.25), "intact", modulus,
                                         k0[modulus], [f_hz], alpha, beta)
            base = production.get((modulus, "intact"))
            row = {"soil_E_Pa": modulus, "freq_hz": f_hz, "h_m": 0.25,
                   "half_domain_compare_m": [50, 60], "status": "ok"}
            if base is not None:
                row["delta_layer_L60_vs_L50"] = dl.complex_l2_relative(
                    ref60[f_hz]["strain"], base["ref"][f_hz]["strain"])
                row["delta_local_L60_vs_L50"] = dl.complex_l2_relative(
                    pred60[f_hz]["strain"], base["pred"][f_hz]["strain"])
            boundary_rows.append(row)
            print(f"[boundary] E={modulus/1e6:.0f}MPa f={f_hz}Hz "
                  f"delta={row.get('delta_layer_L60_vs_L50', float('nan')):.3e}", flush=True)

    # --- Save raw data -------------------------------------------------------
    np.savez_compressed(RESULTS / "layer_fields.npz", **fields)

    def write_csv(path: Path, rows: list[dict]) -> None:
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(RESULTS / "layer_metrics.csv", case_rows)
    write_csv(RESULTS / "mesh_study.csv", mesh_rows)
    write_csv(RESULTS / "boundary_sensitivity.csv", boundary_rows)

    # --- Figure ---------------------------------------------------------------
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-layer")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    labels = {
        (50.0e6, "intact"): "E=50 MPa, intact",
        (50.0e6, "asymmetric"): "E=50 MPa, asymmetric",
        (200.0e6, "intact"): "E=200 MPa, intact",
        (200.0e6, "asymmetric"): "E=200 MPa, asymmetric",
    }
    markers = {(50.0e6, "intact"): "o", (50.0e6, "asymmetric"): "s",
               (200.0e6, "intact"): "^", (200.0e6, "asymmetric"): "D"}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ok_rows = [r for r in case_rows if r.get("status") == "ok"]
    for (modulus, shape), label in labels.items():
        rows = sorted((r for r in ok_rows if r["soil_E_Pa"] == modulus and r["shape"] == shape),
                      key=lambda r: r["freq_hz"])
        if not rows:
            continue
        f = [r["freq_hz"] for r in rows]
        e = [r["complex_L2"] for r in rows]
        r_norm = [r["amp_norm_ratio"] for r in rows]
        conv = [bool(r["converged"]) for r in rows]
        marker = markers[(modulus, shape)]
        for ax, values in ((axes[0], e), (axes[1], r_norm)):
            faced = [marker if c else "none" for c in conv]
            ax.plot(f, values, marker=marker, markersize=4, linewidth=1.0, label=label)
            for fi, vi, ci in zip(f, values, conv):
                if not ci:
                    ax.plot([fi], [vi], marker=marker, markersize=7,
                            markerfacecolor="none", markeredgecolor="red", linewidth=0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel(r"complex $L_2$ mismatch $E$")
    axes[1].axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel(r"$\|\epsilon_{local}\|_2\,/\,\|\epsilon_{layer}\|_2$")
    for ax in axes:
        ax.set_xticks(list(freqs))
        ax.tick_params(which="both", direction="in", top=True, right=True)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig_path = FIGURES / "layer_transfer_mismatch.pdf"
    fig.savefig(fig_path)
    plt.close(fig)
    write_csv(RESULTS / "layer_figure_data.csv",
              [{"case_id": r["case_id"], "freq_hz": r["freq_hz"],
                "complex_L2": r["complex_L2"], "amp_norm_ratio": r["amp_norm_ratio"],
                "converged": r["converged"]} for r in ok_rows])

    # --- Report ---------------------------------------------------------------
    worst_residual = max((r["layer_residual_relative"] for r in ok_rows), default=None)
    worst_power = max((r["layer_power_balance_relative_error"] for r in ok_rows), default=None)
    report = {
        "task": "audit/task_layer.md (DYNAMIC_PROTOCOL step 4)",
        "created_epoch_s": time.time(),
        "elapsed_s": budget.elapsed,
        "budget_s": args.budget_s,
        "budget_exhausted": budget.exhausted,
        "parameters": {
            "m_line_kg_m": dl.M_LINE_KG_M, "rho_soil_kg_m3": dl.RHO_SOIL_KG_M3,
            "zeta_target": dl.ZETA_TARGET, "rayleigh_alpha": alpha, "rayleigh_beta": beta,
            "rayleigh_rule": "C = alpha M + beta K, exact zeta at 2 and 20 Hz, identical rule and coefficients in both model families",
            "force_N": dl.FORCE_N, "footprint_m": dl.FOOTPRINT_M, "load_center_m": dl.LOAD_CENTER_M,
            "gauge_length_m": dl.GAUGE_LENGTH_M, "gauge_points": "61 in [-12, 12] m",
            "EI_N_m2": dl.EI_N_M2, "radius_m": dl.RADIUS_M,
            "interface_stiffness_N_m2": dl.INTERFACE_STIFFNESS_N_M2,
            "soil_nu": dl.SOIL_NU, "width_m": dl.WIDTH_M, "layer_depth_m": dl.LAYER_DEPTH_M,
            "half_domain_m": dl.HALF_DOMAIN_M, "frequencies_hz": freqs,
            "production_h_m": 0.25, "mesh_study_h_m": [0.5, 0.25, 0.125],
            "convergence_tol_complex_L2": CONVERGENCE_TOL,
        },
        "calibration": calibration,
        "transfer_assumption": TRANSFER_ASSUMPTION,
        "boundary_condition_caveat": BC_CAVEAT,
        "no_half_space_impedance_claim": True,
        "verification": verification,
        "worst_production_layer_residual_relative": worst_residual,
        "worst_production_layer_power_balance_relative_error": worst_power,
        "case_metrics": ok_rows,
        "mesh_study": mesh_rows,
        "boundary_sensitivity": boundary_rows,
        "outputs": {
            "fields_npz": "results/layer/layer_fields.npz",
            "metrics_csv": "results/layer/layer_metrics.csv",
            "mesh_study_csv": "results/layer/mesh_study.csv",
            "boundary_csv": "results/layer/boundary_sensitivity.csv",
            "figure": "manuscript/figures_v5/layer_transfer_mismatch.pdf",
            "figure_data_csv": "results/layer/layer_figure_data.csv",
        },
        "commands": {
            "run": ("PYTHONPATH=code:vendor OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-layer "
                    "python code/dynamic_layer_run.py"),
            "test": ("PYTHONPATH=code:vendor OMP_NUM_THREADS=1 "
                     "python -m pytest "
                     "tests/test_dynamic_layer.py -q"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "skfem": __import__("skfem").__version__,
            "matplotlib": matplotlib.__version__,
        },
        "deviations": [],
    }
    (AUDIT / "layer_report.json").write_text(json.dumps(report, indent=2))
    (AUDIT / "layer_verification.json").write_text(json.dumps(verification, indent=2))
    print(f"[done] elapsed={budget.elapsed:.1f}s budget_exhausted={budget.exhausted}")
    print(f"[done] report: {AUDIT / 'layer_report.json'}")
    print(f"[done] figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
