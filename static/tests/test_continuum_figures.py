from pathlib import Path
import sys

import numpy as np
import pytest


CODE = Path(__file__).resolve().parents[1] / "code"
VENDOR = Path(__file__).resolve().parents[1] / "vendor"
for directory in (CODE, VENDOR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import make_continuum_figures as figures  # noqa: E402


def _row(method: str, seed: int, value: float | None, *, selected: bool = True) -> dict:
    compatibility = {
        "selection_compatible": True if selected else None,
        "optimizer_failed": not selected,
        "no_feasible_candidate": False,
        "path_bracketed": selected,
        "unbracketed_high": selected and seed == 1203,
        "selection_reason": "largest_feasible_lambda" if selected else "optimizer_failed",
    }
    return {
        "dataset_key": f"small-{method}-{seed}",
        "path_key": f"small-{method}-{seed}-path",
        "method": method,
        "configuration": {
            "shape": "asymmetric",
            "soil_E_Pa": 50.0e6,
            "soil_E_MPa": 50.0,
            "tau": 0.10,
            "tau_percent": 10.0,
            "seed": seed,
            "sigma": 5.0e-6,
            "sensors": 20,
        },
        "status": "selected" if selected else "optimizer_failed",
        "selected": selected,
        "compatibility": compatibility,
        "selected_gamma": [1.0, 1.0, 1.0] if selected else None,
        "metrics": {
            "inferred_effective_winkler_loss_centroid_m": value,
            "unused_load_strain_relative_L2": None if value is None else value / 100.0,
        },
        "candidate_counts": {
            "candidates": 6,
            "successful_candidates": 6 if selected else 0,
            "optimizer_failed_candidates": 0 if selected else 6,
            "starts": 12,
            "successful_starts": 12 if selected else 0,
            "failed_starts": 0 if selected else 12,
        },
        "record_hashes": {"path.json": "a" * 64, "selected_arrays.npz": "b" * 64 if selected else None},
    }


def _small_bundle(rows: list[dict]) -> dict:
    candidate = {
        "successful_start": all(row["selected"] for row in rows),
        "starts": [
            {"success": row["selected"]}
            for row in rows
            if row["configuration"]["seed"] == rows[0]["configuration"]["seed"]
        ],
    }
    return {
        "results_dir": Path("/tmp/not-a-formal-result"),
        "continuum": {"interpretation": "unit fixture only"},
        "verification": {
            "passed": True,
            "checked_candidates": 1,
            "checked_paths": 1,
        },
        "candidate_records": {row["path_key"]: [candidate] for row in rows},
        "input_hashes": {"unit": "fixture"},
        "calibrated_k0": {50.0e6: 1.7e7},
    }


def test_summary_groups_nulls_and_accounts_failures_without_imputation():
    rows = [
        _row("fixed", 1201, 1.0),
        _row("fixed", 1202, 2.0),
        _row("fixed", 1203, None, selected=False),
        _row("free", 1201, 10.0),
        _row("free", 1202, 20.0),
        _row("free", 1203, 30.0),
    ]
    summary = figures.build_statistical_summary(_small_bundle(rows), rows)
    fixed = next(group for group in summary["groups"] if group["method"] == "fixed")
    free = next(group for group in summary["groups"] if group["method"] == "free")
    assert fixed["counts"] == {
        "selected": 2,
        "optimizer_failed": 1,
        "incompatible": 0,
        "unbracketed_high": 0,
    }
    assert fixed["metrics"]["inferred_effective_winkler_loss_centroid_m"] == {
        **fixed["metrics"]["inferred_effective_winkler_loss_centroid_m"],
        "n": 3,
        "n_valid": 2,
        "null_count": 1,
    }
    assert fixed["metrics"]["inferred_effective_winkler_loss_centroid_m"]["median"] == pytest.approx(1.5)
    assert free["metrics"]["inferred_effective_winkler_loss_centroid_m"]["null_count"] == 0
    assert free["counts"]["unbracketed_high"] == 1
    assert summary["counts"]["optimizer_failed_paths"] == 1
    assert summary["counts"]["selected_reasons"]["optimizer_failed"] == 1
    assert not figures.FORBIDDEN_RELATIVE_METRICS.intersection(fixed["metrics"])


def test_gates_reject_incomplete_unverified_or_wrong_count_results():
    status = {
        "schema_version": "continuum-run-status-v3.0",
        "status": "completed",
        "completed_paths": 144,
        "expected_counts": figures.EXPECTED_COUNTS,
    }
    verification = {
        "schema_version": "continuum-verification-v3.0",
        "passed": True,
        "incomplete": False,
        "checked_candidates": 864,
        "checked_paths": 144,
        "expected_counts": figures.EXPECTED_COUNTS,
        "mesh_passed": True,
        "calibration_passed": True,
        "inventory": {"complete": True},
        "errors": {},
    }
    figures._validate_gates(status, verification)
    for mutation in (
        lambda report: report.update(passed=False),
        lambda report: report.update(incomplete=True),
        lambda report: report.update(checked_candidates=863),
        lambda report: report.update(errors={"path": ["missing"]}),
        lambda report: report["inventory"].update(complete=False),
    ):
        changed = {key: (dict(value) if isinstance(value, dict) else value) for key, value in verification.items()}
        mutation(changed)
        with pytest.raises(figures.SummaryError, match="continuum gates"):
            figures._validate_gates(status, changed)


def _unit_reference_fields() -> dict[float, dict[str, np.ndarray]]:
    x = np.linspace(-14.0, 14.0, 15)
    z = np.linspace(0.0, 10.0, 6)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    nodes = np.column_stack((xx.ravel(), zz.ravel()))
    quads = []
    for i in range(x.size - 1):
        for j in range(z.size - 1):
            lower_left = i * z.size + j
            quads.append(
                (
                    lower_left,
                    lower_left + z.size,
                    lower_left + z.size + 1,
                    lower_left + 1,
                )
            )
    uz = np.zeros((nodes.shape[0], 4))
    uz[:, 1] = 2.0 * np.exp(-(nodes[:, 0] ** 2) / 18.0) * (1.0 - nodes[:, 1] / 10.0)
    return {
        50.0e6: {"soil_nodes": nodes, "soil_quads": np.asarray(quads), "soil_uz": uz},
        200.0e6: {
            "soil_nodes": nodes,
            "soil_quads": np.asarray(quads),
            "soil_uz": uz * [1, 0.7, 1, 1],
        },
    }


def test_reference_figure_has_exact_pdf_contract_and_rejects_overwrite(tmp_path):
    report = figures.render_reference_figure(_unit_reference_fields(), tmp_path)
    pdf = Path(report["pdf"])
    png = Path(report["png"])
    assert pdf.is_file() and png.is_file()
    assert report["inspection"]["width_inches"] == figures.style.FIGURE_WIDTH_INCHES
    assert report["inspection"]["text_sizes_pt"] == [figures.style.FONT_SIZE_PT]
    assert report["inspection"]["pdf_fonttype"] == 42
    assert report["inspection"]["mathtext_fontset"] == "cm"
    assert report["inspection"]["media_box_width_pt"] == pytest.approx(468.0)
    assert any("cmr10" in font.lower() for font in report["inspection"]["fonts"])
    assert report["layout_warnings"] == []
    with pytest.raises(figures.SummaryError, match="refusing to overwrite"):
        figures.render_reference_figure(_unit_reference_fields(), tmp_path)


def test_reference_schematic_uses_B_width_shows_clamps_and_actual_color_range(tmp_path, monkeypatch):
    from matplotlib.collections import QuadMesh

    captured = {}
    real_save = figures.style.save_and_audit_figure

    def spy(fig, out_dir, stem):
        captured["fig"] = fig
        return real_save(fig, out_dir, stem)

    monkeypatch.setattr(figures.style, "save_and_audit_figure", spy)
    fields = _unit_reference_fields()
    figures.render_reference_figure(fields, tmp_path)
    fig = captured["fig"]
    texts = "\n".join(text.get_text() for axis in fig.axes for text in axis.texts)
    # out-of-plane beam width is uppercase B; lowercase b is the loading footprint
    assert "$B=0.6$" in texts
    assert "$b=0.6$" not in texts
    assert "$b=4.3$" in texts
    # clamped beam ends are shown and labelled, distinct from soil side rollers
    assert "w=w'" in texts
    assert "$u_x=u_z=0$" in texts
    # common color range covers the actual min/max of both fields, including 0
    values = np.concatenate(
        [
            np.asarray(arrays["soil_uz"][:, 1], dtype=float).ravel() * 1000.0
            for arrays in fields.values()
        ]
    )
    meshes = [
        collection
        for axis in fig.axes
        for collection in axis.collections
        if isinstance(collection, QuadMesh)
    ]
    assert meshes
    expected = (min(0.0, float(values.min())), max(0.0, float(values.max())))
    for mesh in meshes:
        assert mesh.get_clim() == pytest.approx(expected)


def _prediction_fixture_rows() -> list[dict]:
    rows = []
    for shape in ("asymmetric", "broad_shallow"):
        for soil_e in (50.0e6, 200.0e6):
            for tau in (0.01, 0.10):
                for method in ("fixed", "free", "calibrated", "oracle"):
                    for seed in (1201, 1202, 1203):
                        row = _row(method, seed, float(seed - 1200))
                        row["configuration"].update(
                            shape=shape,
                            soil_E_Pa=soil_e,
                            soil_E_MPa=soil_e / 1.0e6,
                            tau=tau,
                            tau_percent=100.0 * tau,
                        )
                        rows.append(row)
    return rows


def test_prediction_and_response_figures_meet_pdf_and_layout_contract(tmp_path):
    prediction_rows = _prediction_fixture_rows()
    prediction = figures.make_prediction_figure(prediction_rows, tmp_path)

    response_x = np.linspace(-12.0, 12.0, 241)
    reference = np.sin(response_x / 2.0) * np.exp(-np.abs(response_x) / 10.0)
    response_rows = []
    response_bundle = {"selected_arrays": {}}
    for soil_e in (50.0e6, 200.0e6):
        for method in ("fixed", "free", "calibrated", "oracle"):
            row = _row(method, 1201, 1.0)
            row["configuration"].update(
                soil_E_Pa=soil_e,
                soil_E_MPa=soil_e / 1.0e6,
                tau=0.10,
                tau_percent=10.0,
            )
            response_rows.append(row)
            response_bundle["selected_arrays"][row["path_key"]] = {
                "unused_x": response_x,
                "reference_strain": reference,
                "predicted_strain": reference * (0.90 if method == "fixed" else 0.95),
            }
    response = figures.make_response_figure(response_bundle, response_rows, tmp_path)

    for report in (prediction, response):
        assert report["inspection"]["width_inches"] == figures.style.FIGURE_WIDTH_INCHES
        assert report["inspection"]["text_sizes_pt"] == [figures.style.FONT_SIZE_PT]
        assert report["inspection"]["pdf_fonttype"] == 42
        assert report["inspection"]["mathtext_fontset"] == "cm"
        assert report["inspection"]["media_box_width_pt"] == pytest.approx(468.0)
        assert report["layout_warnings"] == []
def test_predesignated_figure_data_fail_rather_than_substitute(tmp_path):
    rows = [_row("fixed", 1201, 1.0)]
    with pytest.raises(figures.SummaryError, match="missing or duplicated"):
        figures._exact_rows(
            rows,
            shape="asymmetric",
            soil_e=50.0e6,
            tau=0.10,
            method="fixed",
        )


def test_prediction_and_response_xlabels_use_physical_semantics(tmp_path, monkeypatch):
    captured = []
    real_save = figures.style.save_and_audit_figure

    def spy(fig, out_dir, stem):
        captured.append((stem, fig))
        return real_save(fig, out_dir, stem)

    monkeypatch.setattr(figures.style, "save_and_audit_figure", spy)

    figures.make_prediction_figure(_prediction_fixture_rows(), tmp_path)
    response_x = np.linspace(-12.0, 12.0, 241)
    reference = np.sin(response_x / 2.0) * np.exp(-np.abs(response_x) / 10.0)
    response_rows = []
    response_bundle = {"selected_arrays": {}}
    for soil_e in (50.0e6, 200.0e6):
        for method in ("fixed", "free", "calibrated", "oracle"):
            row = _row(method, 1201, 1.0)
            row["configuration"].update(
                soil_E_Pa=soil_e,
                soil_E_MPa=soil_e / 1.0e6,
                tau=0.10,
                tau_percent=10.0,
            )
            response_rows.append(row)
            response_bundle["selected_arrays"][row["path_key"]] = {
                "unused_x": response_x,
                "reference_strain": reference,
                "predicted_strain": reference * (0.90 if method == "fixed" else 0.95),
            }
    figures.make_response_figure(response_bundle, response_rows, tmp_path)

    stems = {stem for stem, _ in captured}
    assert stems == {"continuum_prediction", "continuum_response"}
    for stem, fig in captured:
        xlabels = {axis.get_xlabel() for axis in fig.axes if axis.get_visible()} - {""}
        if stem == "continuum_prediction":
            # tau is the measured load uncertainty level, not a treatment-assigned value
            assert xlabels == {r"Load measurement uncertainty $\tau$ (%)"}
        else:
            # load centre is fixed; x is the strain observation coordinate
            assert xlabels == {"Position $x$ (m)"}
