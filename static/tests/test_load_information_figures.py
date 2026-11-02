"""Bounded tests for the v3 load-information figure/postprocessing module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import make_load_information_figures as figures


def _row(dataset: str, method: str, value: float | None) -> dict:
    return {
        "dataset_key": dataset,
        "path_key": f"{dataset}__method-{method}",
        "configuration": {"shape": "asymmetric", "tau": 0.05, "seed": int(dataset[-1])},
        "memberships": ["primary"],
        "method": method,
        "method_family": "primary_or_resolution",
        "selected": value is not None,
        "status": "selected" if value is not None else "optimizer_failed",
        "metrics": {"support_rel_L2_except_intact": value},
    }


def test_aggregate_metric_counts_null_and_failed_paths_without_imputing():
    rows = [_row("shape-a_seed-1", "calibrated", 1.0), _row("shape-a_seed-2", "calibrated", None),
            _row("shape-a_seed-3", "calibrated", 3.0), _row("shape-a_seed-4", "calibrated", 4.0)]
    summary = figures.aggregate_metric(rows, "support_rel_L2_except_intact")
    assert summary["n"] == 4
    assert summary["n_valid"] == 3
    assert summary["null_count"] == 1
    assert summary["failed_path_count"] == 1
    assert summary["median"] == pytest.approx(3.0)
    assert summary["q25"] == pytest.approx(float(np.percentile([1.0, 3.0, 4.0], 25)))
    assert [item["dataset_key"] for item in summary["replicates"]] == [
        "shape-a_seed-1", "shape-a_seed-3", "shape-a_seed-4"
    ]


def test_paired_comparison_uses_exact_dataset_key_and_counts_null_pairs():
    rows = [
        _row("shape-a_seed-1", "calibrated", 2.0),
        _row("shape-a_seed-1", "fixed", 4.0),
        _row("shape-a_seed-2", "calibrated", 1.0),
        _row("shape-a_seed-2", "fixed", None),
        _row("shape-a_seed-3", "calibrated", 3.0),
    ]
    result = figures.paired_metric_comparison(
        rows, "calibrated", "fixed", "support_rel_L2_except_intact"
    )
    assert result["n_dataset_keys"] == 3
    assert result["n_valid_pairs"] == 1
    assert result["missing_method_datasets"] == ["shape-a_seed-3"]
    assert result["null_pair_datasets"] == ["shape-a_seed-2"]
    assert result["pairs"] == [
        {
            "dataset_key": "shape-a_seed-1",
            "seed": 1,
            "calibrated": 2.0,
            "fixed": 4.0,
            "delta_a_minus_b": -2.0,
        }
    ]


def test_verification_rejects_incomplete_or_unsuccessful_report():
    report = {
        "schema_version": "load-information-verification-v3.0",
        "passed": True,
        "incomplete": False,
        "checked_candidates": figures.EXPECTED_COUNTS["candidates"] - 1,
        "checked_completed_paths": figures.EXPECTED_COUNTS["paths"],
        "expected_candidates": figures.EXPECTED_COUNTS["candidates"],
        "expected_paths": figures.EXPECTED_COUNTS["paths"],
        "errors": {},
        "incomplete_paths": [],
    }
    with pytest.raises(figures.SummaryError, match="checked_candidates=5087"):
        figures._validate_verification(report)


def test_current_design_expands_to_required_campaign_size():
    datasets, paths = figures._validate_expected_expansion(figures.runner.load_design())
    assert len(datasets) == 200
    assert len(paths) == 848
    assert figures.EXPECTED_COUNTS["candidates"] == 5088
    assert figures.EXPECTED_COUNTS["starts"] == 10176


def _candidate(penalty: float, success: bool) -> dict:
    return {
        "penalty_lambda": penalty,
        "successful_start": success,
        "chosen_start": 0,
        "starts": [{"success": success}, {"success": not success}],
        "elapsed_s": 0.25,
    }


def test_failed_path_remains_accounted_with_null_metrics_and_candidate_status():
    dataset = {
        "key": "shape-a_tau-0.05_sigma-5e-06_sensors-20_seed-1",
        "shape": "asymmetric", "tau": 0.05, "sigma": 5e-6, "sensors": 20, "seed": 1,
        "memberships": ["primary"],
    }
    dataset_record = {
        "measured_load_mean": [1.0, 1.0, 1.0],
        "true_load_standard_uncertainty": 0.05,
    }
    path_config = {"key": "failed__method-free", "method": "free", "method_family": "primary_or_resolution"}
    path = {
        "status": "optimizer_failed",
        "selected": False,
        "metrics": {name: None for name in figures.METRICS},
        "selection": {
            "selected_lambda": 0.03,
            "optimizer_failed": True,
            "no_feasible_candidate": False,
            "path_bracketed": False,
            "unbracketed_high": False,
            "selection_reason": "optimizer_failed",
            "N_strain_observations": 60,
            "M_calibration_observations": 0,
            "threshold": 1.2 * np.sqrt(60),
            "selection_norm": 10.0,
            "data_norm": 10.0,
            "calibration_norm": 0.0,
        },
    }
    candidates = [_candidate(0.03, False), _candidate(0.3, True)]
    summary = figures._path_summary(dataset, dataset_record, path_config, path, candidates)
    assert summary["selected"] is False
    assert summary["metrics"]["support_rel_L2_except_intact"] is None
    assert summary["compatibility"]["optimizer_failed"] is True
    assert summary["candidate_statuses"] == {
        figures.runner._format_number(0.03): "optimizer_failed",
        figures.runner._format_number(0.3): "successful",
    }
    assert summary["counts"]["candidates"] == 2
    assert summary["counts"]["successful_candidates"] == 1
    assert summary["counts"]["optimizer_failed_candidates"] == 1
    assert summary["counts"]["successful_starts"] == 2
    assert summary["counts"]["failed_starts"] == 2


def test_model_figure_has_exact_width_fonts_and_layout_report(tmp_path):
    report = figures.make_model_figure(tmp_path)
    pdf = Path(report["pdf"])
    png = Path(report["png"])
    assert pdf.is_file() and png.is_file()
    assert report["inspection"]["width_inches"] == figures.FIGURE_WIDTH_INCHES
    assert report["inspection"]["text_sizes_pt"] == [figures.FONT_SIZE_PT]
    assert report["inspection"]["pdf_fonttype"] == 42
    assert report["inspection"]["mathtext_fontset"] == "cm"
    assert report["inspection"]["media_box_width_pt"] == pytest.approx(468.0)
    assert any("cmr10" in font.lower() for font in report["inspection"]["fonts"])
    assert report["layout_warnings"] == []

    # The command must never silently replace an existing deliverable.
    with pytest.raises(figures.SummaryError, match="refusing to overwrite"):
        figures.make_model_figure(tmp_path)


def test_model_figure_semantics_and_three_panel_layout(tmp_path, monkeypatch):
    captured = {}
    original = figures.save_and_audit_figure

    def spy(fig, out_dir, stem):
        captured["axes_count"] = len(fig.axes)
        captured["texts"] = [axis.get_title() for axis in fig.axes] + [
            text.get_text() for axis in fig.axes for text in axis.texts
        ]
        return original(fig, out_dir, stem)

    monkeypatch.setattr(figures, "save_and_audit_figure", spy)
    figures.make_model_figure(tmp_path)
    assert captured["axes_count"] == 3
    joined = "\n".join(captured["texts"])
    # Panels: mechanical configuration, explicit separate load states, process.
    assert "Mechanical configuration" in joined
    assert "Separate load states" in joined
    assert "Load-information process" in joined
    # Physical definitions that the first accepted diagram omitted.
    assert "$g=0.20$ m" in joined
    assert "$[-12\\,\\mathrm{m},\\,12\\,\\mathrm{m}]$" in joined
    assert "$b=4.3$ m footprint" in joined
    assert "clamped remote end" in joined
    for centre in ("$-4$ m", "$0$ m", "$+4$ m", "$+6$ m"):
        assert f"centre {centre}" in joined
    # The amplitude shortcut formula belongs to the manuscript, not the diagram.
    assert "\\hat\\gamma=(" not in joined
    assert "not to scale" in joined


def test_information_analysis_is_analytic_and_marks_actual_tau_levels():
    information = figures._information_curves()
    assert len(information["curves"]) == 9
    assert information["definition"].startswith("chi_j=")
    for curve in information["curves"]:
        tau = np.asarray(curve["tau"])
        chi = np.asarray(curve["chi"])
        assert np.all((chi > 0.0) & (chi <= 1.0))
        assert chi[0] == 1.0
        assert np.all(np.diff(chi) <= 0.0)
        assert np.allclose(chi, 1.0 / (1.0 + tau**2 * curve["response_norm2_whitened"]))


def test_json_output_rejects_nan_and_uses_null_for_intact_relative_metric():
    value = {"relative": None, "invalid": float("nan")}
    with pytest.raises(figures.runner.ProtocolError, match="NaN"):
        figures.runner.safe_json(value)
    assert json.loads(figures.runner.canonical_json(figures.runner.safe_json({"relative": None}))) == {
        "relative": None
    }


def test_figure_audit_helper_is_vendored_without_absolute_user_path():
    module_path = Path(figures.figure_layout_audit.__file__).resolve()
    assert module_path == (figures.ROOT / "vendor" / "figure_layout_audit.py").resolve()
    provenance = (figures.ROOT / "vendor" / "FIGURE_AUDIT_PROVENANCE.md").read_text()
    assert "paper-figure-consistency" in provenance
    source = (figures.ROOT / "code" / "make_load_information_figures.py").read_text()
    assert "/Users/" not in source


def _confidence_row(shape: str, method: str, family: str, value: float, seed: int) -> dict:
    return {
        "dataset_key": f"{shape}_seed-{seed}",
        "path_key": f"{shape}_seed-{seed}__method-{method}",
        "configuration": {
            "shape": shape,
            "tau": 0.05,
            "sigma": figures.FIELDS_SIGMA,
            "sigma_microstrain": figures.FIELDS_SIGMA * 1.0e6,
            "sensors": figures.FIELDS_SENSORS,
            "seed": seed,
        },
        "memberships": ["primary"],
        "method": method,
        "method_family": family,
        "selected": True,
        "status": "selected",
        "metrics": {"support_rel_L2_except_intact": value},
    }


def _confidence_rows() -> list[dict]:
    rows = []
    for shape in ("asymmetric", "broad_shallow"):
        for index, seed in enumerate((1101, 1102, 1103)):
            # Distinct medians per treatment expose any cross-series mix-up.
            rows.append(
                _confidence_row(shape, "overconfident", "confidence_sensitivity", 0.10 + 0.02 * index, seed)
            )
            rows.append(
                _confidence_row(shape, "conservative", "confidence_sensitivity", 0.30 + 0.02 * index, seed)
            )
            # The factor-1 calibrated treatment is the reused primary path.
            rows.append(
                _confidence_row(shape, "calibrated", "primary_or_resolution", 0.20 + 0.02 * index, seed)
            )
            rows.append(_confidence_row(shape, "fixed", "primary_or_resolution", 0.40 + 0.02 * index, seed))
            rows.append(_confidence_row(shape, "free", "primary_or_resolution", 0.50 + 0.02 * index, seed))
    return rows


def test_confidence_figure_plots_each_treatment_only_at_its_assigned_factor(tmp_path, monkeypatch):
    captured = {}
    original = figures.save_and_audit_figure

    def spy(fig, out_dir, stem):
        captured["fig"] = fig
        return original(fig, out_dir, stem)

    monkeypatch.setattr(figures, "save_and_audit_figure", spy)
    figures.make_confidence_figure(_confidence_rows(), tmp_path)
    expected_medians = {"overconfident": 12.0, "calibrated": 22.0, "conservative": 32.0}
    for axis in captured["fig"].axes:
        # No fictitious horizontal treatment bands/curves across the three
        # factors (errorbar caps are LineCollections and are legitimate).
        assert not [
            collection
            for collection in axis.collections
            if isinstance(collection, PolyCollection)
        ]
        for line in axis.lines:
            if line.get_color() == "black":
                continue
            assert not (
                len(line.get_xdata()) == 3
                and list(line.get_xdata()) == [0.2, 1.0, 3.0]
            ), "treatment median repeated at all three factors"
        for method in figures.SENSITIVITY_METHODS:
            marker_lines = [
                line
                for line in axis.lines
                if line.get_marker() == figures.METHOD_MARKERS[method]
                and line.get_color() == figures.METHOD_COLORS[method]
            ]
            assert len(marker_lines) == 1, f"missing or duplicate series: {method}"
            line = marker_lines[0]
            assert list(line.get_xdata()) == [figures.SENSITIVITY_FACTORS[method]]
            assert list(line.get_ydata()) == [pytest.approx(expected_medians[method])]
        # Fixed/free stay genuine factor-independent reference lines.
        reference_lines = [line for line in axis.lines if line.get_color() == "black"]
        assert len(reference_lines) == 2


def test_confidence_figure_reports_absent_calibrated_primary_path(tmp_path):
    rows = [
        row
        for row in _confidence_rows()
        if not (row["method"] == "calibrated" and row["configuration"]["shape"] == "asymmetric")
    ]
    with pytest.raises(figures.SummaryError, match="confidence series is absent: asymmetric/calibrated"):
        figures.make_confidence_figure(rows, tmp_path)


def _resolution_row(method: str, tau: float, sigma_micro: float, sensors: int, value: float) -> dict:
    return {
        "dataset_key": f"asym_tau-{tau}_sigma-{sigma_micro}_sensors-{sensors}_seed-1101",
        "path_key": f"asym_tau-{tau}_sigma-{sigma_micro}_sensors-{sensors}_seed-1101__method-{method}",
        "configuration": {
            "shape": "asymmetric",
            "tau": tau,
            "sigma": sigma_micro * 1.0e-6,
            "sigma_microstrain": sigma_micro,
            "sensors": sensors,
            "seed": 1101,
        },
        "memberships": ["resolution"],
        "method": method,
        "method_family": "primary_or_resolution",
        "selected": True,
        "status": "selected",
        "metrics": {"support_rel_L2_except_intact": value},
    }


def test_measurement_map_cells_and_colorbar_share_percent_units(tmp_path, monkeypatch):
    sensor_levels = (10, 20, 40)
    sigma_levels = (1.0, 5.0, 20.0)
    values = {}
    rows = []
    for tau_index, tau in enumerate((0.01, 0.10)):
        for method_index, method in enumerate(("calibrated", "free")):
            for iy, sigma in enumerate(sigma_levels):
                for ix, sensors in enumerate(sensor_levels):
                    value = 0.01 + 0.03 * tau_index + 0.06 * method_index + 0.01 * iy + 0.002 * ix
                    values[(tau, method, sigma, sensors)] = value
                    rows.append(_resolution_row(method, tau, sigma, sensors, value))

    captured = {}
    original = figures.save_and_audit_figure

    def spy(fig, out_dir, stem):
        captured["fig"] = fig
        return original(fig, out_dir, stem)

    monkeypatch.setattr(figures, "save_and_audit_figure", spy)
    figures.make_measurement_map_figure(rows, tmp_path)
    panels = {
        "(a)": (0.01, "calibrated"),
        "(b)": (0.01, "free"),
        "(c)": (0.10, "calibrated"),
        "(d)": (0.10, "free"),
    }
    map_axes = [axis for axis in captured["fig"].axes if axis.images]
    assert len(map_axes) == 4
    expected_percent = [100.0 * value for value in values.values()]
    global_min, global_max = min(expected_percent), max(expected_percent)
    for axis in map_axes:
        tau, method = panels[axis.get_title()[:3]]
        data = np.ma.filled(axis.images[0].get_array(), np.nan).astype(float)
        texts = {(round(t.get_position()[0]), round(t.get_position()[1])): t.get_text() for t in axis.texts}
        for iy, sigma in enumerate(sigma_levels):
            for ix, sensors in enumerate(sensor_levels):
                expected = 100.0 * values[(tau, method, sigma, sensors)]
                assert data[iy, ix] == pytest.approx(expected)
                # Numeric cell text and the displayed color use identical
                # percent units.
                assert texts[(ix, iy)] == f"{expected:.1f}"
        assert axis.images[0].get_clim() == (pytest.approx(global_min), pytest.approx(global_max))
    color_axes = [axis for axis in captured["fig"].axes if not axis.images]
    assert len(color_axes) == 1
    color_axis = color_axes[0]
    assert "%" in color_axis.get_xlabel()
    ticks = color_axis.get_xticks()
    assert ticks.min() >= global_min - 1e-9
    assert ticks.max() <= global_max + 1e-9


def test_model_figure_inspected_span_covers_displayed_beam(tmp_path, monkeypatch):
    captured = {}
    original = figures.save_and_audit_figure

    def spy(fig, out_dir, stem):
        captured["fig"] = fig
        return original(fig, out_dir, stem)

    monkeypatch.setattr(figures, "save_and_audit_figure", spy)
    figures.make_model_figure(tmp_path)
    mechanical = captured["fig"].axes[0]
    # The inspected-span dimension covers the displayed beam between the
    # omitted-domain ellipses (x = 18 to x = 82), not only central springs.
    span_lines = [line for line in mechanical.lines if list(line.get_ydata()) == [23.5, 23.5]]
    assert len(span_lines) == 1
    assert list(span_lines[0].get_xdata()) == [18.0, 82.0]
    ticks = [line for line in mechanical.lines if list(line.get_ydata()) == [22.4, 24.6]]
    assert sorted(line.get_xdata()[0] for line in ticks) == [18.0, 82.0]
    texts = [text.get_text() for text in mechanical.texts]
    assert any("inspected span" in text for text in texts)
    # The bounded reduced-stiffness region remains a separate annotation.
    assert "bounded region of reduced stiffness (schematic)" in texts


def test_tau_axis_label_is_load_measurement_uncertainty():
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    figures._tau_axis(axis)
    # tau is the actual load-measurement uncertainty of every dataset; only
    # some methods assign it, so the axis must not claim "assigned".
    assert axis.get_xlabel() == r"Load measurement uncertainty $\tau$ (%)"
    plt.close(fig)


def _fields_bundle(shape: str) -> tuple[str, dict]:
    dataset_key = f"shape-{shape}_tau-0.1_sigma-5e-06_sensors-20_seed-1101"
    score_x = np.linspace(-12.0, 12.0, 2400)
    unused_x = np.linspace(-12.0, 12.0, 241)
    reference_d = 0.5 * np.exp(-0.5 * (score_x - 4.0) ** 2)
    reference_strain = 1.0e-6 * np.sin(unused_x)
    return dataset_key, {
        "score_x": score_x,
        "reference_d": reference_d,
        "unused_x": unused_x,
        "reference_strain": reference_strain,
    }


def _fields_row(shape: str, method: str, dataset_key: str) -> dict:
    return {
        "dataset_key": dataset_key,
        "path_key": f"{dataset_key}__method-{method}",
        "configuration": {
            "shape": shape,
            "tau": 0.10,
            "sigma": figures.FIELDS_SIGMA,
            "sensors": figures.FIELDS_SENSORS,
            "seed": figures.FIELDS_SEED,
        },
        "memberships": ["primary"],
        "method": method,
        "method_family": "primary_or_resolution",
        "selected": True,
        "status": "selected",
        "metrics": {"support_rel_L2_except_intact": 0.1},
    }


def test_fields_prediction_reference_once_and_spaced_hollow_markers(tmp_path, monkeypatch):
    rows = []
    field_arrays = {}
    for shape in ("asymmetric", "double"):
        dataset_key, base = _fields_bundle(shape)
        for method_index, method in enumerate(figures.MAIN_METHODS):
            rows.append(_fields_row(shape, method, dataset_key))
            arrays = dict(base)
            arrays["estimated_d"] = base["reference_d"] * (1.0 + 0.05 * method_index)
            arrays["predicted_strain"] = base["reference_strain"] * (1.0 + 0.05 * method_index)
            field_arrays[(dataset_key, method)] = arrays
    bundle = {"field_arrays": field_arrays}

    captured = {}
    original = figures.save_and_audit_figure

    def spy(fig, out_dir, stem):
        captured["fig"] = fig
        return original(fig, out_dir, stem)

    monkeypatch.setattr(figures, "save_and_audit_figure", spy)
    report = figures.make_fields_prediction_figure(bundle, rows, tmp_path)
    # The style contract is preserved on the edited figure.
    assert report["inspection"]["width_inches"] == figures.FIGURE_WIDTH_INCHES
    assert report["inspection"]["text_sizes_pt"] == [figures.FONT_SIZE_PT]

    expected_lengths = {"support": 2400, "response": 241}
    for axis in captured["fig"].axes:
        kind = "support" if "support" in axis.get_title().lower() else "response"
        n_points = expected_lengths[kind]
        # Exactly one black dashed reference line per panel, not one per method.
        reference_lines = [
            line
            for line in axis.lines
            if line.get_color() == "black" and line.get_linestyle() == "--"
        ]
        assert len(reference_lines) == 1
        assert len(reference_lines[0].get_xdata()) == n_points
        # Four method curves: full-resolution lines, hollow markers thinned to
        # about ten per curve with a distinct offset per method.
        starts = []
        for method_index, method in enumerate(figures.MAIN_METHODS):
            method_lines = [
                line
                for line in axis.lines
                if line.get_color() == figures.METHOD_COLORS[method]
                and line.get_linestyle() == figures.METHOD_LINESTYLES[method]
            ]
            assert len(method_lines) == 1, f"missing or duplicate series: {method}"
            line = method_lines[0]
            assert len(line.get_xdata()) == n_points, "line data must not be downsampled"
            assert line.get_markerfacecolor() == "none"
            markevery = line.get_markevery()
            assert isinstance(markevery, tuple) and len(markevery) == 2
            start, step = markevery
            assert step == n_points // 10
            assert start == method_index * (step // len(figures.MAIN_METHODS))
            starts.append(start)
            n_markers = len(range(start, n_points, step))
            assert 8 <= n_markers <= 12
        assert len(set(starts)) == len(figures.MAIN_METHODS)
        assert len(axis.lines) == 1 + len(figures.MAIN_METHODS)
