"""Postprocessing module for creating figures and summaries.

Creates 6 publication-quality figures with fixed typography (6.5in width,
10pt Computer Modern text) plus machine-readable summary files.

Requires complete formal experiment results. Does not generate placeholder
data or fabricate missing inputs.
"""

import sys
import json
import csv
import argparse
import builtins
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Typography configuration
FIGURE_WIDTH_INCHES = 6.5
FONT_SIZE_PT = 10
DPI = 300

# Computer Modern fonts
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['cmr10']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.formatter.use_mathtext'] = True
matplotlib.rcParams['font.size'] = FONT_SIZE_PT
matplotlib.rcParams['axes.labelsize'] = FONT_SIZE_PT
matplotlib.rcParams['xtick.labelsize'] = FONT_SIZE_PT
matplotlib.rcParams['ytick.labelsize'] = FONT_SIZE_PT
matplotlib.rcParams['legend.fontsize'] = FONT_SIZE_PT
matplotlib.rcParams['pdf.fonttype'] = 42

# Actual schema constants from protocol_frozen_v1.json
ACTUAL_METHODS = ['H1_fixed', 'H1_profile', 'TV_fixed', 'TV_profile', 'H1_oracle']
ACTUAL_SHAPES = ['intact', 'rectangle', 'asymmetric', 'double', 'broad_shallow']
PRIMARY_SEEDS = [2026110101, 2026110102, 2026110103, 2026110104, 2026110105]
ROBUSTNESS_SEEDS = [2026110111, 2026110112, 2026110113]
FIRST_SEED = 2026110101
ACTUAL_REGIMES = ['calibrated', 'amplitude_bias']
ROBUSTNESS_REGIMES = [
    'amplitude_bias_sigma20microstrain',
    'amplitude_bias_pasternak_kG0.2k0ell2',
    'amplitude_bias_load_position_plus0.25m'
]


def verify_font_availability():
    """Check that Computer Modern fonts are available."""
    fonts = [f.name for f in font_manager.fontManager.ttflist
             if 'cmr10' in f.name.lower()]
    if not fonts:
        warnings.warn("cmr10 font not found; may fall back to default serif")


def _compute_canonical_protocol_hash(protocol_path: Path) -> str:
    """Compute canonical protocol hash from sorted compact JSON.

    Args:
        protocol_path: Path to protocol_frozen_v1.json

    Returns:
        SHA256 hex digest of canonical sorted compact JSON
    """
    import hashlib

    with builtins.open(protocol_path) as f:
        protocol = json.load(f)

    # Canonical form: sorted keys, compact (no whitespace)
    canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _verify_file_hash(file_path: Path, expected_hash: str, description: str):
    """Verify SHA256 hash of a file.

    Args:
        file_path: Path to file
        expected_hash: Expected SHA256 hex digest
        description: Description for error messages

    Raises:
        ValueError: If hash does not match
    """
    import hashlib

    hasher = hashlib.sha256()
    with builtins.open(file_path, 'rb') as f:
        hasher.update(f.read())
    actual_hash = hasher.hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(f"{description} hash mismatch: expected {expected_hash}, got {actual_hash}")


def load_and_validate_results(results_dir: Path) -> Dict[str, Any]:
    """Load and validate formal experiment results.

    Requires complete status in all manifests, validates protocol hash against
    canonical sorted compact JSON, validates all source hashes against current
    code files, validates effective_plan hash, validates dataset NPZ hashes,
    validates selected array hashes, and enforces exact expected unique dataset/method keys.

    Args:
        results_dir: Root results directory

    Raises:
        FileNotFoundError: If manifest, protocol, source files, or data files are missing
        ValueError: If validation fails (incomplete manifest, hash mismatch, wrong counts, unexpected keys)
    """
    import hashlib

    results = {
        'primary': {},
        'robustness': {},
        'joint': {},
        'manifests': {},
        'results_root': results_dir
    }

    # Locate protocol file relative to this module
    code_dir = Path(__file__).parent
    protocol_path = code_dir.parent / 'protocol_frozen_v1.json'

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    # Compute canonical protocol hash once
    canonical_protocol_hash = _compute_canonical_protocol_hash(protocol_path)

    # Load and validate primary manifest
    primary_manifest = results_dir / 'primary' / 'run_manifest.json'
    if not primary_manifest.exists():
        raise FileNotFoundError(f"Primary manifest not found: {primary_manifest}")

    with builtins.open(primary_manifest) as f:
        manifest = json.load(f)

    # Validate manifest completeness
    status = manifest.get('status', '')
    if status != 'complete':
        raise ValueError(f"Primary manifest status must be 'complete', got '{status}'")

    # Validate protocol_hash
    if 'protocol_hash' not in manifest:
        raise ValueError("Primary manifest missing protocol_hash")
    if manifest['protocol_hash'] != canonical_protocol_hash:
        raise ValueError(f"Primary manifest protocol_hash mismatch: "
                        f"expected {canonical_protocol_hash}, got {manifest['protocol_hash']}")

    # Validate source_hashes - require exact 8 files
    if 'source_hashes' not in manifest:
        raise ValueError("Primary manifest missing source_hashes")

    expected_sources = {
        'beam_inverse.py', 'campaign.py', 'shape_factory.py', 'skfem_reference.py',
        'run_campaign.py', 'verification.py', 'green_verification.py', 'run_formal.py'
    }
    actual_sources = set(manifest['source_hashes'].keys())

    if actual_sources != expected_sources:
        missing = expected_sources - actual_sources
        extra = actual_sources - expected_sources
        msg = "Primary manifest source_hashes key mismatch"
        if missing:
            msg += f"; missing: {sorted(missing)}"
        if extra:
            msg += f"; unexpected: {sorted(extra)}"
        raise ValueError(msg)

    # Verify each source file hash against current code
    for source_name, expected_hash in manifest['source_hashes'].items():
        source_path = code_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        _verify_file_hash(source_path, expected_hash, f"Primary source {source_name}")

    # Validate effective_plan hash
    if 'effective_plan' not in manifest:
        raise ValueError("Primary manifest missing effective_plan")

    effective_plan_canonical = json.dumps(manifest['effective_plan'], sort_keys=True, separators=(',', ':'))
    effective_plan_hash = hashlib.sha256(effective_plan_canonical.encode('utf-8')).hexdigest()

    # The effective_plan should match the protocol content
    with builtins.open(protocol_path) as f:
        protocol_content = json.load(f)
    protocol_canonical = json.dumps(protocol_content, sort_keys=True, separators=(',', ':'))
    protocol_content_hash = hashlib.sha256(protocol_canonical.encode('utf-8')).hexdigest()

    if effective_plan_hash != protocol_content_hash:
        raise ValueError(f"Primary manifest effective_plan hash mismatch with protocol content")

    results['manifests']['primary'] = manifest

    # Load and validate primary results
    primary_results_dir = results_dir / 'primary' / 'results'
    primary_files = [f for f in primary_results_dir.glob('*.json')
                     if not f.name.startswith('._')]

    expected_primary = 225  # 5 shapes × 2 regimes × 5 seeds × 4 methods + 5 shapes × 5 seeds × oracle
    if len(primary_files) != expected_primary:
        raise ValueError(f"Expected {expected_primary} primary results, found {len(primary_files)}")

    # Build expected unique keys for primary
    expected_primary_keys = set()
    for shape in ACTUAL_SHAPES:
        for regime in ACTUAL_REGIMES:
            for seed in PRIMARY_SEEDS:
                for method in ['H1_fixed', 'H1_profile', 'TV_fixed', 'TV_profile']:
                    expected_primary_keys.add((shape, regime, seed, method))
        # Oracle only on amplitude_bias
        for seed in PRIMARY_SEEDS:
            expected_primary_keys.add((shape, 'amplitude_bias', seed, 'H1_oracle'))

    actual_primary_keys = set()

    for result_file in primary_files:
        with builtins.open(result_file) as f:
            result = json.load(f)

        ds = result['dataset']

        # Validate dataset has required integrity fields
        if 'npz_relative_path' not in ds:
            raise ValueError(f"Primary result {result_file.name} missing dataset.npz_relative_path")
        if 'sha256' not in ds:
            raise ValueError(f"Primary result {result_file.name} missing dataset.sha256")

        # Verify dataset NPZ file hash
        dataset_path = results_dir / 'primary' / ds['npz_relative_path']
        if not dataset_path.exists():
            raise FileNotFoundError(f"Primary dataset not found: {dataset_path}")
        _verify_file_hash(dataset_path, ds['sha256'], f"Primary dataset {ds['npz_relative_path']}")

        key = (ds['shape'], ds['regime'], ds['seed'], result['method'])
        actual_primary_keys.add(key)

        if key in results['primary']:
            raise ValueError(f"Duplicate primary result key: {key}")

        # Validate selected array integrity
        array_filename = result.get('selected_array_filename')
        if not array_filename:
            raise ValueError(f"Primary result {result_file.name} missing selected_array_filename")

        array_hash = result.get('array_sha256')
        if not array_hash:
            raise ValueError(f"Primary result {result_file.name} missing array_sha256")

        array_file = primary_results_dir / array_filename
        if not array_file.exists():
            raise FileNotFoundError(f"Primary array file missing: {array_file}")

        _verify_file_hash(array_file, array_hash, f"Primary array {array_filename}")

        results['primary'][key] = result

    # Verify exact key match
    if actual_primary_keys != expected_primary_keys:
        missing = expected_primary_keys - actual_primary_keys
        extra = actual_primary_keys - expected_primary_keys
        msg = "Primary dataset/method key mismatch"
        if missing:
            msg += f"; missing {len(missing)} keys (first 3): {sorted(list(missing))[:3]}"
        if extra:
            msg += f"; unexpected {len(extra)} keys (first 3): {sorted(list(extra))[:3]}"
        raise ValueError(msg)

    # Load and validate robustness manifest
    robust_manifest = results_dir / 'robustness' / 'run_manifest.json'
    if not robust_manifest.exists():
        raise FileNotFoundError(f"Robustness manifest not found: {robust_manifest}")

    with builtins.open(robust_manifest) as f:
        manifest = json.load(f)

    # Validate robustness manifest
    status = manifest.get('status', '')
    if status != 'complete':
        raise ValueError(f"Robustness manifest status must be 'complete', got '{status}'")

    if 'protocol_hash' not in manifest:
        raise ValueError("Robustness manifest missing protocol_hash")
    if manifest['protocol_hash'] != canonical_protocol_hash:
        raise ValueError(f"Robustness manifest protocol_hash mismatch")

    if 'source_hashes' not in manifest:
        raise ValueError("Robustness manifest missing source_hashes")
    if set(manifest['source_hashes'].keys()) != expected_sources:
        raise ValueError("Robustness manifest source_hashes key mismatch")

    # Verify robustness source hashes
    for source_name, expected_hash in manifest['source_hashes'].items():
        source_path = code_dir / source_name
        _verify_file_hash(source_path, expected_hash, f"Robustness source {source_name}")

    # Validate effective_plan
    if 'effective_plan' not in manifest:
        raise ValueError("Robustness manifest missing effective_plan")
    robust_plan_canonical = json.dumps(manifest['effective_plan'], sort_keys=True, separators=(',', ':'))
    robust_plan_hash = hashlib.sha256(robust_plan_canonical.encode('utf-8')).hexdigest()
    if robust_plan_hash != protocol_content_hash:
        raise ValueError("Robustness manifest effective_plan hash mismatch with protocol content")

    results['manifests']['robustness'] = manifest

    # Load and validate robustness results
    robust_results_dir = results_dir / 'robustness' / 'results'
    robust_files = [f for f in robust_results_dir.glob('*.json')
                   if not f.name.startswith('._')]

    expected_robust = 54  # 3 shapes × 3 regimes × 3 seeds × 2 methods
    if len(robust_files) != expected_robust:
        raise ValueError(f"Expected {expected_robust} robustness results, found {len(robust_files)}")

    # Build expected unique keys for robustness
    expected_robust_keys = set()
    robust_shapes = ['intact', 'asymmetric', 'double']
    for shape in robust_shapes:
        for regime in ROBUSTNESS_REGIMES:
            for seed in ROBUSTNESS_SEEDS:
                for method in ['H1_fixed', 'H1_profile']:
                    expected_robust_keys.add((shape, regime, seed, method))

    actual_robust_keys = set()

    for result_file in robust_files:
        with builtins.open(result_file) as f:
            result = json.load(f)

        ds = result['dataset']

        # Validate dataset integrity
        if 'npz_relative_path' not in ds:
            raise ValueError(f"Robustness result {result_file.name} missing dataset.npz_relative_path")
        if 'sha256' not in ds:
            raise ValueError(f"Robustness result {result_file.name} missing dataset.sha256")

        dataset_path = results_dir / 'robustness' / ds['npz_relative_path']
        if not dataset_path.exists():
            raise FileNotFoundError(f"Robustness dataset not found: {dataset_path}")
        _verify_file_hash(dataset_path, ds['sha256'], f"Robustness dataset {ds['npz_relative_path']}")

        key = (ds['shape'], ds['regime'], ds['seed'], result['method'])
        actual_robust_keys.add(key)

        if key in results['robustness']:
            raise ValueError(f"Duplicate robustness key: {key}")

        # Validate selected array integrity
        array_filename = result.get('selected_array_filename')
        if not array_filename:
            raise ValueError(f"Robustness result {result_file.name} missing selected_array_filename")

        array_hash = result.get('array_sha256')
        if not array_hash:
            raise ValueError(f"Robustness result {result_file.name} missing array_sha256")

        array_file = robust_results_dir / array_filename
        if not array_file.exists():
            raise FileNotFoundError(f"Robustness array file missing: {array_file}")

        _verify_file_hash(array_file, array_hash, f"Robustness array {array_filename}")

        results['robustness'][key] = result

    # Verify exact key match for robustness
    if actual_robust_keys != expected_robust_keys:
        missing = expected_robust_keys - actual_robust_keys
        extra = actual_robust_keys - expected_robust_keys
        msg = "Robustness dataset/method key mismatch"
        if missing:
            msg += f"; missing {len(missing)} keys"
        if extra:
            msg += f"; unexpected {len(extra)} keys"
        raise ValueError(msg)

    # Load and validate joint manifest
    joint_manifest = results_dir / 'joint' / 'run_manifest.json'
    if not joint_manifest.exists():
        raise FileNotFoundError(f"Joint manifest not found: {joint_manifest}")

    with builtins.open(joint_manifest) as f:
        manifest = json.load(f)

    # Validate joint manifest
    status = manifest.get('status', '')
    if status != 'complete':
        raise ValueError(f"Joint manifest status must be 'complete', got '{status}'")

    if 'protocol_hash' not in manifest:
        raise ValueError("Joint manifest missing protocol_hash")
    if manifest['protocol_hash'] != canonical_protocol_hash:
        raise ValueError(f"Joint manifest protocol_hash mismatch")

    if 'source_hashes' not in manifest:
        raise ValueError("Joint manifest missing source_hashes")
    if set(manifest['source_hashes'].keys()) != expected_sources:
        raise ValueError("Joint manifest source_hashes key mismatch")

    # Verify joint source hashes
    for source_name, expected_hash in manifest['source_hashes'].items():
        source_path = code_dir / source_name
        _verify_file_hash(source_path, expected_hash, f"Joint source {source_name}")

    # Validate effective_plan
    if 'effective_plan' not in manifest:
        raise ValueError("Joint manifest missing effective_plan")
    joint_plan_canonical = json.dumps(manifest['effective_plan'], sort_keys=True, separators=(',', ':'))
    joint_plan_hash = hashlib.sha256(joint_plan_canonical.encode('utf-8')).hexdigest()
    if joint_plan_hash != protocol_content_hash:
        raise ValueError("Joint manifest effective_plan hash mismatch with protocol content")

    results['manifests']['joint'] = manifest

    # Load and validate joint results
    joint_results_dir = results_dir / 'joint' / 'results'
    joint_files = [f for f in joint_results_dir.glob('*_joint_comparison.json')
                  if not f.name.startswith('._')]

    expected_joint = 5  # 5 shapes at first seed
    if len(joint_files) != expected_joint:
        raise ValueError(f"Expected {expected_joint} joint comparison files, found {len(joint_files)}")

    # Build expected keys for joint (all 5 shapes at first seed)
    expected_joint_keys = set()
    for shape in ACTUAL_SHAPES:
        expected_joint_keys.add((shape, FIRST_SEED))

    actual_joint_keys = set()

    # Joint resolves dataset relative to primary_dir in manifest
    # The manifest stores a relative path like '../primary'
    primary_dir_rel = manifest.get('primary_dir', '../primary')
    primary_dir_path = (results_dir / 'joint' / primary_dir_rel).resolve()

    for result_file in joint_files:
        with builtins.open(result_file) as f:
            result = json.load(f)

        ds = result['dataset']

        # Validate dataset integrity
        if 'npz_relative_path' not in ds:
            raise ValueError(f"Joint result {result_file.name} missing dataset.npz_relative_path")
        if 'sha256' not in ds:
            raise ValueError(f"Joint result {result_file.name} missing dataset.sha256")

        # Joint datasets resolve relative to primary_dir
        dataset_path = primary_dir_path / ds['npz_relative_path']
        if not dataset_path.exists():
            raise FileNotFoundError(f"Joint dataset not found: {dataset_path}")
        _verify_file_hash(dataset_path, ds['sha256'], f"Joint dataset {ds['npz_relative_path']}")

        shape = ds['shape']
        seed = ds['seed']
        key = (shape, seed)
        actual_joint_keys.add(key)

        if key in results['joint']:
            raise ValueError(f"Duplicate joint key: {key}")

        # Validate fits structure: exactly 3 repeats (0, 1, 2) × 2 methods (H1_profile, H1_joint)
        fits = result.get('fits', [])
        if len(fits) != 6:
            raise ValueError(f"Joint result {result_file.name} expected 6 fits, found {len(fits)}")

        # Check repeat_idx and method pairs
        expected_fit_pairs = set()
        for repeat_idx in [0, 1, 2]:
            expected_fit_pairs.add((repeat_idx, 'H1_profile'))
            expected_fit_pairs.add((repeat_idx, 'H1_joint'))

        actual_fit_pairs = set()
        for fit in fits:
            if 'repeat_idx' not in fit:
                raise ValueError(f"Joint fit in {result_file.name} missing repeat_idx")
            if 'method' not in fit:
                raise ValueError(f"Joint fit in {result_file.name} missing method")
            actual_fit_pairs.add((fit['repeat_idx'], fit['method']))

        if actual_fit_pairs != expected_fit_pairs:
            raise ValueError(f"Joint result {result_file.name} has incorrect repeat_idx/method pairs")

        results['joint'][key] = result

    # Verify exact key match for joint
    if actual_joint_keys != expected_joint_keys:
        missing = expected_joint_keys - actual_joint_keys
        extra = actual_joint_keys - expected_joint_keys
        msg = "Joint dataset key mismatch"
        if missing:
            msg += f"; missing {len(missing)} shapes"
        if extra:
            msg += f"; unexpected {len(extra)} shapes"
        raise ValueError(msg)

    print(f"Loaded: {len(results['primary'])} primary, "
          f"{len(results['robustness'])} robustness, "
          f"{len(results['joint'])} joint datasets")
    print(f"Validated: protocol hash, {len(expected_sources)} source hashes, "
          f"effective_plan hash, all dataset and array integrity")

    return results


def load_verification(verification_path: Path) -> Dict[str, Any]:
    """Load verification results."""
    if not verification_path.exists():
        raise FileNotFoundError(f"Verification file not found: {verification_path}")

    with open(verification_path) as f:
        verification = json.load(f)

    expected_checks = [
        'load_conservation', 'symmetry', 'finite_difference_jacobian',
        'mesh_convergence', 'scikit_fem_intact', 'scikit_fem_graded',
        'green_intact'
    ]

    for check in expected_checks:
        if check not in verification:
            raise ValueError(f"Missing verification check: {check}")

    return verification


def create_verification_figure(verification: Dict, output_path: Path):
    """Create verification.pdf showing Green intact response and FE refinement."""
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 6.3))

    # Two full-width stacked panels - reduce width slightly to prevent right edge clipping
    ax1 = fig.add_axes([0.13, 0.57, 0.80, 0.27])  # Green + FE response
    ax2 = fig.add_axes([0.13, 0.12, 0.80, 0.27])  # Mesh convergence

    # Panel 1: Green intact reference and actual FE response from scikit_fem_intact
    green = verification['green_intact']
    x_eval = np.array(green['arrays']['evaluation'])
    green_strain = np.array(green['arrays']['green_strain'])

    scifem = verification['scikit_fem_intact']
    custom_strain = np.array(scifem['arrays']['custom_strain'])

    ax1.plot(x_eval, green_strain * 1e6, 'g-', linewidth=1.5, label='Green reference')
    ax1.plot(x_eval, custom_strain * 1e6, 'b--', linewidth=1.5, label='FE response')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel(r'Strain ($\mu\varepsilon$)')
    ax1.legend(bbox_to_anchor=(0.5, 1.14), loc='lower center', ncol=2, frameon=False)
    ax1.grid(True, alpha=0.3)
    fig.text(0.13, 0.94, '(a) Intact response', ha='left', va='top', fontsize=FONT_SIZE_PT)

    # Panel 2: Mesh refinement - Green h_sequence (5 points) and graded errors (4 points)
    mesh_conv = verification['mesh_convergence']
    h_seq_green = np.array(green['h_sequence'])
    fe_errors_green = np.array(green['fe_vs_green_errors'])

    h_seq_mesh = np.array(mesh_conv['h_sequence'])
    rel_errors_mesh = np.array(mesh_conv['relative_errors'])

    # 5 Green-relative FE errors
    ax2.loglog(h_seq_green, fe_errors_green, 'go-', markersize=5, linewidth=1.5,
               label='Green-relative error')

    # 4 graded mesh convergence errors (finest mesh used as reference)
    ax2.loglog(h_seq_mesh[:-1], rel_errors_mesh, 'bs-', markersize=5, linewidth=1.5,
               label='Graded-reference error')

    ax2.set_xlabel('Mesh size $h$ (m)')
    ax2.set_ylabel('Relative error')

    # Set exactly 5 x-axis ticks with simple decimal format
    x_ticks = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{x:.5f}'.rstrip('0').rstrip('.') for x in x_ticks])

    # Disable minor x-axis labels to prevent collisions
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())

    ax2.legend(bbox_to_anchor=(0.5, 1.14), loc='lower center', ncol=2, frameon=False)
    ax2.grid(True, alpha=0.3, which='both')
    fig.text(0.13, 0.51, '(b) Mesh refinement', ha='left', va='top', fontsize=FONT_SIZE_PT)

    pdf_path = output_path / 'verification.pdf'
    png_path = output_path / 'verification.png'
    fig.savefig(pdf_path, dpi=DPI, bbox_inches=None)
    fig.savefig(png_path, dpi=DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Created: {pdf_path.name}")
    return pdf_path


def create_intact_control_figure(results: Dict, output_path: Path):
    """Create intact_control.pdf showing support field and spurious loss areas."""
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 7.1))

    # Reduce width slightly to prevent right edge clipping
    ax_field = fig.add_axes([0.12, 0.59, 0.81, 0.27])
    ax_loss = fig.add_axes([0.12, 0.14, 0.32, 0.27])
    ax_gamma = fig.add_axes([0.62, 0.14, 0.31, 0.27])

    # Top panel: only Reference, H1-F, H1-P, Actual force (4 methods)
    field_methods = ['H1_fixed', 'H1_profile', 'H1_oracle']
    field_labels = ['H1-F', 'H1-P', 'Actual\nforce']
    field_colors = ['C0', 'C1', 'C4']

    # Loss panel: all 5 methods
    loss_methods = ['H1_fixed', 'H1_profile', 'TV_fixed', 'TV_profile', 'H1_oracle']
    loss_labels = ['H1-F', 'H1-P', 'TV-F', 'TV-P', 'Actual\nforce']
    loss_colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    results_root = results['results_root']

    # Plot field for first seed - reference first, then methods
    key_ref = ('intact', 'amplitude_bias', FIRST_SEED, 'H1_fixed')
    if key_ref in results['primary']:
        result = results['primary'][key_ref]
        arrays_path = results_root / 'primary' / 'results' / result['selected_array_filename']
        if arrays_path.exists():
            arrays = np.load(arrays_path)
            ax_field.plot(arrays['score_x'], arrays['reference_d'],
                         'k-', linewidth=1.5, label='Reference', zorder=10)

    for idx, (method, label, color) in enumerate(zip(field_methods, field_labels, field_colors)):
        key = ('intact', 'amplitude_bias', FIRST_SEED, method)
        if key not in results['primary']:
            continue

        result = results['primary'][key]
        arrays_path = results_root / 'primary' / 'results' / result['selected_array_filename']

        if not arrays_path.exists():
            continue

        arrays = np.load(arrays_path)
        score_x = arrays['score_x']
        estimated_d = arrays['estimated_d']

        linestyle = ['-', '--', '-.'][idx]
        ax_field.plot(score_x, estimated_d, linestyle=linestyle,
                     color=color, linewidth=1.5, label=label)

    ax_field.set_xlabel('Position $x$ (m)')
    ax_field.set_ylabel('Support field $d$')
    ax_field.legend(bbox_to_anchor=(0.5, 1.16), loc='lower center', ncol=4, frameon=False)
    ax_field.grid(True, alpha=0.3)
    fig.text(0.12, 0.90, '(a) Support field recovery', ha='left', va='top', fontsize=FONT_SIZE_PT)

    # Loss area across 5 seeds for all 5 methods
    loss_data = {m: [] for m in loss_methods}
    loss_incompatible = {m: [] for m in loss_methods}  # Track incompatible methods from actual flags

    for seed in PRIMARY_SEEDS:
        for method in loss_methods:
            key = ('intact', 'amplitude_bias', seed, method)
            if key in results['primary']:
                result = results['primary'][key]
                loss_area = result['metrics'].get('loss_area_m', 0.0)
                loss_data[method].append(loss_area)

                # Mark incompatible EXACTLY from no_feasible_candidate flag
                if result['selection'].get('no_feasible_candidate', False):
                    loss_incompatible[method].append(loss_area)

    x_pos = np.arange(len(loss_methods))
    for i, (method, label, color) in enumerate(zip(loss_methods, loss_labels, loss_colors)):
        vals = loss_data[method]
        if vals:
            median_val = np.median(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)

            ax_loss.plot([i, i], [min_val, max_val], 'k-', linewidth=1)
            ax_loss.plot(i, median_val, 'o', color=color, markersize=6)

            for val in vals:
                ax_loss.plot(i, val, '.', color=color, markersize=3, alpha=0.5)

            # Mark incompatible points with small black x
            for val in loss_incompatible.get(method, []):
                ax_loss.plot(i, val, 'x', color='k', markersize=6,
                           markeredgewidth=1.5, zorder=10)

    ax_loss.set_xticks(x_pos)
    ax_loss.set_xticklabels(loss_labels, rotation=0)
    ax_loss.set_ylabel('Spurious loss area (m)')
    ax_loss.grid(True, alpha=0.3, axis='y')
    fig.text(0.12, 0.44, '(b) Spurious loss detection', ha='left', va='top', fontsize=FONT_SIZE_PT)

    # Gamma across loads and seeds - only H1 methods have gamma
    gamma_methods = ['H1_fixed', 'H1_profile', 'H1_oracle']
    gamma_labels = ['H1-F', 'H1-P', 'Actual\nforce']
    gamma_colors = [field_colors[0], field_colors[1], field_colors[2]]
    gamma_data = {m: {0: [], 1: [], 2: []} for m in gamma_methods}
    true_gamma = [1.12, 0.9, 1.08]  # amplitude_bias regime

    for seed in PRIMARY_SEEDS:
        for method in gamma_methods:
            key = ('intact', 'amplitude_bias', seed, method)
            if key in results['primary']:
                result = results['primary'][key]
                # Find selected candidate
                selected_lambda = result.get('selection', {}).get('selected_lambda')
                if selected_lambda is not None:
                    selected = [c for c in result['candidates']
                               if c['penalty_lambda'] == selected_lambda]
                    if selected:
                        gamma = selected[0]['gamma']
                        for load_idx in range(3):
                            gamma_data[method][load_idx].append(gamma[load_idx])

    load_positions = [-4, 0, 4]
    x_offset = 0.15

    for method_idx, (method, color) in enumerate(zip(gamma_methods, gamma_colors)):
        for load_idx in range(3):
            vals = gamma_data[method][load_idx]
            if vals:
                x_loc = load_idx + (method_idx - 1) * x_offset
                median_val = np.median(vals)
                min_val = np.min(vals)
                max_val = np.max(vals)

                ax_gamma.plot([x_loc, x_loc], [min_val, max_val],
                             color=color, linewidth=1)
                ax_gamma.plot(x_loc, median_val, 'o',
                             color=color, markersize=4)

    # Add true values as short dashed segments in data coordinates
    for load_idx, true_val in enumerate(true_gamma):
        ax_gamma.plot([load_idx - 0.25, load_idx + 0.25], [true_val, true_val],
                     color='gray', linestyle='--', linewidth=1)

    ax_gamma.set_xticks([0, 1, 2])
    ax_gamma.set_xticklabels([f'{p} m' for p in load_positions])
    ax_gamma.set_xlabel('Load position')
    ax_gamma.set_ylabel(r'Estimated $\gamma$')
    ax_gamma.grid(True, alpha=0.3, axis='y')

    # Add explicit legend for applied amplitude in reserved gap area
    from matplotlib.lines import Line2D
    dashed_handle = Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='Applied amplitude')
    ax_gamma.legend(handles=[dashed_handle], loc='center', bbox_to_anchor=(0.78, 0.49), bbox_transform=fig.transFigure, frameon=False)

    fig.text(0.62, 0.44, '(c) Force multiplier recovery', ha='left', va='top', fontsize=FONT_SIZE_PT)

    pdf_path = output_path / 'intact_control.pdf'
    png_path = output_path / 'intact_control.png'
    fig.savefig(pdf_path, dpi=DPI, bbox_inches=None)
    fig.savefig(png_path, dpi=DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Created: {pdf_path.name}")
    return pdf_path


def create_field_errors_figure(results: Dict, output_path: Path):
    """Create field_errors.pdf comparing calibrated/amplitude-biased regimes."""
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 6.8))

    # Two full-width rows for calibrated and amplitude_bias - reduce width slightly
    ax_cal = fig.add_axes([0.12, 0.57, 0.81, 0.27])
    ax_bias = fig.add_axes([0.12, 0.12, 0.81, 0.27])

    # Use actual defective shapes (exclude intact)
    shapes = ['rectangle', 'asymmetric', 'double', 'broad_shallow']
    shape_labels = ['Rectangle', 'Asymmetric', 'Double', 'Broad shallow']
    methods = ['H1_fixed', 'H1_profile', 'TV_fixed', 'TV_profile', 'H1_oracle']
    method_labels = ['H1-F', 'H1-P', 'TV-F', 'TV-P', 'Actual force']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    markers = ['o', 's', '^', 'd', 'D']  # Diamond for Actual force

    for regime, ax, panel_label, panel_title in [
        ('calibrated', ax_cal, '(a)', 'Calibrated forces'),
        ('amplitude_bias', ax_bias, '(b)', 'Biased nominal forces')
    ]:

        for shape_idx, shape in enumerate(shapes):
            for method_idx, method in enumerate(methods):

                errors = []
                infeasible_points = []

                for seed in PRIMARY_SEEDS:
                    key = (shape, regime, seed, method)
                    if key in results['primary']:
                        result = results['primary'][key]
                        # Check if infeasible
                        is_infeasible = result.get('selection', {}).get('no_feasible_candidate', False)

                        error = result['metrics'].get('support_rel_L2_except_intact')
                        if error is not None and not np.isnan(error) and not np.isinf(error):
                            errors.append(error)
                            if is_infeasible:
                                infeasible_points.append(error)

                if errors:
                    x_pos = shape_idx + (method_idx - 2) * 0.15
                    median_val = np.median(errors)
                    min_val = np.min(errors)
                    max_val = np.max(errors)

                    ax.plot([x_pos, x_pos], [min_val, max_val],
                           color=colors[method_idx], linewidth=1, alpha=0.6)
                    ax.plot(x_pos, median_val, markers[method_idx],
                           color=colors[method_idx], markersize=5,
                           label=method_labels[method_idx] if shape_idx == 0 else '')

                    for err in errors:
                        ax.plot(x_pos, err, '.', color=colors[method_idx],
                               markersize=2, alpha=0.4)

                    # Mark ALL infeasible points with small black x overlay
                    if infeasible_points:
                        for err in infeasible_points:
                            ax.plot(x_pos, err, 'x', color='k', markersize=6,
                                   markeredgewidth=1.5, zorder=10)

        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels(shape_labels)
        ax.set_ylabel(r'Relative field $E_d$')
        ax.grid(True, alpha=0.3, axis='y')
        fig.text(0.12, 0.90 if regime == 'calibrated' else 0.45,
                f'{panel_label} {panel_title}', ha='left', va='top', fontsize=FONT_SIZE_PT)

        if regime == 'calibrated':
            # Create explicit legend with 6 handles in 2 rows
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], marker='o', color=colors[0], linestyle='', markersize=5, label='H1-F'),
                Line2D([0], [0], marker='s', color=colors[1], linestyle='', markersize=5, label='H1-P'),
                Line2D([0], [0], marker='^', color=colors[2], linestyle='', markersize=5, label='TV-F'),
                Line2D([0], [0], marker='d', color=colors[3], linestyle='', markersize=5, label='TV-P'),
                Line2D([0], [0], marker='D', color=colors[4], linestyle='', markersize=5, label='Actual force'),
                Line2D([0], [0], marker='x', color='k', linestyle='', markersize=6, markeredgewidth=1.5, label='Incompatible')
            ]
            ax.legend(handles=handles, bbox_to_anchor=(0.5, 1.16), loc='lower center', ncol=3, frameon=False)

    pdf_path = output_path / 'field_errors.pdf'
    png_path = output_path / 'field_errors.png'
    fig.savefig(pdf_path, dpi=DPI, bbox_inches=None)
    fig.savefig(png_path, dpi=DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Created: {pdf_path.name}")
    return pdf_path


def create_fields_prediction_figure(results: Dict, output_path: Path):
    """Create fields_prediction.pdf showing recovered support and unused-load strain."""
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 7.1))

    # Reduce width more to prevent right edge clipping
    ax_asym_support = fig.add_axes([0.12, 0.58, 0.31, 0.27])
    ax_asym_unused = fig.add_axes([0.62, 0.58, 0.30, 0.27])
    ax_double_support = fig.add_axes([0.12, 0.12, 0.31, 0.27])
    ax_double_unused = fig.add_axes([0.62, 0.12, 0.30, 0.27])

    # Use H1_fixed, H1_profile, H1_oracle (actual force)
    methods = ['H1_fixed', 'H1_profile', 'H1_oracle']
    method_labels = ['H1-F', 'H1-P', 'Actual force']
    colors = ['C0', 'C1', 'C4']
    linestyles = ['-', '--', '-.']
    markers = ['o', 's', 'D']

    results_root = results['results_root']

    for shape, ax_sup, ax_uns, panel_a, panel_b in [
        ('asymmetric', ax_asym_support, ax_asym_unused, '(a)', '(b)'),
        ('double', ax_double_support, ax_double_unused, '(c)', '(d)')
    ]:

        # Plot reference first
        key = (shape, 'amplitude_bias', FIRST_SEED, methods[0])
        if key in results['primary']:
            result = results['primary'][key]
            arrays_path = results_root / 'primary' / 'results' / result['selected_array_filename']
            if arrays_path.exists():
                arrays = np.load(arrays_path)
                ax_sup.plot(arrays['score_x'], arrays['reference_d'],
                           'k-', linewidth=1.5, label='Reference', zorder=10)

                if 'unused_x' in arrays:
                    unused_x = arrays['unused_x']
                    unused_true = arrays['reference_strain'] * 1e6
                    ax_uns.plot(unused_x, unused_true, 'k-', linewidth=1.5,
                               label='Reference', zorder=10)

        # Plot estimated fields
        for method_idx, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            key = (shape, 'amplitude_bias', FIRST_SEED, method)
            if key not in results['primary']:
                continue

            result = results['primary'][key]
            arrays_path = results_root / 'primary' / 'results' / result['selected_array_filename']

            if not arrays_path.exists():
                continue

            arrays = np.load(arrays_path)

            score_x = arrays['score_x']
            estimated_d = arrays['estimated_d']

            # Check if H1_fixed is incompatible
            is_infeasible = result.get('selection', {}).get('no_feasible_candidate', False)

            ax_sup.plot(score_x, estimated_d, linestyle=linestyles[method_idx],
                       color=color, linewidth=1.5,
                       marker=markers[method_idx], markersize=2, markevery=150,
                       label=label)

            # Mark incompatible H1-F points with small black x
            if is_infeasible and method == 'H1_fixed':
                # Mark a few representative points
                for i in range(0, len(score_x), 200):
                    ax_sup.plot(score_x[i], estimated_d[i], 'x', color='k',
                               markersize=4, markeredgewidth=1, zorder=10)

            if 'unused_x' in arrays:
                unused_x = arrays['unused_x']
                unused_pred = arrays['predicted_strain'] * 1e6

                ax_uns.plot(unused_x, unused_pred, linestyle=linestyles[method_idx],
                           color=color, linewidth=1.5,
                           marker=markers[method_idx], markersize=2, markevery=25,
                           label=label)

        ax_sup.set_xlabel('Position (m)')
        ax_sup.set_ylabel('$d$')
        ax_sup.grid(True, alpha=0.3)
        fig.text(0.12 if shape == 'asymmetric' else 0.12,
                0.90 if shape == 'asymmetric' else 0.44,
                f'{panel_a} {shape.capitalize()}', ha='left', va='top', fontsize=FONT_SIZE_PT)

        ax_uns.set_xlabel('Position (m)')
        ax_uns.set_ylabel(r'Strain ($\mu\varepsilon$)')
        ax_uns.grid(True, alpha=0.3)
        fig.text(0.62 if shape == 'asymmetric' else 0.62,
                0.90 if shape == 'asymmetric' else 0.44,
                f'{panel_b} {shape.capitalize()}', ha='left', va='top', fontsize=FONT_SIZE_PT)

    # Create explicit global legend at top with Reference + methods + Incompatible
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='k', linestyle='-', linewidth=1.5, label='Reference'),
        Line2D([0], [0], color=colors[0], linestyle=linestyles[0], marker=markers[0],
               linewidth=1.5, markersize=4, label='H1-F'),
        Line2D([0], [0], color=colors[1], linestyle=linestyles[1], marker=markers[1],
               linewidth=1.5, markersize=4, label='H1-P'),
        Line2D([0], [0], color=colors[2], linestyle=linestyles[2], marker=markers[2],
               linewidth=1.5, markersize=4, label='Actual force'),
        Line2D([0], [0], marker='x', color='k', linestyle='', markersize=6,
               markeredgewidth=1.5, label='Incompatible')
    ]
    fig.legend(handles=handles, bbox_to_anchor=(0.5, 0.995), loc='upper center',
              ncol=3, frameon=False, fontsize=FONT_SIZE_PT)

    pdf_path = output_path / 'fields_prediction.pdf'
    png_path = output_path / 'fields_prediction.png'
    fig.savefig(pdf_path, dpi=DPI, bbox_inches=None)
    fig.savefig(png_path, dpi=DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Created: {pdf_path.name}")
    return pdf_path


def create_model_discrepancy_figure(results: Dict, output_path: Path):
    """Create model_discrepancy.pdf showing robustness under noise/shear/position."""
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 7.1))

    # Use exact 2x2 positions from fields_prediction - reduce width more
    ax_asym_sup = fig.add_axes([0.12, 0.58, 0.31, 0.27])
    ax_asym_uns = fig.add_axes([0.62, 0.58, 0.30, 0.27])
    ax_double_sup = fig.add_axes([0.12, 0.12, 0.31, 0.27])
    ax_double_uns = fig.add_axes([0.62, 0.12, 0.30, 0.27])

    # Actual robustness regimes from protocol
    regimes = ROBUSTNESS_REGIMES
    regime_labels = ['Noise 20', 'Shear', 'Position']

    methods = ['H1_fixed', 'H1_profile']
    method_labels = ['H1-F', 'H1-P']
    colors = ['C0', 'C1']
    markers = ['o', 's']  # Distinct markers

    for shape, ax_sup, ax_uns, panel_a, panel_b in [
        ('asymmetric', ax_asym_sup, ax_asym_uns, '(a)', '(b)'),
        ('double', ax_double_sup, ax_double_uns, '(c)', '(d)')
    ]:

        for regime_idx, regime in enumerate(regimes):
            for method_idx, method in enumerate(methods):

                sup_errors = []
                uns_errors = []
                infeasible_sup = []
                infeasible_uns = []

                for seed in ROBUSTNESS_SEEDS:
                    key = (shape, regime, seed, method)
                    if key in results['robustness']:
                        result = results['robustness'][key]
                        is_infeasible = result.get('selection', {}).get('no_feasible_candidate', False)

                        sup_err = result['metrics'].get('support_rel_L2_except_intact')
                        uns_err = result['metrics'].get('unused_load_strain_relative_L2')

                        if sup_err is not None and not np.isnan(sup_err) and not np.isinf(sup_err):
                            sup_errors.append(sup_err)
                            if is_infeasible:
                                infeasible_sup.append(sup_err)

                        if uns_err is not None and not np.isnan(uns_err) and not np.isinf(uns_err):
                            uns_errors.append(uns_err)
                            if is_infeasible:
                                infeasible_uns.append(uns_err)

                if sup_errors:
                    x_pos = regime_idx + (method_idx - 0.5) * 0.2
                    median_val = np.median(sup_errors)
                    min_val = np.min(sup_errors)
                    max_val = np.max(sup_errors)

                    ax_sup.plot([x_pos, x_pos], [min_val, max_val],
                               color=colors[method_idx], linewidth=1.5)
                    ax_sup.plot(x_pos, median_val, markers[method_idx],
                               color=colors[method_idx], markersize=5,
                               label=method_labels[method_idx] if regime_idx == 0 else '')

                    # Small points for individual draws
                    for err in sup_errors:
                        ax_sup.plot(x_pos, err, '.', color=colors[method_idx],
                                   markersize=3, alpha=0.5)

                    # Mark ALL infeasible with small black X
                    for err in infeasible_sup:
                        ax_sup.plot(x_pos, err, 'x', color='k', markersize=6,
                                   markeredgewidth=1.5, zorder=10)

                if uns_errors:
                    x_pos = regime_idx + (method_idx - 0.5) * 0.2
                    median_val = np.median(uns_errors)
                    min_val = np.min(uns_errors)
                    max_val = np.max(uns_errors)

                    ax_uns.plot([x_pos, x_pos], [min_val, max_val],
                               color=colors[method_idx], linewidth=1.5)
                    ax_uns.plot(x_pos, median_val, markers[method_idx],
                               color=colors[method_idx], markersize=5,
                               label=method_labels[method_idx] if regime_idx == 0 else '')

                    for err in uns_errors:
                        ax_uns.plot(x_pos, err, '.', color=colors[method_idx],
                                   markersize=3, alpha=0.5)

                    for err in infeasible_uns:
                        ax_uns.plot(x_pos, err, 'x', color='k', markersize=6,
                                   markeredgewidth=1.5, zorder=10)

        ax_sup.set_xticks(range(len(regimes)))
        ax_sup.set_xticklabels(regime_labels)
        ax_sup.set_ylabel(r'$E_d$')
        ax_sup.grid(True, alpha=0.3, axis='y')
        fig.text(0.12 if shape == 'asymmetric' else 0.12,
                0.90 if shape == 'asymmetric' else 0.44,
                f'{panel_a} {shape.capitalize()}', ha='left', va='top', fontsize=FONT_SIZE_PT)

        ax_uns.set_xticks(range(len(regimes)))
        ax_uns.set_xticklabels(regime_labels)
        ax_uns.set_ylabel(r'$E_\varepsilon$')
        ax_uns.grid(True, alpha=0.3, axis='y')
        fig.text(0.62 if shape == 'asymmetric' else 0.62,
                0.90 if shape == 'asymmetric' else 0.44,
                f'{panel_b} {shape.capitalize()}', ha='left', va='top', fontsize=FONT_SIZE_PT)

    # Create explicit global legend at top - lower even more to prevent clipping
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color=colors[0], linestyle='', markersize=5, label='H1-F'),
        Line2D([0], [0], marker='s', color=colors[1], linestyle='', markersize=5, label='H1-P'),
        Line2D([0], [0], marker='x', color='k', linestyle='', markersize=6, markeredgewidth=1.5, label='Incompatible')
    ]
    fig.legend(handles=handles, bbox_to_anchor=(0.5, 0.955), loc='lower center',
              ncol=3, frameon=False, fontsize=FONT_SIZE_PT)

    pdf_path = output_path / 'model_discrepancy.pdf'
    png_path = output_path / 'model_discrepancy.png'
    fig.savefig(pdf_path, dpi=DPI, bbox_inches=None)
    fig.savefig(png_path, dpi=DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Created: {pdf_path.name}")
    return pdf_path


def create_joint_profile_figure(results: Dict, output_path: Path):
    """Create joint_profile.pdf showing same-objective comparison."""
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 6.3))

    # Two full-width rows: objective gap and timing - reduce width slightly
    ax_obj = fig.add_axes([0.13, 0.57, 0.80, 0.27])
    ax_time = fig.add_axes([0.13, 0.12, 0.80, 0.27])

    # All 5 shapes at first seed
    shapes = ACTUAL_SHAPES
    shape_labels = ['Intact', 'Rectangle', 'Asymmetric', 'Double', 'Broad\nshallow']

    colors = {'H1_profile': 'C0', 'H1_joint': 'C1'}
    markers = {'H1_profile': 'o', 'H1_joint': 's'}
    labels = {'H1_profile': 'Profile', 'H1_joint': 'Joint'}

    for shape_idx, shape in enumerate(shapes):
        if (shape, FIRST_SEED) not in results['joint']:
            continue

        joint_data = results['joint'][(shape, FIRST_SEED)]
        fits = joint_data['fits']

        profile_fits = [f for f in fits if f['method'] == 'H1_profile']
        joint_fits = [f for f in fits if f['method'] == 'H1_joint']

        # Should have 3 repeats each
        profile_costs = [f['cost'] for f in profile_fits]
        joint_costs = [f['cost'] for f in joint_fits]

        profile_fit_times = [f['fit_time_s'] for f in profile_fits]
        joint_fit_times = [f['fit_time_s'] for f in joint_fits]

        # Compute relative objective gap and scale to 1e-12: gap * 1e12
        if profile_costs and joint_costs:
            for i, (pc, jc) in enumerate(zip(profile_costs, joint_costs)):
                gap = abs(pc - jc) / max(1.0, abs(jc))
                gap_scaled = gap * 1e12  # Scale for display
                offset = (i - 1) * 0.03  # Small offset for 3 repeats
                ax_obj.plot(shape_idx + offset, gap_scaled, 'ko', markersize=4, alpha=0.7)

        # Plot fit-only time (3 repeats visible with small offsets)
        if profile_fit_times and joint_fit_times:
            for i, (pft, jft) in enumerate(zip(profile_fit_times, joint_fit_times)):
                offset = (i - 1) * 0.03
                ax_time.plot(shape_idx - 0.15 + offset, pft, markers['H1_profile'],
                           color=colors['H1_profile'], markersize=5,
                           label=labels['H1_profile'] if shape_idx == 0 and i == 0 else '')
                ax_time.plot(shape_idx + 0.15 + offset, jft, markers['H1_joint'],
                           color=colors['H1_joint'], markersize=5,
                           label=labels['H1_joint'] if shape_idx == 0 and i == 0 else '')

            # Add median markers
            prof_median = np.median(profile_fit_times)
            joint_median = np.median(joint_fit_times)
            ax_time.plot(shape_idx - 0.15, prof_median, '_',
                        color=colors['H1_profile'], markersize=10, markeredgewidth=2)
            ax_time.plot(shape_idx + 0.15, joint_median, '_',
                        color=colors['H1_joint'], markersize=10, markeredgewidth=2)

    ax_obj.set_xticks(range(len(shapes)))
    ax_obj.set_xticklabels(shape_labels, fontsize=FONT_SIZE_PT)
    ax_obj.set_ylabel(r'Objective gap ($10^{-12}$)')
    # Use linear scale with manual ticks
    ax_obj.set_yscale('linear')
    ax_obj.set_yticks([0, 1, 2, 3])
    ax_obj.grid(True, alpha=0.3, axis='y')
    fig.text(0.13, 0.90, '(a) Objective agreement', ha='left', va='top', fontsize=FONT_SIZE_PT)

    ax_time.set_xticks(range(len(shapes)))
    ax_time.set_xticklabels(shape_labels, fontsize=FONT_SIZE_PT)
    ax_time.set_ylabel('Fit-only time (s)')
    ax_time.legend(bbox_to_anchor=(0.5, 1.14), loc='lower center', ncol=2, frameon=False)
    ax_time.grid(True, alpha=0.3, axis='y')
    fig.text(0.13, 0.46, '(b) Fit-only cost', ha='left', va='top', fontsize=FONT_SIZE_PT)

    pdf_path = output_path / 'joint_profile.pdf'
    png_path = output_path / 'joint_profile.png'
    fig.savefig(pdf_path, dpi=DPI, bbox_inches=None)
    fig.savefig(png_path, dpi=DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Created: {pdf_path.name}")
    return pdf_path


def compute_summary_metrics(results: Dict) -> Dict[str, Any]:
    """Compute machine-readable summary statistics.

    Includes all eight scientific metrics, complete_path_seconds, selected_nfev,
    and full_path_nfev. Preserves backward-compatible legacy keys where needed.
    """
    summary = {
        'primary_counts': {},
        'robustness_counts': {},
        'joint_counts': {},
        'method_comparison': {},
        'groups': [],
        'paired_comparisons': [],
        'failure_statistics': {},
        'timing_statistics': {},
        'evaluation_statistics': {},
        'selection_statistics': {}
    }

    methods = ACTUAL_METHODS
    shapes = ACTUAL_SHAPES
    regimes = ACTUAL_REGIMES

    # All eight scientific metrics from protocol
    all_scientific_metrics = [
        'support_rel_L2_except_intact',
        'absolute_support_L2',
        'loss_area_m',
        'loss_area_absolute_error_m',
        'max_false_loss_for_intact',
        'gamma_RMSE',
        'unused_load_strain_relative_L2',
        'unused_load_peak_strain_error'
    ]

    # Compute failure statistics from actual data
    total_paths = 0
    total_candidates = 0
    total_candidate_failures = 0
    total_starts = 0
    total_start_failures = 0
    infeasible_selections = 0
    unbracketed_high_selections = 0

    # Count primary and robustness results
    for mode in ['primary', 'robustness']:
        for key, result in results[mode].items():
            total_paths += 1

            # Count candidates and their failures
            candidates = result.get('candidates', [])
            for candidate in candidates:
                total_candidates += 1
                if not candidate.get('successful_start', True):
                    total_candidate_failures += 1

                # Count starts within each candidate
                all_starts = candidate.get('all_starts', [])
                for start in all_starts:
                    total_starts += 1
                    if not start.get('success', True):
                        total_start_failures += 1

            # Count selection failures
            selection = result.get('selection', {})
            if selection.get('no_feasible_candidate', False):
                infeasible_selections += 1
            # Unbracketed high: explicit field
            if selection.get('unbracketed_high', False):
                unbracketed_high_selections += 1

    summary['failure_statistics'] = {
        'total_paths': total_paths,
        'total_candidates': total_candidates,
        'total_candidate_failures': total_candidate_failures,
        'total_starts': total_starts,
        'total_start_failures': total_start_failures,
        'infeasible_selections': infeasible_selections,
        'unbracketed_high_selections': unbracketed_high_selections
    }

    # Build groups statistics - all eight scientific metrics
    for mode in ['primary', 'robustness']:
        for regime in (regimes if mode == 'primary' else ROBUSTNESS_REGIMES):
            for shape in shapes:
                for method in (methods if mode == 'primary' else ['H1_fixed', 'H1_profile']):
                    # Collect all eight scientific metrics for this group
                    group_metrics = {metric: [] for metric in all_scientific_metrics}

                    dataset = results[mode] if mode in results else {}
                    for key, result in dataset.items():
                        if key[0] == shape and key[1] == regime and key[3] == method:
                            metrics = result.get('metrics', {})
                            for metric_name in all_scientific_metrics:
                                val = metrics.get(metric_name)
                                if val is not None and not np.isnan(val) and not np.isinf(val):
                                    group_metrics[metric_name].append(val)

                    # Add group entry for each metric with data
                    for metric_name, values in group_metrics.items():
                        if values:
                            summary['groups'].append({
                                'mode': mode,
                                'regime': regime,
                                'shape': shape,
                                'method': method,
                                'metric': metric_name,
                                'count': len(values),
                                'median': float(np.median(values)),
                                'min': float(np.min(values)),
                                'max': float(np.max(values))
                            })

    # Add timing and evaluation statistics groups
    for mode in ['primary', 'robustness']:
        for regime in (regimes if mode == 'primary' else ROBUSTNESS_REGIMES):
            for shape in shapes:
                for method in (methods if mode == 'primary' else ['H1_fixed', 'H1_profile']):
                    # Collect timing metrics
                    complete_path_times = []
                    selected_nfev_values = []
                    full_path_nfev_values = []

                    dataset = results[mode] if mode in results else {}
                    for key, result in dataset.items():
                        if key[0] == shape and key[1] == regime and key[3] == method:
                            # complete_path_seconds
                            cps = result.get('complete_path_seconds')
                            if cps is not None and not np.isnan(cps):
                                complete_path_times.append(cps)

                            # selected_nfev = top-level nfev
                            nfev = result.get('nfev')
                            if nfev is not None:
                                selected_nfev_values.append(nfev)

                            # full_path_nfev = sum over all candidates, all starts
                            total_nfev = 0
                            for candidate in result.get('candidates', []):
                                for start in candidate.get('all_starts', []):
                                    start_nfev = start.get('nfev', 0)
                                    if start_nfev is not None:
                                        total_nfev += start_nfev
                            if total_nfev > 0:
                                full_path_nfev_values.append(total_nfev)

                    # Add timing group entries
                    if complete_path_times:
                        summary['groups'].append({
                            'mode': mode,
                            'regime': regime,
                            'shape': shape,
                            'method': method,
                            'metric': 'complete_path_seconds',
                            'count': len(complete_path_times),
                            'median': float(np.median(complete_path_times)),
                            'min': float(np.min(complete_path_times)),
                            'max': float(np.max(complete_path_times))
                        })

                    if selected_nfev_values:
                        summary['groups'].append({
                            'mode': mode,
                            'regime': regime,
                            'shape': shape,
                            'method': method,
                            'metric': 'selected_nfev',
                            'count': len(selected_nfev_values),
                            'median': float(np.median(selected_nfev_values)),
                            'min': float(np.min(selected_nfev_values)),
                            'max': float(np.max(selected_nfev_values))
                        })

                    if full_path_nfev_values:
                        summary['groups'].append({
                            'mode': mode,
                            'regime': regime,
                            'shape': shape,
                            'method': method,
                            'metric': 'full_path_nfev',
                            'count': len(full_path_nfev_values),
                            'median': float(np.median(full_path_nfev_values)),
                            'min': float(np.min(full_path_nfev_values)),
                            'max': float(np.max(full_path_nfev_values))
                        })

    # Build paired comparisons (method wins/losses/ties)
    # Only use defective shapes (exclude intact) for paired comparisons
    defective_shapes = ['rectangle', 'asymmetric', 'double', 'broad_shallow']
    comparison_pairs = [
        ('H1_fixed', 'H1_profile'),
        ('TV_fixed', 'TV_profile')
    ]

    for regime in regimes:
        for metric_name in ['support_rel_L2_except_intact', 'unused_load_strain_relative_L2']:
            for method_a, method_b in comparison_pairs:
                wins = 0
                losses = 0
                ties = 0
                count = 0

                for shape in defective_shapes:  # Only defective shapes
                    for seed in PRIMARY_SEEDS:
                        key_a = (shape, regime, seed, method_a)
                        key_b = (shape, regime, seed, method_b)

                        if key_a in results['primary'] and key_b in results['primary']:
                            val_a = results['primary'][key_a]['metrics'].get(metric_name)
                            val_b = results['primary'][key_b]['metrics'].get(metric_name)

                            if val_a is not None and val_b is not None:
                                if not np.isnan(val_a) and not np.isnan(val_b):
                                    count += 1
                                    if val_a < val_b:
                                        wins += 1
                                    elif val_a > val_b:
                                        losses += 1
                                    else:
                                        ties += 1

                if count > 0:
                    summary['paired_comparisons'].append({
                        'regime': regime,
                        'method_a': method_a,
                        'method_b': method_b,
                        'metric': metric_name,
                        'wins': wins,
                        'losses': losses,
                        'ties': ties,
                        'count': count
                    })

    # Method comparison statistics (legacy format, keep for backward compatibility)
    for method in methods:
        summary['primary_counts'][method] = 0
        summary['method_comparison'][method] = {
            'support_rel_L2': [],
            'absolute_support_L2': [],
            'loss_area_m': [],
            'gamma_RMSE': [],
            'unused_strain_rel_L2': [],
            'unused_peak_error': []
        }

    for key, result in results['primary'].items():
        shape, regime, seed, method = key

        summary['primary_counts'][method] = summary['primary_counts'].get(method, 0) + 1

        metrics = result['metrics']

        sup_rel = metrics.get('support_rel_L2_except_intact')
        if sup_rel is not None and not np.isnan(sup_rel) and not np.isinf(sup_rel):
            summary['method_comparison'][method]['support_rel_L2'].append(sup_rel)

        sup_abs = metrics.get('absolute_support_L2')
        if sup_abs is not None and not np.isnan(sup_abs) and not np.isinf(sup_abs):
            summary['method_comparison'][method]['absolute_support_L2'].append(sup_abs)

        loss_area = metrics.get('loss_area_m')
        if loss_area is not None and not np.isnan(loss_area) and not np.isinf(loss_area):
            summary['method_comparison'][method]['loss_area_m'].append(loss_area)

        gamma_rmse = metrics.get('gamma_RMSE')
        if gamma_rmse is not None and not np.isnan(gamma_rmse) and not np.isinf(gamma_rmse):
            summary['method_comparison'][method]['gamma_RMSE'].append(gamma_rmse)

        uns_rel = metrics.get('unused_load_strain_relative_L2')
        if uns_rel is not None and not np.isnan(uns_rel) and not np.isinf(uns_rel):
            summary['method_comparison'][method]['unused_strain_rel_L2'].append(uns_rel)

        uns_peak = metrics.get('unused_load_peak_strain_error')
        if uns_peak is not None and not np.isnan(uns_peak) and not np.isinf(uns_peak):
            summary['method_comparison'][method]['unused_peak_error'].append(uns_peak)

    # Compute statistics from collected values
    for method in methods:
        stats = summary['method_comparison'][method]

        for metric_key in ['support_rel_L2', 'absolute_support_L2', 'loss_area_m',
                          'gamma_RMSE', 'unused_strain_rel_L2', 'unused_peak_error']:
            vals = stats[metric_key]
            if vals:
                stats[f'{metric_key}_median'] = float(np.median(vals))
                stats[f'{metric_key}_min'] = float(np.min(vals))
                stats[f'{metric_key}_max'] = float(np.max(vals))
                stats[f'{metric_key}_count'] = len(vals)
            del stats[metric_key]

    summary['robustness_counts']['total'] = len(results['robustness'])
    summary['joint_counts']['total_datasets'] = len(results['joint'])
    summary['joint_counts']['total_fits'] = sum(len(d['fits']) for d in results['joint'].values())

    # Joint timing, gap, and per-shape statistics
    if results['joint']:
        objective_gaps = []
        coefficient_max_diffs = []
        gamma_max_diffs = []
        profile_times = []
        joint_times = []
        profile_assembly_times = []
        joint_assembly_times = []

        # Per-shape timing lists
        per_shape_timings = {}

        for (shape, seed), joint_data in results['joint'].items():
            fits = joint_data['fits']
            profile_fits = [f for f in fits if f['method'] == 'H1_profile']
            joint_fits = [f for f in fits if f['method'] == 'H1_joint']

            # Per-shape three-repeat timings
            shape_profile_fit_times = []
            shape_profile_assembly_times = []
            shape_joint_fit_times = []
            shape_joint_assembly_times = []

            if profile_fits and joint_fits:
                # Match by repeat_idx for gap computation
                for repeat_idx in [0, 1, 2]:
                    pf = [f for f in profile_fits if f['repeat_idx'] == repeat_idx]
                    jf = [f for f in joint_fits if f['repeat_idx'] == repeat_idx]

                    if pf and jf:
                        pf = pf[0]
                        jf = jf[0]

                        # Objective gap
                        obj_gap = abs(pf['cost'] - jf['cost']) / max(1.0, abs(jf['cost']))
                        objective_gaps.append(obj_gap)

                        # Coefficient maximum absolute difference
                        coeff_p = np.array(pf['coefficients'])
                        coeff_j = np.array(jf['coefficients'])
                        coeff_diff = np.max(np.abs(coeff_p - coeff_j))
                        coefficient_max_diffs.append(coeff_diff)

                        # Gamma maximum absolute difference
                        gamma_p = np.array(pf['gamma'])
                        gamma_j = np.array(jf['gamma'])
                        gamma_diff = np.max(np.abs(gamma_p - gamma_j))
                        gamma_max_diffs.append(gamma_diff)

                        # Timing
                        profile_times.append(pf['fit_time_s'])
                        joint_times.append(jf['fit_time_s'])
                        profile_assembly_times.append(pf.get('assembly_time_s', 0))
                        joint_assembly_times.append(jf.get('assembly_time_s', 0))

                        # Per-shape timings
                        shape_profile_fit_times.append(pf['fit_time_s'])
                        shape_joint_fit_times.append(jf['fit_time_s'])
                        shape_profile_assembly_times.append(pf.get('assembly_time_s', 0))
                        shape_joint_assembly_times.append(jf.get('assembly_time_s', 0))

            # Store per-shape timings (3 repeats each)
            per_shape_timings[shape] = {
                'profile_fit_times': shape_profile_fit_times,
                'joint_fit_times': shape_joint_fit_times,
                'profile_assembly_times': shape_profile_assembly_times,
                'joint_assembly_times': shape_joint_assembly_times
            }

        if objective_gaps:
            summary['joint_counts']['objective_gap_median'] = float(np.median(objective_gaps))
            summary['joint_counts']['objective_gap_min'] = float(np.min(objective_gaps))
            summary['joint_counts']['objective_gap_max'] = float(np.max(objective_gaps))

        if coefficient_max_diffs:
            summary['joint_counts']['coefficient_max_diff_median'] = float(np.median(coefficient_max_diffs))
            summary['joint_counts']['coefficient_max_diff_min'] = float(np.min(coefficient_max_diffs))
            summary['joint_counts']['coefficient_max_diff_max'] = float(np.max(coefficient_max_diffs))

        if gamma_max_diffs:
            summary['joint_counts']['gamma_max_diff_median'] = float(np.median(gamma_max_diffs))
            summary['joint_counts']['gamma_max_diff_min'] = float(np.min(gamma_max_diffs))
            summary['joint_counts']['gamma_max_diff_max'] = float(np.max(gamma_max_diffs))

        if profile_times:
            summary['joint_counts']['profile_fit_time_median_s'] = float(np.median(profile_times))
            summary['joint_counts']['joint_fit_time_median_s'] = float(np.median(joint_times))
            summary['joint_counts']['profile_assembly_time_median_s'] = float(np.median(profile_assembly_times))
            summary['joint_counts']['joint_assembly_time_median_s'] = float(np.median(joint_assembly_times))

        # Add per-shape timings
        summary['joint_counts']['per_shape_timings'] = per_shape_timings

    # Success counts
    success_counts = {}
    for key, result in results['primary'].items():
        method = key[3]
        success = result.get('success', False)
        if method not in success_counts:
            success_counts[method] = {'success': 0, 'total': 0}
        success_counts[method]['total'] += 1
        if success:
            success_counts[method]['success'] += 1

    summary['selection_statistics'] = success_counts

    return summary


def save_summary_files(summary: Dict, summary_dir: Path):
    """Save summary_metrics.json and summary_tables.csv."""
    summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = summary_dir / 'summary_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print(f"Created: {json_path.name}")

    # Write CSV with grouped structure: mode/regime/shape/method/metric/count/median/min/max
    csv_path = summary_dir / 'summary_tables.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mode', 'regime', 'shape', 'method', 'metric', 'count', 'median', 'min', 'max'])
        writer.writeheader()

        # Write groups data
        for group in summary.get('groups', []):
            writer.writerow(group)

    print(f"Created: {csv_path.name}")

    return json_path, csv_path


def main():
    """Main entry point for figure generation."""
    parser = argparse.ArgumentParser(
        description='Generate publication figures from formal experiment results'
    )
    parser.add_argument(
        '--results',
        required=True,
        help='Path to results directory (e.g., revised/results_v1)'
    )
    parser.add_argument(
        '--verification',
        required=True,
        help='Path to verification_results.json'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output directory for figures (e.g., revised/manuscript/figures)'
    )
    parser.add_argument(
        '--summary-out',
        required=True,
        help='Output directory for summary files (e.g., revised/results_v1/summary)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results)
    verification_path = Path(args.verification)
    output_dir = Path(args.out)
    summary_dir = Path(args.summary_out)

    print("=" * 60)
    print("Postprocessing: Figure generation")
    print("=" * 60)

    verify_font_availability()

    print("\n[1/8] Loading and validating results...")
    try:
        results = load_and_validate_results(results_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    print("\n[2/8] Loading verification results...")
    try:
        verification = load_verification(verification_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/8] Creating verification figure...")
    create_verification_figure(verification, output_dir)

    print("\n[4/8] Creating intact control figure...")
    create_intact_control_figure(results, output_dir)

    print("\n[5/8] Creating field errors figure...")
    create_field_errors_figure(results, output_dir)

    print("\n[6/8] Creating fields prediction figure...")
    create_fields_prediction_figure(results, output_dir)

    print("\n[7/8] Creating model discrepancy figure...")
    create_model_discrepancy_figure(results, output_dir)

    print("\n[8/8] Creating joint profile figure...")
    create_joint_profile_figure(results, output_dir)

    print("\n[Summary] Computing summary metrics...")
    summary = compute_summary_metrics(results)
    save_summary_files(summary, summary_dir)

    print("\n" + "=" * 60)
    print("Figure generation complete")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Summary directory: {summary_dir}")
    print("\nGenerated figures:")
    for fig_name in ['verification.pdf', 'intact_control.pdf', 'field_errors.pdf',
                     'fields_prediction.pdf', 'model_discrepancy.pdf', 'joint_profile.pdf']:
        fig_path = output_dir / fig_name
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            print(f"  {fig_name:30s} {size_kb:8.1f} KB")

    print("\nSummary files:")
    for summary_file in ['summary_metrics.json', 'summary_tables.csv']:
        summary_path = summary_dir / summary_file
        if summary_path.exists():
            print(f"  {summary_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

