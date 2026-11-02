"""Independent scalar optimization, derivative and information checks."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from beam_inverse import profile_load_amplitude

rng = np.random.default_rng(20261102)
errors = {"gamma": [], "residual_derivative": [], "schur": []}
for index in range(200):
    f = rng.normal(size=13) * 10
    df = rng.normal(size=(13, 7))
    y = f * rng.uniform(0.6, 1.4) + rng.normal(size=13)
    mean = rng.uniform(0.65, 1.35)
    tau = 10 ** rng.uniform(-3, 0)
    gamma, derivative, prior, prior_jac = profile_load_amplitude(f, df, y, mean, tau)
    objective = lambda a: np.sum((a * f - y) ** 2) + ((a - mean) / tau) ** 2
    trial = minimize_scalar(objective, bounds=(0.7, 1.3), method="bounded", options={"xatol": 1e-13})
    scalar = min([trial.x, 0.7, 1.3], key=objective)
    errors["gamma"].append(abs(gamma - scalar))
    direction = rng.normal(size=7)
    step = 1e-5
    def augmented(t):
        ft = f + t * (df @ direction)
        gt, _, pt, _ = profile_load_amplitude(ft, df, y, mean, tau)
        return np.r_[gt * ft - y, pt]
    fd = (augmented(step) - augmented(-step)) / (2 * step)
    jac = np.vstack([gamma * df + np.outer(f, derivative), prior_jac])
    errors["residual_derivative"].append(np.linalg.norm(fd - jac @ direction) / max(1.0, np.linalg.norm(fd)))
    # Eliminate the amplitude column from the augmented JOINT Jacobian.
    joint = np.block([[gamma * df, f[:, None]], [np.zeros((1, 7)), np.array([[1 / tau]])]])
    normal = joint.T @ joint
    eliminated = normal[:7, :7] - np.outer(normal[:7, 7], normal[7, :7]) / normal[7, 7]
    expression = gamma ** 2 * df.T @ (np.eye(13) - np.outer(f, f) / (f @ f + tau ** -2)) @ df
    errors["schur"].append(np.linalg.norm(eliminated - expression) / np.linalg.norm(expression))
    known = gamma ** 2 * df.T @ df
    free = gamma ** 2 * df.T @ (np.eye(13) - np.outer(f, f) / (f @ f)) @ df
    assert np.linalg.eigvalsh(known - expression).min() > -1e-10
    assert np.linalg.eigvalsh(expression - free).min() > -1e-10
summary = {"trials": 200, "max_errors": {key: max(value) for key, value in errors.items()}}
assert summary["max_errors"]["gamma"] < 1e-7
assert summary["max_errors"]["residual_derivative"] < 1e-7
assert summary["max_errors"]["schur"] < 1e-12
print(json.dumps(summary, indent=2))
