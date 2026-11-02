"""Refine the high-frequency fixture cases that failed the original mesh check.
Run from the extracted research root with PYTHONPATH=code:vendor.
"""
from pathlib import Path
import csv
import json
import numpy as np
import dynamic_layer as d
import dynamic_layer_run as driver

out = Path("results/layer_refined")
out.mkdir(parents=True, exist_ok=True)
cal = json.loads(Path("inputs/layer_calibration.json").read_text())["calibrations"]
rows, fields = [], {"x": d.GAUGE_POINTS}
for entry in cal:
    modulus, k = entry["soil_E_Pa"], entry["k0_N_m2"]
    for shape in ("intact", "asymmetric"):
        # The unchanged 0.25-to-0.125 m comparison selected these groups.
        freqs = [10., 20.] if modulus == 50e6 else [20.]
        runs = []
        for h in (1/8, 1/12):
            mesh = driver.make_mesh(h, 50., shape)
            runs.append(driver.pair_strains(mesh, driver.make_depth(h), shape,
                                            modulus, k, freqs, *d.rayleigh_coefficients()))
        for f in freqs:
            (rc, pc), (rf, pf) = runs
            y, z = rf[f]["strain"], pf[f]["strain"]
            dy, dz = d.complex_l2_relative(rc[f]["strain"], y), d.complex_l2_relative(pc[f]["strain"], z)
            key = f"E{int(modulus/1e6)}_{shape}_{int(f)}Hz"
            fields["ref_"+key], fields["pred_"+key] = y, z
            fields["coarse_ref_"+key], fields["coarse_pred_"+key] = rc[f]["strain"], pc[f]["strain"]
            rows.append([key, modulus, shape, f, 1/12, dy, dz, d.complex_l2_relative(z,y),
                         float(np.linalg.norm(z)/np.linalg.norm(y)), max(dy,dz)<0.01])
            print(rows[-1], flush=True)
np.savez_compressed(out/"fields.npz", **fields)
with (out/"cases.csv").open("w",newline="") as file:
    w=csv.writer(file);w.writerow(["key","E_Pa","shape","Hz","h_m","layer_change","local_change","error","amplitude_ratio","within_one_percent"]);w.writerows(rows)
