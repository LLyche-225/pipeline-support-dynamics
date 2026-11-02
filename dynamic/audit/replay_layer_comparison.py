import numpy as np,json,csv
from pathlib import Path
from dynamic_beam import aligned_mesh
from dynamic_beam_skfem import SkfemDynamics
from replay_layer_mass import replay
from shape_factory import get_shape_function
cal=json.load(open("inputs/layer_calibration.json"))["calibrations"];rows=[];fields={}
for entry in cal:
 E=int(entry["soil_E_Pa"]/1e6);k=entry["k0_N_m2"];mesh=aligned_mesh(.125,extra_edges=(1.2,));bare=SkfemDynamics(mesh,lambda x:np.ones_like(x)).K
 for sh in ("intact","asymmetric"):
  b=SkfemDynamics(mesh,get_shape_function(sh));b.K=bare+(b.K-bare)*(k/18e6);wa,wb=2*np.pi*np.array([2.,20.]);b.C=.04*(wa*wb*b.M+b.K)/(wa+wb)
  light=replay(E,sh,.5);fine=replay(E,sh,.25);long=replay(E,sh,.25,60.) if sh=="intact" else None
  fields[f"E{E}_{sh}_fine"]=fine
  for i,f in enumerate([0.,2.,5.,10.,20.]):
   z=b.solve_frequency(2*np.pi*f)["force"]["strain"];y=fine[i]
   e=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
   rows.append([E,sh,f,e(z,y),e(light[i],y),e(y,long[i]) if long is not None else None])
with open("audit/layer_replay_metrics.csv","w") as out:
 w=csv.writer(out);w.writerow(["E_MPa","shape","Hz","error","mesh_05_to_025","half50_to_half60"]);w.writerows(rows)
np.savez_compressed("audit/layer_replay_fine.npz",**fields)
print(rows)
