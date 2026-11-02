"""Independent consistent-mass replay on the unchanged static layer stiffness."""
import sys
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
import continuum_reference as ref
from shape_factory import get_shape_function
from beam_inverse import EI

def replay(E, shape, h, half=50.):
    x=np.unique(np.r_[np.linspace(-half,half,round(2*half/h)+1),3.85,8.15,1.2])
    z=np.linspace(0.,10.,round(10/h)+1); gauges=np.linspace(-12.,12.,61)
    # Observe the static solver's verified assembled stiffness; leave it unmodified.
    state={}
    def snapshot(frame,event,arg):
        if frame.f_code is ref.continuum_forward.__code__ and event=="return":
            state.update({k:frame.f_locals[k] for k in ("stiffness","free","rhs","beam_basis","soil_mesh")})
    sys.setprofile(snapshot)
    try:
        static=ref.continuum_forward(x,z,np.array([6.]),get_shape_function(shape),gauges,.2,EI,36e6,E*1e6,.3,.6,1000.,4.3,.3)
    finally:
        sys.setprofile(None)
    mesh=state["soil_mesh"];t=mesh.t.T;areas=np.ptp(mesh.p[0,t],axis=1)*np.ptp(mesh.p[1,t],axis=1)
    S=np.array([[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]])/36
    def assemble(ix,local,ndof):
        rows=np.broadcast_to(ix[:,:,None],local.shape).ravel();cols=np.broadcast_to(ix[:,None,:],local.shape).ravel()
        return sp.coo_matrix((local.ravel(),(rows,cols)),shape=(ndof,ndof)).tocsr()
    Ms=sp.kron(assemble(t,(1800*.6*areas)[:,None,None]*S,len(mesh.p.T)),sp.eye(2))
    hb=np.diff(x);blocks=[]
    for l in hb:
        blocks.append(400*l/420*np.array([[156,22*l,54,-13*l],[22*l,4*l*l,13*l,-3*l*l],[54,13*l,156,-22*l],[-13*l,-3*l*l,-22*l,4*l*l]]))
    dof=2*np.arange(len(hb))[:,None]+np.arange(4);Mb=assemble(dof,np.array(blocks),len(x)*2)
    free=state["free"];K=state["stiffness"][free][:,free];M=sp.block_diag((Mb,Ms),format="csr")[free][:,free]
    b=state["rhs"][free,0];basis=state["beam_basis"]
    G=-.3/.2*(ref._line_probe_matrices(basis,gauges+.1)[1]-ref._line_probe_matrices(basis,gauges-.1)[1])
    wa,wb=2*np.pi*np.array([2.,20.]);alpha=.04*wa*wb/(wa+wb);beta=.04/(wa+wb)
    out=[]
    for f in [0,2,5,10,20]:
        w=2*np.pi*f;A=(1+1j*w*beta)*K+(-w*w+1j*w*alpha)*M;u=np.zeros(len(state["rhs"]),complex)
        u[free]=spsolve(A.tocsc(),b);out.append(G@u[:basis.N])
    assert np.linalg.norm(out[0]-static["strain"][:,0])/np.linalg.norm(out[0])<1e-7
    return np.array(out)

if __name__=="__main__":
    data={f"E{E}_{s}":replay(E,s,.5) for E in (50,200) for s in ("intact","asymmetric")}
    np.savez_compressed(Path("audit/layer_independent_mass.npz"),freq=[0,2,5,10,20],x=np.linspace(-12,12,61),**data)
    print({k:np.linalg.norm(v,axis=1).tolist() for k,v in data.items()})
