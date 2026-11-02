"""Independent clamped uniform-pipe check; no FE matrices or plotting code."""
import numpy as np
from scipy.linalg import expm,solve
from functools import lru_cache
import json
from pathlib import Path
from math import pi,sqrt
D,t,tl=.6,.01,.005
EI=pi/64*(200e9*(D**4-(D-2*t)**4)+2.6e9*((D-2*t)**4-(D-2*t-2*tl)**4))
k,m=18e6,400.;c=.04*sqrt(k*m)
x=np.unique(np.r_[np.arange(-50,51,2.),6-4.3/2,6+4.3/2]); n=len(x);g=.2
gauges=np.linspace(-12,12,61); rows=[]
for fr in [0.,.25,.75,1.,1.15]:
 w=fr*sqrt(k/m)
 for channel in ['force','base']:
  def prop(x0,dx,mid):
   f=(1000/4.3 if channel=='force' and abs(mid-6)<4.3/2 else 0.)
   a=(.0005*(k+1j*w*c) if channel=='base' and abs(mid)<20 else 0.)
   h=np.zeros((7,7),complex);h[0,1]=h[1,2]=h[2,3]=1;h[3,0]=-(k-m*w*w+1j*w*c)/EI;h[3,4]=(f+a)/EI;h[3,5]=a/EI;h[5,6]=-pi/20;h[6,5]=pi/20
   E=expm(h*dx)
   return E[:4,:4],E[:4,4:]@np.array([1,np.cos(pi*x0/20),np.sin(pi*x0/20)])
  A=np.zeros((4*n,4*n),complex);b=np.zeros(4*n,complex)
  for j in range(n-1):
   sl=slice(4*j,4*j+4); p,q=prop(x[j],x[j+1]-x[j],(x[j]+x[j+1])/2)
   A[sl,sl]=-p;A[sl,4*j+4:4*j+8]=np.eye(4);b[sl]=q
  A[-4,0]=A[-3,1]=A[-2,-4]=A[-1,-3]=1;v=solve(A,b).reshape(-1,4)
  def rotation(z):
   j=np.searchsorted(x,z,side='right')-1;p,q=prop(x[j],z-x[j],(x[j]+x[j+1])/2);return (p@v[j]+q)[1]
  e=np.array([-.3/g*(rotation(z+g/2)-rotation(z-g/2)) for z in gauges])
  rows.append(dict(ratio=fr,channel=channel,real=e.real.tolist(),imag=e.imag.tolist()))
Path('audit/intact_exact.json').write_text(json.dumps(dict(method='piecewise matrix exponential; 100 m clamped; continuous field; 0.2 m gauge',EI=EI,x=gauges.tolist(),rows=rows)))
for channel in ['force','base']:
 fields=[np.array(r['real'])+1j*np.array(r['imag']) for r in rows if r['channel']==channel]
 print(channel,'quasistatic errors:',[float(np.linalg.norm(fields[0]-y)/np.linalg.norm(y)) for y in fields[1:]])
