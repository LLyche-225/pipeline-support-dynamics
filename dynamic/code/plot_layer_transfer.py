"""Plot the final mesh-checked layer comparison at the manuscript width."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parents[1]
def read(path):
    with (root / path).open() as stream:
        return list(csv.DictReader(stream))

primary = read("results/layer/layer_metrics.csv")
refined = {(float(r["E_Pa"]), r["shape"], float(r["Hz"])): r
           for r in read("results/layer_refined/cases.csv")}
extent = read("audit/layer_replay_metrics.csv")
colors = {50e6: "#205a86", 200e6: "#a04823"}
plt.rcParams.update({"font.family": "cmr10", "font.size": 10, "axes.titlesize": 10,
                     "axes.labelsize": 10, "legend.fontsize": 10,
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
                     "mathtext.fontset": "cm", "pdf.fonttype": 42})
fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 3.3))
fig.subplots_adjust(left=.105, right=.985, bottom=.29, top=.85, wspace=.39)
for E, color in colors.items():
    for shape, marker, line in [("intact","o","-"),("asymmetric","s","--")]:
        rows = sorted((r for r in primary if float(r["soil_E_Pa"])==E and r["shape"]==shape),
                      key=lambda r: float(r["freq_hz"]))
        assert len(rows)==4
        freq, error = [], []
        for r in rows:
            f = float(r["freq_hz"])
            fine = refined.get((E,shape,f))
            if fine:
                assert fine["within_one_percent"]=="True"
            else:
                assert r["converged"]=="True", r["case_id"]
            freq.append(f)
            error.append(100*float(fine["error"] if fine else r["complex_L2"]))
        left.plot(freq,error,color=color,marker=marker,ls=line,ms=5,mfc="white")
    rows = sorted((r for r in extent if float(r["E_MPa"])*1e6==E and
                   r["shape"]=="intact" and float(r["Hz"])>0),key=lambda r: float(r["Hz"]))
    right.plot([float(r["Hz"]) for r in rows], [100*float(r["half50_to_half60"]) for r in rows],
               color=color,marker="o",ms=5,mfc="white")
for ax,title in [(left,"(a) Local springs vs soil layer"),(right,"(b) Lateral boundary check")]:
    ax.set_title(title,loc="left",pad=7)
    ax.set(xlabel="Frequency (Hz)",xticks=[2,5,10,20],xlim=(1,21))
    ax.tick_params(labelsize=10)
    ax.grid(axis="y",color=".87",lw=.5)
left.set(yscale="log",ylabel="Strain error (%)")
left.set_yticks([10,30,100,300],labels=["10","30","100","300"])
right.set(ylabel="Change in strain (%)",ylim=(-3,82),yticks=[0,20,40,60,80])
handles = [Line2D([],[],color=c,label=f"{int(E/1e6)} MPa") for E,c in colors.items()]
handles += [Line2D([],[],color=".2",marker=m,ls=ls,mfc="white",label=lab)
            for m,ls,lab in [("o","-","Intact"),("s","--","Asymmetric")]]
fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=4,frameon=False)
out = root/"manuscript/figures_v5/layer_publication.pdf"
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out)
plt.close(fig)
