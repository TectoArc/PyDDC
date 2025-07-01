import V
from simulation import ModelInitialization
from tables import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import numpy as np

ModelInitialization("inputs.json")
out = "/home/sayan/ctrans_mod/results/"
      
dfs = [f for f in os.listdir(out) if f.endswith(".h5")]

flst = []
for df in [dfs[0]]:
    f = open_file(out + df, "r")
    for node in f.walk_nodes("/", classname="Group"):
        n = node
        print(n)
    grs = f.get_node(n)
    c = grs.concentration[:-1, 1:-1, 1:-1]
    rho = grs.density[:-1, 1:-1, 1:-1]
    ppos = grs["plume_config"][:-1]
    J = grs["flux"][:-1, :]
    t = grs["time"][:-1]
    J /= J.max()
    D = grs.dispersion[100, :, :, 0]
    
# c_int = c[1:-1, 1:-1]
c /= rho
ct = c[140, :, :] 
ct = np.ma.masked_where(ct==0, ct)
cmap = cm.jet_r.copy()
cmap.set_bad(color='black')

P = []
for p in ppos:
    P.append(len(p))

fig, ax  = plt.subplots()
C = []
for conc in c:
    C.append(conc*V.dv[1:-1, 1:-1])

C = np.array(C)
C = np.sum(C, axis=(1, 2))

fig, ax = plt.subplots()
plt.plot(t, C)
# ax.scatter(ppos[:, 1], ppos[:, 2], s=0.5)
# ax.set_xlim([0, V.L]); ax.set_ylim([0, V.H])
# c_im1 = ax.imshow(c, origin="lower", aspect="equal", interpolation=None, cmap=cmap, extent=(0., V.L, 0., V.H))
# c_im1 = ax.imshow(ct, origin="lower", aspect="equal", interpolation=None, cmap=cmap, extent=(0., V.L, 0., V.H))
# fig.colorbar(c_im1, ax=ax, location="top", ticks=np.linspace(ct[ct!=0].min(), ct[ct!=0].max(), 4), shrink=0.8) 
# fig.tight_layout()
# ax.set_aspect("equal")
fig.savefig(out + "total_mass.png", dpi=300)
# print(V.x.shape, V.y.shape, c.shape)
# sdr = np.gradient(c, V.y, V.x)
# print(sdr.shape)
