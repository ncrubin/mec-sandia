import numpy as np
from ase.units import Hartree
import math
import matplotlib.pyplot as plt
from pyscf.lib.numpy_helper import cartesian_prod

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "stix"

colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]

# Check convergence wrt PW cutoff
v_proj = 4.0 # atomic units just taken from carbon
mass_proj = 1836
ke = 0.5 * mass_proj * v_proj**2.0 # classical ke
kproj = np.array([mass_proj*v_proj, 0, 0])
sigma_k = 4.0

def get_ngmax(ecut, box_length):
    ng_max = math.ceil(np.sqrt(2*ecut)/(2*np.pi/box_length))
    if ng_max % 2 == 0:
        ng_max += 1
    return ng_max

L_bohr = 15
import pandas as pd
df = pd.DataFrame({"cutoffs": [1000, 10000, 5e4, 1e5]}) 
for sigma_k in [4.0]:
    results = []
    for ecut_ev in df.cutoffs.values: 
        ecut_ha = (ecut_ev/Hartree)
        nmax = get_ngmax(ecut_ha, L_bohr)
        norm = 0.0
        sum_k = 0.0
        #for ni, nj, nk in itertools.product(range(-nmax, nmax), repeat=3):
        kgrid = np.arange(-nmax/2, nmax/2)
        kxyz_grid = (2*np.pi/L_bohr) * cartesian_prod([kgrid, kgrid, kgrid])
        kpsq = np.sum((kxyz_grid+kproj[None,:])**2.0, axis=-1)
        ksq = np.sum(kxyz_grid**2.0, axis=-1)
        prefactor = (np.sqrt(2*np.pi)/(sigma_k*L_bohr))**3.0
        sum_k = np.sum(kpsq*np.exp(-ksq/(2*sigma_k**2.0)))
        norm = np.sum(np.exp(-ksq/(2*sigma_k**2.0)))
        ke_sum = sum_k*prefactor/(2*mass_proj) - ke
        ke_int = 3*sigma_k**2.0/(2*mass_proj)
        print(ke_sum, ke_int, ke_sum-ke_int)
        results.append((ke_sum-ke_int))
    df[f"sigma_{sigma_k}"] = results

# for i, sigma_k in enumerate([4.0, 6.0, 10.0][::-1]):
#     sigma_d = int(sigma_k)
#     plt.plot(df.cutoffs, np.abs(df[f"sigma_{sigma_k}"]), marker="o", label=f"$\sigma_k$ = {sigma_d}", color=colors[i])
# plt.tick_params(which="both", labelsize=14, direction="in")
# plt.legend(frameon=False, loc="upper right", fontsize=14)
# plt.axhline(1e-4, ls=':', color="grey")
# plt.xlabel(r"$E_{\mathrm{cut}} (\mathrm{eV})$", fontsize=14)
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("Kinetic Energy Error (Ha)", fontsize=14)
# plt.savefig("cutoff_convergence.pdf", bbox_inches="tight", dpi=300)