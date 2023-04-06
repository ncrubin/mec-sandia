import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from ase.units import Hartree

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"
plt.rcParams['font.sans-serif'] = 'Arial'

colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]

def estimate_error_kinetic_energy(sigma: float, kcut: float) -> float:
    a = kcut 
    b = (2*sigma**2.0)
    #t1 = a * np.exp(-(a**2.0)*b) / (2*b)
    t1 = 2 * a * np.exp(-a**2.0/b)
    #t2 = np.sqrt(np.pi) * scipy.special.erfc(a*b**0.5)/(4*b**(3.0/2.0)) 
    t2 = np.sqrt(np.pi*b) * scipy.special.erfc(a/(b**0.5))
    return 0.25*b*(t1 + t2)

L_bohr = 15
nx_grid = 2**np.arange(2, 9)  # quantum algorithm takes things as a power of 2
ke_cutoffs_eV = 0.5 * (2 * np.pi)**2 * nx_grid**2 / L_bohr**2 * 27.11 # highest energy components in eV
sigmas = [1, 4, 6, 10]
errors = []
mass_proj = 1836
fig, ax = plt.subplots(nrows=1, ncols=1)
sigma = 10
prefactor_erfc = (L_bohr/2*np.pi) / (2*mass_proj*np.sqrt(2*np.pi)*sigma)
prefactor = (L_bohr/2*np.pi) / (2*mass_proj)
print("erfc ke = ", 2*estimate_error_kinetic_energy(10, 0)*prefactor_erfc)
print("expected ke = ", prefactor*sigma**2.0)
for isigma, sigma in enumerate(sigmas[::-1]):
    errors = []
    for ib, ecut_ev in enumerate(ke_cutoffs_eV):
        ecut_ha = ecut_ev / Hartree
        kcut_au = (2*ecut_ha)**0.5
        prefactor = (L_bohr/2*np.pi) / (2*mass_proj*sigma*np.sqrt(2*np.pi))
        error = estimate_error_kinetic_energy(sigma, kcut_au)
        errors.append(np.sqrt(prefactor*error)) 

    ax.plot(ke_cutoffs_eV, errors, marker="o", label=fr"$\sigma_k$ = {sigma}", color=colors[isigma])
ax.tick_params(which='both', labelsize=14, direction='in')
ax.set_xlabel("$E_{cut}$ [eV]", fontsize=14)
ax.set_ylabel(r"Approximate Projectile Kinetic Energy Error [Ha]", fontsize=14)
ax.tick_params(which='both', labelsize=14, direction='in')
ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
ax.set_title("One Dimensional Gaussian Kinetic Energy standard Error")
ax.set_xscale("log")
ax.set_yscale("log")
plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
plt.ylim([1e-11, 1])
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig("cutoff_approx_erfc.png", dpi=200)
plt.savefig("cutoff_approx_erfc.pdf", dpi=300)