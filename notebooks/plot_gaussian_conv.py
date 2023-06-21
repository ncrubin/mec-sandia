"""
How fine a mesh does the projectile need?

We can answer this by looking at the Gaussian and comparing the expectation value
to the infinite mesh limit. 
"""
import matplotlib.pyplot as plt
import numpy as np
from ase.units import Hartree

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "stix"

colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]

# Check convergence wrt PW cutoff
v_proj = 4.0  # atomic units just taken from carbon
mass_proj = 1836
ke = 0.5 * mass_proj * v_proj**2.0  # classical ke
sigma_k = 4.0


box_length = 15
ecut_ev = np.logspace(1, 6, 8)
mass_proj = 1836
from mec_sandia.gaussians import (
    estimate_error_kinetic_energy,
    estimate_error_kinetic_energy_with_proj,
    kinetic_energy,
)

fig, ax = plt.subplots(nrows=1, ncols=1)

ndim = 1
kproj = np.zeros(ndim)
kproj[0] = mass_proj * v_proj

for isk, sigma_k in enumerate([10, 6, 4, 1]):
    deltas = []
    deltas_approx = []
    for ec in ecut_ev:
        ecut_ha = ec / Hartree
        # Compute the kinetic energy of the projectile using a discrete gaussian
        # in 1d.
        discrete_kin_e = (
            kinetic_energy(
                ecut_ha,
                box_length,
                sigma_k,
                ndim=1,
                kproj=kproj,
            )
            / mass_proj
        )
        # This is the KE of the projectile evaluated from the integral
        # expression.
        intgl_kin_e = (ndim * sigma_k**2 + np.dot(kproj, kproj)) / (2 * mass_proj)
        # Absolute error in the kinetic energy.
        deltas.append(np.abs(discrete_kin_e - intgl_kin_e))
        # The factor of 1/2 is because of how we're using ecut, we only sum from
        # [-nmax/2, nmax/2] which means we should inegrate from [-kmax/2,
        # kmax/2]. (kmax = (2Ecut)^{1/2}) is a spherical cutoff.
        kcut = 0.5 * (2 * ecut_ha) ** 0.5
        # Estimate the error using the integral formula from the overleaf.
        ke_err_apprx = (
            estimate_error_kinetic_energy_with_proj(kcut, sigma_k, kproj[0]) / mass_proj
        )
        deltas_approx.append(ke_err_apprx)
    ax.plot(
        ecut_ev,
        deltas,
        marker="o",
        color=f"{colors[isk]}",
        label=f"$\sigma_k = {sigma_k}$",
        markerfacecolor="None",
    )
    ax.plot(
        ecut_ev,
        deltas_approx,
        marker="^",
        color=f"{colors[isk]}",
        lw=0,
        markerfacecolor="None",
    )
ax.tick_params(which="both", labelsize=14, direction="in")
ax.set_xlabel(r"$E_{\mathrm{cut}}$ [eV]", fontsize=14)
ax.set_ylabel(r"Projectile Kinetic Energy Error [Ha]", fontsize=14)
ax.tick_params(which="both", labelsize=14, direction="in")
ax.legend(loc="lower left", fontsize=14, ncol=1, frameon=False)
plt.xscale("log")
plt.yscale("log")
plt.ylim([1e-12, 1e4])

plt.savefig("cutoff_convergence_1d.pdf", bbox_inches="tight", dpi=300)
plt.savefig("cutoff_convergence_1d.png", bbox_inches="tight", dpi=200)
