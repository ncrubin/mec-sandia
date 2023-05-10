import math

import matplotlib.pyplot as plt
import numpy as np
from ase.units import Hartree
from pyscf.lib.numpy_helper import cartesian_prod

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "stix"

colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]

# Check convergence wrt PW cutoff
v_proj = 4.0  # atomic units just taken from carbon
mass_proj = 1836
ke = 0.5 * mass_proj * v_proj**2.0  # classical ke
kproj = np.array([mass_proj * v_proj, 0, 0])
kproj[0] = 0.0
sigma_k = 4.0


# def get_ngmax(ecut, box_length):
#     ng_max = math.ceil(np.sqrt(2 * ecut) / (2 * np.pi / box_length))
#     # if ng_max % 2 == 0:
#     #     ng_max += 1
#     return ng_max


# L_bohr = 15
# from dataclasses import dataclass, field
# from typing import List

# nx_grid = 2**np.arange(2, 9)  # quantum algorithm takes things as a power of 2
# ke_cutoffs_eV = 0.5 * (2 * np.pi)**2 * nx_grid**2 / L_bohr**2 * 27.11 # highest energy components in eV

# @dataclass
# class Results:
#     cutoffs: List = field(
#         default_factory=lambda: list(ke_cutoffs_eV)
#     )
#     deltas: List = field(default_factory=lambda: [])
#     integral_val: List = field(default_factory=lambda: [])
#     norm_inf: List = field(default_factory=lambda: [])
#     norm: List = field(default_factory=lambda: [])


# #sigmas = {4**0.5: Results(), 6**0.5: Results(), 8**0.5: Results(), 10**0.5: Results()}
# sigma_vals = [1, 4, 6, 10]
# sigmas = {s: Results() for s in sigma_vals}
# #sigmas = {4: Results()}

# fig, ax = plt.subplots(nrows=1, ncols=1)
# for sigma_k, res_dict in sigmas.items():
#     for ecut_ev in res_dict.cutoffs:
#         ecut_ha = ecut_ev / Hartree
#         nmax = get_ngmax(ecut_ha, L_bohr)
#         kgrid = np.arange(-nmax/2, nmax/2+1)
#         grid_spacing = 2 * np.pi / L_bohr
#         kxyz_grid = grid_spacing * kgrid
#         kpsq = (kxyz_grid + kproj[0]) ** 2.0
#         ksq = kxyz_grid**2.0
#         _prefactor = np.sqrt(2 * np.pi) / (sigma_k * L_bohr)
#         gaussian = np.exp(-ksq / (2 * sigma_k**2.0))
#         sum_k = np.sum(kpsq * gaussian)
#         prefactor = 1.0 / np.sum(gaussian)
#         ke_sum = sum_k * prefactor / (2 * mass_proj)
#         ke_int = (sigma_k**2.0 + np.dot(kproj, kproj))/ (2 * mass_proj)
#         res_dict.deltas.append(np.abs(ke_sum-ke_int))
#         res_dict.norm.append(prefactor)
#         res_dict.norm_inf.append(_prefactor)
#         res_dict.integral_val.append(ke_int)
#         plt.plot(kxyz_grid, kpsq*gaussian * prefactor, marker="o", label=f"{nmax}")
#         xs = np.linspace(-abs(kxyz_grid[0]), abs(kxyz_grid[0]), 200)
#         dg = abs(xs[0] - xs[1])
#         print("ecut = ", ecut_ev)
#         print("nmax = ", nmax)
#         print("min k = ", min(grid_spacing*kgrid))
#         print("ke exact = ", ke_int)
#         print("ke sum= ", ke_sum)
#         print("norm exact = ", _prefactor)
#         print("norm sum= ", prefactor)
#         print("sigma = ", sigma_k)
#         if ecut_ev == 1e5:
#             plt.plot(xs, xs**2.0*np.exp(-xs**2.0/(2*sigma_k**2.0))*_prefactor, marker="o", label="continuous")
#             plt.plot(xs, np.exp(-xs**2.0/(2*sigma_k**2.0))*_prefactor, marker="o", label="continuous")
#     print(res_dict.deltas)
#     plt.legend()
#     plt.savefig(f"gaussian_fig_{sigma_k}.pdf")
#     plt.cla()


# plt.cla()
# ax.tick_params(which='both', labelsize=14, direction='in')
# ax.set_xlabel(r"$E_{\mathrm{cut}}$ [eV]", fontsize=14)
# ax.set_ylabel(r"Projectile Kinetic Energy Error [Ha]", fontsize=14)
# ax.tick_params(which='both', labelsize=14, direction='in')
# ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
# #ax.set_title("One Dimensional Gaussian Kinetic Energy standard Error")
# ax.set_xscale("log")
# ax.set_yscale("log")
# plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
# for i, sigma_k in enumerate(list(sigmas.keys())[::-1]):
#     plt.plot(
#         sigmas[sigma_k].cutoffs,
#         np.abs(sigmas[sigma_k].deltas),
#         marker="o",
#         label=rf"$\sigma_k$ = {sigma_k}",
#         color=colors[i],
#     )
# #plt.ylim([1e-11, 1e-2])
# ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
# plt.ylabel("Kinetic Energy Error (Ha)", fontsize=14)
# plt.savefig(f"cutoff_convergence_1d_{kproj[0]}.pdf", bbox_inches="tight", dpi=300)
# plt.savefig(f"cutoff_convergence_1d_{kproj[0]}.png", bbox_inches="tight", dpi=200)

box_length = 15
ecut_ev = np.logspace(1, 6, 8)
mass_proj = 1836
prefactor = 1.0 / mass_proj
from mec_sandia.gaussians import estimate_error_kinetic_energy, kinetic_energy

fig, ax = plt.subplots(nrows=1, ncols=1)

for isk, sigma_k in enumerate([10, 6, 4, 1]):
    deltas = []
    deltas_approx = []
    for ec in ecut_ev:
        ecut_ha = ec / Hartree
        ke = prefactor * kinetic_energy(ecut_ha, box_length, sigma_k)
        ke_int = prefactor * sigma_k**2 / 2
        deltas.append(np.abs(ke - ke_int))
        # The factor of 1/2 is because of how we're using ecut, we only sum from
        # [-nmax/2, nmax/2] which means we should inegrate from [-kmax/2,
        # kmax/2]. (kmax = (2Ecut)^{1/2}) is a spherical cutoff.
        kcut = 0.5 * (2 * ecut_ha) ** 0.5
        ke_err_apprx = prefactor * estimate_error_kinetic_energy(kcut, sigma_k)
        deltas_approx.append(ke_err_apprx)
        # if sigma_k == 10.0:
        #     print(sigma_k, ke, ke_int, ke-ke_int)
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
plt.ylim([1e-11, 1])

plt.savefig("cutoff_convergence_1d.pdf", bbox_inches="tight", dpi=300)
plt.savefig("cutoff_convergence_1d.png", bbox_inches="tight", dpi=200)
