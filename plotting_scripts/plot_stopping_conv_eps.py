import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from ase.units import Bohr, Hartree
from scipy.interpolate import CubicSpline
import scipy.optimize

from mec_sandia.gaussians import kinetic_variance_exact
from mec_sandia.stopping_power import parse_stopping_data
from mec_sandia.vasp_utils import compute_wigner_seitz_radius, read_vasp

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"

colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]


sim_res = []
act_res = []

qData2 = np.loadtxt(
    "../notebooks/AndrewsFirstGaussian/C_10gpcc_1eV_stopping_config1.txt"
)
velocity_au = qData2[:, 0]  # the velocities are already given in atomic units
stopping_au = qData2[:, 1] * (
    Bohr / Hartree
)  # the stopping powers are given in eV/A, so we multiply by the number of A/bohr and divide by the number of eV/Ha

stopping_spl = CubicSpline(velocity_au, stopping_au)
xs = np.linspace(velocity_au[0], velocity_au[-1], 100)
stopping_deriv_spl = stopping_spl.derivative(1)

stopping_err = 0.1  # eV/A
stopping_err_au = stopping_err * Bohr / Hartree
sigma_k = 4
mass_proj = 1836

ase_cell = read_vasp("../vasp_data/C_POSCAR")
# Next we can get some system paramters
volume_ang = ase_cell.get_volume()
print(f"Volume = {volume_ang} A^3")

# To compute rs parameter we need volume in Bohr

volume_bohr = volume_ang / Bohr**3
# and the number of valence electrons
num_carbon = len(np.where(ase_cell.get_atomic_numbers() == 6)[0])
# There is 1 hydrogen atom in the cell
num_elec = 1 + num_carbon * 4

# Get the Wigner-Seitz radius
rs = compute_wigner_seitz_radius(volume_bohr, num_elec)
print("rs = {} bohr".format(rs))

ecut_ha = 2000
box_length = volume_bohr ** (1.0 / 3.0)
print("box_length = {} bohr".format(box_length))


all_sims = []


def linear(x, a, b):
    return a * x + b


def fit_linear(x, y, sigma=None, absolute_sigma=False):
    try:
        popt, pcov = scipy.optimize.curve_fit(
            linear, x, y, sigma=sigma, absolute_sigma=absolute_sigma
        )
        return popt, pcov
    except np.linalg.LinAlgError:
        return None


def plot_figures(read_from_file=False):
    outdir = "figures/stopping_sampling_epsilon"
    # read_from_file = False
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # subsample involves randomly sampling the input
    np.random.seed(7)
    stopping_err = 0.1  # eV/A
    stopping_err_au = stopping_err * Bohr / Hartree
    sigma_k = 4
    mass_proj = 1836
    for vel in velocity_au[::2]:
        dft_data = parse_stopping_data(
            f"../notebooks/AndrewsFirstGaussian/{vel}_work_vs_dist",
            vel,
            mass_proj=mass_proj,
            num_points=10,
            rare_event=0.25,
            random_sub_sample=False,
        )
        kproj = dft_data.kproj[0]
        var = kinetic_variance_exact(kproj**2, sigma_k)
        sigma_ke = np.sqrt(var) / (2 * mass_proj)
        ke = kproj**2 / (2 * mass_proj)
        gen_s = lambda x: dft_data.stopping * x + ke
        print(f"vel = {vel}, ke = {ke}, sigma_ke = {sigma_ke}")
        xmax = dft_data.distance[-1]
        eps_t = []
        eps_s = []
        samples = [int(x) for x in np.logspace(0.5, 4, 10)]
        for ns in samples:
            xs = np.linspace(0, xmax, 10)
            # Generate 10 time/disatance points with some noise and average to get sigma_T
            ys = gen_s(xs)[:, None] + np.random.normal(0.0, sigma_ke, 10 * ns).reshape(
                (10, ns)
            )
            ys_mean = np.mean(ys, axis=1)
            ys_err = np.std(ys, axis=1, ddof=1) / np.sqrt(ns)
            popt, pcov = fit_linear(xs, ys_mean, ys_err, absolute_sigma=True)
            eps_t.append(np.mean(ys_err))
            eps_s.append(pcov[0, 0] ** 0.5)
        plt.plot(eps_t, eps_s, marker="o", label=rf"$v_\mathrm{{proj}}={vel}$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(stopping_err_au, color="grey", ls=":")
    plt.axhline(5 * stopping_err_au, color="grey", ls=":")
    plt.xlabel(r"$ϵ_T$ (Ha)", fontsize=14)
    plt.ylabel(r"$ϵ_S$ (Ha)", fontsize=14)
    plt.tick_params(which="both", labelsize=14, direction="in")
    plt.savefig(f"{outdir}/epsilon_TS.pdf", bbox_inches="tight", dpi=300)
    plt.cla()
    stopping_err = 0.1  # eV/A
    stopping_err_au = stopping_err * Bohr / Hartree
    sigma_k = 4
    mass_proj = 1836
    for vel in velocity_au[::2]:
        dft_data = parse_stopping_data(
            f"../notebooks/AndrewsFirstGaussian/{vel}_work_vs_dist",
            vel,
            mass_proj=mass_proj,
            num_points=10,
            rare_event=0.25,
            random_sub_sample=False,
        )
        kproj = dft_data.kproj[0]
        var = kinetic_variance_exact(kproj**2, sigma_k)
        sigma_ke = np.sqrt(var) / (2 * mass_proj)
        ke = kproj**2 / (2 * mass_proj)
        gen_s = lambda x: dft_data.stopping * x + ke
        print(f"vel = {vel}, ke = {ke}, sigma_ke = {sigma_ke}")
        xmax = dft_data.distance[-1]
        eps_t = []
        eps_s = []
        samples = [int(x) for x in np.logspace(0.5, 4, 10)]
        for ns in samples:
            xs = np.linspace(0, xmax, 10)
            # Generate 10 points with some noise and average to get sigma_T
            ys = gen_s(xs)[:, None] + np.random.normal(0.0, sigma_ke, 10 * ns).reshape(
                (10, ns)
            )
            ys_mean = np.mean(ys, axis=1)
            ys_err = np.std(ys, axis=1, ddof=1) / np.sqrt(ns)
            # if ns == 35:
            #     plt.errorbar(xs, ys_mean, yerr=ys_err, fmt='o')
            popt, pcov = fit_linear(xs, ys_mean, ys_err, absolute_sigma=True)
            eps_t.append(np.mean(ys_err))
            eps_s.append(pcov[0, 0] ** 0.5)
        plt.plot(samples, eps_t, marker="o", label=rf"$v_\mathrm{{proj}}={vel}$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(0.2, color="grey", ls=":")
    plt.axhline(0.8, color="grey", ls=":")
    plt.xlabel(r"$N_s$", fontsize=14)
    plt.xlabel(r"$ϵ_T$ (Ha)", fontsize=14)
    plt.tick_params(which="both", labelsize=14, direction="in")
    plt.savefig(f"{outdir}/epsilon_TNs.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    import os

    if not os.path.isdir("figures/stopping"):
        os.makedirs("figures/stopping")
    if len(sys.argv[1:]) == 0:
        read_files = False
    else:
        read_files = True
    plot_figures(read_files)
