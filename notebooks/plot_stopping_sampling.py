from ase.units import Bohr, Hartree
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import CubicSpline
import sys

from mec_sandia.stopping_power import (
    compute_stopping_power,
    parse_stopping_data,
    compute_stopping_exact,
    StoppingPowerData
)
from mec_sandia.vasp_utils import read_vasp

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"

colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]


sim_res = []
act_res = []

qData2 = np.loadtxt("AndrewsFirstGaussian/C_10gpcc_1eV_stopping_config1.txt")
velocity_au = qData2[:, 0]  # the velocities are already given in atomic units
stopping_au = qData2[:, 1] * (
    Bohr / Hartree
)  # the stopping powers are given in eV/A, so we multiply by the number of A/bohr and divide by the number of eV/Ha

stopping_spl = CubicSpline(velocity_au, stopping_au)
xs = np.linspace(velocity_au[0], velocity_au[-1], 100)
stopping_deriv_spl = stopping_spl.derivative(1)

stopping_err = 0.1  # eV/A
stopping_err_au = stopping_err * Bohr / Hartree
sigma_k = 10
mass_proj = 1836

ase_cell = read_vasp(f"../vasp_data/C_POSCAR")
# Next we can get some system paramters
volume_ang = ase_cell.get_volume()
print("Volume = {} A^3".format(volume_ang))

# To compute rs parameter we need volume in Bohr
from ase.units import Bohr

volume_bohr = volume_ang / Bohr**3
# and the number of valence electrons
num_carbon = len(np.where(ase_cell.get_atomic_numbers() == 6)[0])
# There is 1 hydrogen atom in the cell
num_elec = 1 + num_carbon * 4
from mec_sandia.vasp_utils import compute_wigner_seitz_radius

# Get the Wigner-Seitz radius
rs = compute_wigner_seitz_radius(volume_bohr, num_elec)
print("rs = {} bohr".format(rs))

ecut_ha = 2000
box_length = volume_bohr ** (1.0 / 3.0)
print("box_length = {} bohr".format(box_length))


def plot_figures(read_from_file=False):
    outdir = "stopping_sampling_data"
    #read_from_file = False 
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # subsample involves randomly sampling the input
    np.random.seed(7)
    for isamp, num_samples in enumerate([1_000, 10_000, 50_000]):
        sim_res = []
        act_res = []
        for vel in velocity_au:
            dft_data = parse_stopping_data(
                f"AndrewsFirstGaussian/{vel}_work_vs_dist",
                vel,
                mass_proj=mass_proj,
                num_points=20,
            )
            stopping_deriv = np.abs(stopping_deriv_spl(vel))
            kproj_vals = np.array(
                [np.array([kx, 0, 0]) for kx in dft_data.kproj_sub_sample]
            )
            if read_from_file:
                filename = f"{outdir}/stopping_sampling_{vel}_{num_samples}.json"
                print(f"Reading stopping data from {filename}.")
                stopping_data = StoppingPowerData.from_file(filename)
            else:
                print("Generating stopping data from scratch.")
                print(f"vel = {vel}, ns = {num_samples}")
                stopping_data = compute_stopping_power(
                    ecut_ha,
                    box_length,
                    sigma_k,
                    dft_data.time_sub_sample,
                    kproj_vals,
                    stopping_deriv,
                    mass_proj,
                    num_samples=num_samples,
                )
                filename = f"{outdir}/stopping_sampling_{vel}_{num_samples}.json"
                stopping_data.to_file(filename)
            sim_res.append(stopping_data)
            act_res.append(abs(stopping_data.stopping_expected))
        vals = [abs(s.stopping) for s in sim_res]
        errs = [s.stopping_err for s in sim_res]
        plt.errorbar(
            velocity_au,
            np.array(vals) - np.array(act_res),
            yerr=errs,
            fmt="o",
            color=colors[isamp],
            label=r"$N_s$ = {:d}".format(num_samples),
        )
    plt.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
    plt.axhline(stopping_err_au, color="grey", alpha=0.5)
    plt.axhline(-stopping_err_au, color="grey", alpha=0.5)
    plt.ylim([-10*stopping_err_au, 10*stopping_err_au])
    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.xlabel("initial veloctiy [au]", fontsize=14)
    plt.ylabel("Stopping Power error [au]", fontsize=14)
    plt.savefig("stopping_power_conv.pdf", bbox_inches="tight", dpi=300)


    # Plot ns = 10_000
    plt.cla()
    plt.plot(velocity_au, act_res, label="expected", lw=0, marker="D", color=colors[0])
    vals = [abs(s.stopping) for s in sim_res]
    errs = [s.stopping_err for s in sim_res]
    plt.errorbar(velocity_au, vals, yerr=errs, fmt="o", label="sampling", color=colors[1], markerfacecolor="None")
    xs = np.linspace(velocity_au[0], velocity_au[-1], 100)
    plt.plot(xs, stopping_spl(xs), color=colors[2], label="DFT Data")
    plt.legend()
    plt.xlabel("Velocity [au]", fontsize=14)
    plt.ylabel("Stopping Power [au]", fontsize=14)
    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.savefig("stopping_power_comparison.pdf", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    if len(sys.argv[1:]) == 0:
        read_files = False
    else:
        read_files = True 
    plot_figures(read_files)