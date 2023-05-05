# Let's read in the Carbon example provided by Sandia
import numpy as np
from mec_sandia.vasp_utils import read_vasp
import matplotlib.pyplot as plt
import scipy.optimize


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"

colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]
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
sim_res = []
act_res = []
qData2 = np.loadtxt("AndrewsFirstGaussian/C_10gpcc_1eV_stopping_config1.txt")
velocity_au = qData2[:, 0]  # the velocities are already given in atomic units

from mec_sandia.stopping_power import (
    parse_stopping_data,
    compute_stopping_power,
    _fit_linear,
)
from ase.units import Bohr, Hartree

stopping_err = 0.1  # eV/A
stopping_err_au = stopping_err * Bohr / Hartree
sigma_k = 10
# Get DFT data
qData1 = np.loadtxt("AndrewsFirstGaussian/4.0_work_vs_dist")
position_au = qData1[:, 0] / (
    0.529
)  # divide the positions (angstroms) by the number of angstroms per atomic unit
time_au = (
    position_au / 4.0
)  # divide the positions in atomic units by the velocity in atomic units to get the time in atomic units
work_au = (
    qData1[:, 1] / 27.2
)  # divide the work in eV by the number of eV per atomic unit of energy

# load the full stopping curve
qData2 = np.loadtxt("AndrewsFirstGaussian/C_10gpcc_1eV_stopping_config1.txt")
velocity_au = qData2[:, 0]  # the velocities are already given in atomic units
stopping_au = qData2[:, 1] * (
    0.529 / 27.2
)  # the stopping powers are given in eV/A, so we multiply by the number of A/bohr and divide by the number of eV/Ha
# plt.plot(velocity_au, stopping_au, marker="o", lw=0)
from scipy.interpolate import CubicSpline

stopping_spl = CubicSpline(velocity_au, stopping_au)
xs = np.linspace(velocity_au[0], velocity_au[-1], 100)
stopping_deriv_spl = stopping_spl.derivative(1)

mass_proj = 1838
for sigma_k in [1, 4, 6, 10]:
    if sigma_k < 10:
        ecut_ha = 2000 / Hartree
    else:
        ecut_ha = 10000 / Hartree
    box_length = volume_bohr ** (1.0 / 3.0)
    print("box_length = {} bohr".format(box_length))
    vel = 2.0
    fig, ax = plt.subplots(6, 1, figsize=(15, 15), sharex=True)
    stopping_deriv = abs(stopping_deriv_spl(vel))
    num_mc_samples = [5, 10, 50, 100, 500, 1000]
    for itime, max_time in enumerate([5, 10, 20, 30, 40, 50]):
        for num_pts in [5, 10, 20]:
            dft_data = parse_stopping_data(
                f"AndrewsFirstGaussian/{vel}_work_vs_dist",
                vel,
                mass_proj=mass_proj,
                num_points=num_pts,
                rare_event=0.25,
                random_sub_sample=False,
                max_time=max_time,
            )
            kproj_vals = np.array(
                [np.array([kx, 0, 0]) for kx in dft_data.kproj_sub_sample]
            )
            sim_res = []
            act_res = []
            for num_samples in num_mc_samples:
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
                sim_res.append(stopping_data)
            vals = [abs(s.stopping) for s in sim_res]
            errs = [s.stopping_err for s in sim_res]
            # Plot samples along x, multiple points.
            ax[itime].axhline(stopping_spl(vel))
            ax[itime].errorbar(
                num_mc_samples,
                np.array(vals),
                yerr=errs,
                fmt="o",
                label=f"num points = {num_pts}",
            )
        ax[itime].set_title(f"Max Time = {max_time} [au]")
        ax[itime].set_ylim([stopping_spl(vel)-0.1, stopping_spl(vel)+0.1])
        ax[itime].set_xscale("log")
        ax[itime].tick_params(which='both', labelsize=14, direction='in')
        if itime == 0:
            ax[0].legend(fontsize=14)
    fig.suptitle(f"$\sigma_k = {sigma_k}$", fontsize=14)
    fig.supxlabel("Number of Samples", fontsize=14)
    fig.supylabel("Stopping Power at $v=2$ au", fontsize=14)
    plt.savefig(f"figures/stopping_params_{sigma_k}.png", dpi=200)
    plt.savefig(f"figures/stopping_params_{sigma_k}.pdf", dpi=200)