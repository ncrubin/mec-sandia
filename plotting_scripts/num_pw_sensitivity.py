import os
import numpy as np
import scipy
from mec_sandia.vasp_utils import read_vasp
from mec_sandia.config import VASP_DATA
from mec_sandia.ft_pw_with_projectile import pw_qubitization_with_projectile_costs_from_v4

from ase.units import Bohr
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']

def linear(x, a, c):
    return a * x + c


def fit_linear(x, y, last_n_points=None):
    if last_n_points is None:
        last_n_points = len(x)
    if last_n_points > len(x):
        return None
    # log_x = np.log(x[-last_n_points:])
    # log_y = np.log(y[-last_n_points:])
    try:
        popt, pcov = scipy.optimize.curve_fit(linear, x, y)
        return popt
    except np.linalg.LinAlgError:
        return None

def time_evolution_costs():
    print("_________________________Helium + Hydrogen__________________")
    ase_cell = read_vasp(os.path.join(VASP_DATA, "H_2eV_POSCAR"))
    volume_ang = ase_cell.get_volume()
    print("Volume = {} A^3".format(volume_ang))
    
    # To compute rs parameter we need volume in Bohr
    volume_bohr = volume_ang / Bohr**3
    # and the number of valence electrons
    num_atoms = len(ase_cell.get_atomic_numbers()) 
    # There is 1 hydrogen atom in the cell. Is this just a proton?
    num_elec = num_atoms + 2
    from mec_sandia.vasp_utils import compute_wigner_seitz_radius
    # Get the Wigner-Seitz radius
    rs = compute_wigner_seitz_radius(volume_bohr, num_elec)
    print("rs = {} a0".format(rs))
    print("eta = {} ".format(num_elec))
    L_bohr = volume_bohr**(1.0/3.0)
    print("L = {} a0".format(L_bohr))
    print("Volume Bohr = {} boh4".format(L_bohr**3))

    num_bits_momenta = 6 # Number of bits in each direction for momenta
    eps_total = 1e-3 # Total allowable error
    num_bits_nu = 6 # extra bits for nu 
    num_bits_nuc = 8 # extra bits for (quantum?) nuclear positions 
    num_nuclei = len(ase_cell.get_atomic_numbers()) - 1 # minus one for the projectile
    projectile_mass = 1836 * 4
    projectile_charge = 2

    # calculate projectile wavenumer
    projectile_velocity = 4
    projectile_ke = 0.5 * projectile_mass * projectile_velocity**2
    projectile_wavenumber_au = np.sqrt(2 * projectile_ke / projectile_mass) * projectile_mass # p = m * v

    lambda_vals = []
    block_encoding_vals = []
    np_vals = [4, 5, 6, 7, 8, 9] 

    p_max_vals = [2**(np_val - 1) for np_val in np_vals]
    kp_vals = [2 * np.pi * p_val / volume_bohr**(1/3) for p_val in p_max_vals]
    ke_vals = [0.5 * kp_val**2 * 27.21138602 for kp_val in kp_vals]
    print(ke_vals)
    exit(0)

    for nbm in np_vals: 
        blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v4(
            np=nbm, 
            nn=8,
            eta=num_elec, 
            Omega=volume_bohr, 
            eps=eps_total, 
            nMc=num_bits_nu,
            nbr=20, 
            L=num_nuclei, 
            zeta=projectile_charge,
            mpr=projectile_mass,
            kmean=projectile_wavenumber_au,
            phase_estimation_costs=False
        )
        print(f"Block encdoing costs: Toffolis = {blockencodingtoff:4.3e}, lambda = {lambdaval:f} qubits = {qubit}")
        lambda_vals.append(lambdaval)
        block_encoding_vals.append(blockencodingtoff)

    params = fit_linear(np_vals, block_encoding_vals)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np_vals, block_encoding_vals, color=colors[0], linestyle='-', label=f"Toffolis for block encoding", marker='o', markersize=8 ) 
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("Num. bits per spatial dimension", fontsize=14)
    ax.set_ylabel("Num. Toffolis", fontsize=14, color=colors[0])
    ax.yaxis.label.set_color(colors[0])
    ax.tick_params(axis='y', colors=colors[0])
    ax.legend(loc='upper left', frameon=False)

    ax2 = ax.twinx()
    ax2.plot(np_vals, lambda_vals, color=colors[1], linestyle='-', label="$\lambda$", marker='o', markersize=8 ) 
    ax2.tick_params(which='both', labelsize=14, direction='in')
    ax2.yaxis.label.set_color(colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.legend(loc='lower right', frameon=False)



    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("num_pw_H2_sensitivity.png", format="PNG", dpi=300)
    plt.savefig("num_pw_H2_sensitivity.pdf", format="PDF", dpi=300)


if __name__ == "__main__":
    time_evolution_costs()