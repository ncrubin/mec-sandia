import os
import numpy
import numpy as np
import scipy
import ase
from ase.io import read, write

from mec_sandia.config import VASP_DATA, REPO_DIRECTORY

from ase.units import Bohr

import os
import numpy as np
from mec_sandia.vasp_utils import read_vasp
from mec_sandia.config import VASP_DATA
from mec_sandia.ft_pw_with_projectile import pw_qubitization_with_projectile_costs_from_v5
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False


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
    

def scale_system_D(scale):
    fname = os.path.join(VASP_DATA, "D_POSCAR")
    os.chdir(os.path.join(REPO_DIRECTORY, 'vasp_data'))
    ase_atom = read_vasp(fname)

    ########################
    #
    # Calculate properties
    #
    #########################
    ase_cell = ase_atom


    cut_len = scale
    positions = ase_atom.get_scaled_positions()
    mask = [any(pos > cut_len) for pos in positions]
    new_positions = numpy.asarray(
        [pos for pos, mval in zip(ase_atom.get_positions(), mask) if not mval])
    new_numbers = [num for num, mval in zip(ase_atom.get_atomic_numbers(), mask) if not mval]
    new_cell = ase_atom.cell * cut_len
    new_atoms = ase.Atoms(numbers=new_numbers, positions=new_positions, cell=new_cell)
    return new_atoms

def main():

    total_toffoli_cost = []
    eta_vals = []
    qpe_cost = []
    for ss in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:    
        print("\n\n\n\n")
        ase_cell = scale_system_D(ss)

        volume_ang = ase_cell.get_volume()
        print("Volume = {} A^3".format(volume_ang))

        # To compute rs parameter we need volume in Bohr
        volume_bohr = volume_ang / Bohr**3
        # and the number of valence electrons
        num_atoms = len(ase_cell.get_atomic_numbers()) 
        # There is 1 hydrogen atom in the cell. Is this just a proton?
        num_elec = num_atoms + 1
        eta_vals.append(num_elec)
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
        projectile_mass = 1836 
        projectile_charge = 1

        # calculate projectile wavenumer
        projectile_velocity = 4
        projectile_ke = 0.5 * projectile_mass * projectile_velocity**2
        projectile_wavenumber_au = np.sqrt(2 * projectile_ke / projectile_mass) * projectile_mass # p = m * v
    
        blockencodingtoff, lambdaval, qubit, cost_dataclass = pw_qubitization_with_projectile_costs_from_v5(
            np=num_bits_momenta, 
            nn=num_bits_momenta + 2,
            eta=num_elec, 
            Omega=volume_bohr, 
            eps=eps_total, 
            nMc=num_bits_nu,
            nbr=20, 
            L=num_nuclei, 
            zeta=projectile_charge,
            mpr=projectile_mass,
            kmean=projectile_wavenumber_au,
            phase_estimation_costs=False,
            return_subcosts=True
        )
        print(f"Block encdoing costs: Toffolis = {blockencodingtoff:4.3e}, lambda = {lambdaval:f} qubits = {qubit}")
        total_toffoli_cost.append((2 * (lambdaval + 1.04 * (lambdaval)**(1/3)) * np.log2(1/1.0E-3)**(2/3))*blockencodingtoff)

        qpe_toff, qpe_qubits = pw_qubitization_with_projectile_costs_from_v5(
            np=num_bits_momenta, 
            nn=num_bits_momenta + 2,
            eta=num_elec, 
            Omega=volume_bohr, 
            eps=eps_total, 
            nMc=num_bits_nu,
            nbr=20, 
            L=num_nuclei, 
            zeta=projectile_charge,
            mpr=projectile_mass,
            kmean=projectile_wavenumber_au,
            phase_estimation_costs=True,
            return_subcosts=False
        )
        qpe_cost.append(qpe_toff)


        print("tofc_inequality_c1",             cost_dataclass.tofc_inequality_c1)
        print("tofc_superposition_ij_c2",       cost_dataclass.tofc_superposition_ij_c2)
        print("tofc_superposition_wrs_c3",               cost_dataclass.tofc_superposition_wrs_c3)
        print("tofc_controlled_swaps_c4",                cost_dataclass.tofc_controlled_swaps_c4)
        print("tofc_extra_nuclear_momentum_c5",        cost_dataclass.tofc_extra_nuclear_momentum_c5)
        print("tofc_nested_boxes_c6",                    cost_dataclass.tofc_nested_boxes_c6)
        print("tofc_prep_unprep_nuclear_via_qrom_c7",    cost_dataclass.tofc_prep_unprep_nuclear_via_qrom_c7)
        print("tofc_add_subtract_momentum_for_select_c8", cost_dataclass.tofc_add_subtract_momentum_for_select_c8)
        print("tofc_phasing_by_structure_factor_c9",     cost_dataclass.tofc_phasing_by_structure_factor_c9)
        print("tofc_reflection_costs_cr",                cost_dataclass.tofc_reflection_costs_cr)

    log_eta = numpy.log(eta_vals)
    log_tof_costs = numpy.log(total_toffoli_cost)
    log_qpe_costs = numpy.log(qpe_cost)

    print(log_eta)
    print(log_tof_costs)
    params = fit_linear(log_eta, log_tof_costs)
    print(params)
    x_min = eta_vals[0]
    x_max = eta_vals[-1]
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.exp(params[1]) * x_vals**params[0]

    params2 = fit_linear(log_eta, log_qpe_costs)
    y_vals2 = np.exp(params2[1]) * x_vals**params2[0]


    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.loglog(eta_vals, total_toffoli_cost, color=colors[0], markersize=8, marker='o', label=r"Time evolution $\mathcal{{O}}(\eta^{{{:2.2f}}})$".format(params[0]))
    ax.loglog(x_vals, y_vals, '--', color=colors[0])

    ax.loglog(eta_vals, qpe_cost, color=colors[1], markersize=8, marker='s', label="QPE$(\\epsilon=10^{{-3}})\;\;\mathcal{{O}}(\eta^{{{:2.2f}}})$".format(params2[0]))
    ax.loglog(x_vals, y_vals2, '--', color=colors[1])

    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$\eta$", fontsize=14)
    ax.set_ylabel(r"Toffolis", fontsize=14)
    ax.set_title("$r_{s} \\approx 0.8, \;\;n_{p} = 6$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    plt.savefig("H_D_eta_vs_toff_fixed_rs.png", format="PNG", dpi=300)
    plt.savefig("H_D_eta_vs_toff_fixed_rs.pdf", format="PDF", dpi=300)



if __name__ == "__main__":
    main() 