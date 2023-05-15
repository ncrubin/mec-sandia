import os
import numpy as np
from mec_sandia.vasp_utils import read_vasp
from mec_sandia.config import VASP_DATA
from mec_sandia.ft_pw_with_projectile import pw_qubitization_with_projectile_costs_from_v4

from ase.units import Bohr
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']

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
    
    blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v4(
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
        phase_estimation_costs=False
    )
    print(f"Block encdoing costs: Toffolis = {blockencodingtoff:4.3e}, lambda = {lambdaval:f} qubits = {qubit}")

    eps_total_vals = np.logspace(-2, -4.4, 8)
    tau_vals = [10, 20, 30, 40][::-1]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for cidx, tau in enumerate(tau_vals):
            evolution_costs = []
            for eps in eps_total_vals:
                blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v4(
                    np=num_bits_momenta, 
                    nn=num_bits_momenta + 2,
                    eta=num_elec, 
                    Omega=volume_bohr, 
                    eps=eps, 
                    nMc=num_bits_nu,
                    nbr=20, 
                    L=num_nuclei, 
                    zeta=projectile_charge,
                    mpr=projectile_mass,
                    kmean=projectile_wavenumber_au,
                    phase_estimation_costs=False
                )

                # Total toffoli times
                # 2(λt + 1.04(λt)⅓)log(1/ε)⅔
                lambda_by_time = np.abs(tau) * lambdaval
                num_queries_to_block_encoding = 2 * (lambda_by_time + 1.04 * (lambda_by_time)**(1/3)) * np.log2(1/eps)**(2/3)
                print("Total Time Evolution Toffoli = {: 4.3e}".format(num_queries_to_block_encoding * blockencodingtoff))
                print()
                evolution_costs.append(num_queries_to_block_encoding * blockencodingtoff)
        
            ax.semilogx(eps_total_vals, evolution_costs, color=colors[cidx], linestyle='-', label=fr"$t=${tau}", marker='o', markersize=8)
    
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$\epsilon$", fontsize=14)
    ax.set_ylabel(r"Toffolis [$10^{13}$]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper right', fontsize=14, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("H2_epsilon_vs_evolution_time.png", format="PNG", dpi=300)
    plt.savefig("H2_epsilon_vs_evolution_time.pdf", format="PDF", dpi=300)
    plt.show()



if __name__ == "__main__":
    time_evolution_costs()