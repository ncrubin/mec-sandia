"""
The idea is to get evolution time costs for both systems
"""
import os
import numpy as np
from mec_sandia.vasp_utils import read_vasp
from mec_sandia.config import VASP_DATA
from mec_sandia.ft_pw_resource_estimates import pw_qubitization_costs 
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
def hydrogen_evolution_time_costs():
    # Deuterium
    ase_cell = read_vasp(os.path.join(VASP_DATA, "D_POSCAR"))
    # Next we can get some system paramters
    volume_ang = ase_cell.get_volume()
    print("Volume = {} A^3".format(volume_ang))
    
    # To compute rs parameter we need volume in Bohr
    from ase.units import Bohr
    volume_bohr = volume_ang / Bohr**3
    # and the number of valence electrons
    num_atoms = len(ase_cell.get_atomic_numbers())
    # There is 1 hydrogen atom in the cell. Is this just a proton?
    num_elec = num_atoms 
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
    num_bits_nuc = 6 # extra bits for (quantum?) nuclear positions 
    num_nuclei = len(ase_cell.get_atomic_numbers()) # Number of (quantum?) nuclei
    toff, qubit, finallambda, qpe_lam, eps_ph = pw_qubitization_costs(np=num_bits_momenta, eta=num_elec, Omega=volume_bohr, eps=eps_total, nMc=5, nbr=5, L=num_nuclei)
    print(f"Toffolis = {toff:4.3e}, qubits = {qubit}, lambda={finallambda}")

    # Total toffoli times
    # 2(λt + 1.04(λt)⅓)log(1/ε)⅔
    tau = 40 # evolution time
    lambda_by_time = np.abs(tau) * finallambda
    total_toffoli = 2 * (lambda_by_time + 1.04 * (lambda_by_time)**(1/3)) * np.log2(1/eps_ph)**(2/3)
    print("input eps ", eps_total)
    print("eps_ph ", eps_ph)
    print("Total Num planewaves = {: 4.3e}".format((2**num_bits_momenta - 1)**3)) # 250K planewaves
    print("Total Time Evolution Toffoli = {: 4.3e}".format(total_toffoli))
    print("Total Phase Estimation Toffoli = {: 4.3e}".format(toff * np.ceil(np.pi * finallambda / (2 * eps_ph))))
    print("Total Phase Estimation Toffoli Adjusted for success prob = {: 4.3e}".format(toff * qpe_lam))
    print("Logical Qubits ", qubit)


    eps_total = np.logspace(-2, -4.4, 8) # how much fidelity do I need?
    num_bits_momenta = 6 # Number of bits in each direction for momenta
    num_bits_nu = 6 # extra bits for nu 
    num_bits_nuc = 6 # extra bits for (quantum?) nuclear positions 
    num_nuclei = len(ase_cell.get_atomic_numbers()) # Number of (quantum?) nuclei
    tau = 40 # evolution time


    fig, ax = plt.subplots(nrows=1, ncols=1)

    for cidx, tau in enumerate([10, 20, 30, 40][::-1]):
        evolution_costs = []
        for eps in eps_total:
            toff, qubit, finallambda, qpe_lam, eps_ph = pw_qubitization_costs(np=num_bits_momenta, eta=num_elec, Omega=volume_bohr, eps=eps, nMc=5, nbr=5, L=num_nuclei)
            print(f"epsilon = {eps: 4.4e}, Toffolis = {toff:4.3e}, qubits = {qubit}, lambda={finallambda}")

            # Total toffoli times
            # 2(λt + 1.04(λt)⅓)log(1/ε)⅔
            lambda_by_time = np.abs(tau) * finallambda
            total_toffoli = 2 * (lambda_by_time + 1.04 * (lambda_by_time)**(1/3)) * np.log2(1/eps_ph)**(2/3)
            print("Total Time Evolution Toffoli = {: 4.3e}".format(total_toffoli))
            print("Total Phase Estimation Toffoli = {: 4.3e}".format(toff * np.ceil(np.pi * finallambda / (2 * eps_ph))))
            print()
            evolution_costs.append(total_toffoli)
    
        ax.semilogx(eps_total, evolution_costs, color=colors[cidx], linestyle='-', label=fr"$t=${tau}", marker='o', markersize=8)

    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$\epsilon$", fontsize=14)
    ax.set_ylabel(r"Toffolis [$10^{10}$]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper right', fontsize=14, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("deuterium_epsilon_vs_evolution_time.png", format="PNG", dpi=300)
    plt.savefig("deuterium_epsilon_vs_evolution_time.pdf", format="PDF", dpi=300)




if __name__ == "__main__":
    hydrogen_evolution_time_costs()