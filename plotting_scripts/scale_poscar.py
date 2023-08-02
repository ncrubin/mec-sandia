import os
import numpy
import numpy as np
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

def main():
    fname = os.path.join(VASP_DATA, "H_2eV_POSCAR")
    os.chdir(os.path.join(REPO_DIRECTORY, 'vasp_data'))
    ase_atom = read(fname)

    ########################
    #
    # Calculate properties
    #
    #########################
    ase_cell = ase_atom

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

    Lvals = np.linspace(L_bohr/2, L_bohr, 5)
    grid_spacing = np.array([2**5 - 1, 
                             2**6 - 1,
                             2**6 - 1,
                             2**6 - 1,
                             2**6 - 1
                            ])
    print(Lvals)
    print(Lvals / grid_spacing)

    ########################
    #
    # Shrink cell
    #
    #########################
    cut_len = 0.25
    positions = ase_atom.get_scaled_positions()
    mask = [any(pos > cut_len) for pos in positions]
    new_positions = numpy.asarray(
        [pos for pos, mval in zip(ase_atom.get_positions(), mask) if not mval])
    new_numbers = [num for num, mval in zip(ase_atom.get_atomic_numbers(), mask) if not mval]
    if not numpy.any(new_numbers == 2):
        new_numbers[0] = 2

    new_cell = ase_atom.cell * cut_len
    new_atoms = ase.Atoms(numbers=new_numbers, positions=new_positions, cell=new_cell)
    write('H2_2eV_POSCAR_95_percent.vasp', new_atoms, format='vasp', direct=True)
    exit()

    print('\n\n\n')
    ########################
    #
    # Calculate properties
    #
    #########################
    ase_cell = new_atoms 

    volume_ang = ase_cell.get_volume()
    print("Volume = {} A^3".format(volume_ang))
    print("Rescaled old volume = {} A^3 ".format(ase_atom.get_volume()* 0.75**3))
 
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
    

    num_bits_momenta = 5 # Number of bits in each direction for momenta
    eps_total = 1e-3 # Total allowable error
    num_bits_nu = 8 # extra bits for nu 
    num_bits_nuc = 8 # extra bits for (quantum?) nuclear positions 
    num_nuclei = len(ase_cell.get_atomic_numbers()) - 1 # minus one for the projectile
    projectile_mass = 1836 * 4
    projectile_charge = 2

    # calculate projectile wavenumer
    projectile_velocity = 4
    projectile_ke = 0.5 * projectile_mass * projectile_velocity**2
    projectile_wavenumber_au = np.sqrt(2 * projectile_ke / projectile_mass) * projectile_mass # p = m * v
    
    blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v5(
        np=num_bits_momenta, 
        nn=num_bits_momenta,
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
                blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v5(
                    np=num_bits_momenta, 
                    nn=num_bits_momenta,
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
    ax.set_ylabel(r"Toffolis [$10^{12}$]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper right', fontsize=14, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("H2_75p_epsilon_vs_evolution_time.png", format="PNG", dpi=300)
    plt.savefig("H2_75p_epsilon_vs_evolution_time.pdf", format="PDF", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()