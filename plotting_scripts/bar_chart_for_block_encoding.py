import os
import numpy as np
from mec_sandia.vasp_utils import read_vasp
from mec_sandia.config import VASP_DATA
from mec_sandia.ft_pw_with_projectile import pw_qubitization_with_projectile_costs_from_v5

from ase.units import Bohr
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']

def block_encoding_costs():
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
    
    blockencodingtoff, lambdaval, qubit, cost_dataclass_He = pw_qubitization_with_projectile_costs_from_v5(
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

    print("tofc_inequality_c1",                      cost_dataclass_He.tofc_inequality_c1)
    print("tofc_superposition_ij_c2",                cost_dataclass_He.tofc_superposition_ij_c2)
    print("tofc_superposition_wrs_c3",               cost_dataclass_He.tofc_superposition_wrs_c3)
    print("tofc_controlled_swaps_c4",                cost_dataclass_He.tofc_controlled_swaps_c4)
    print("tofc_extra_nuclear_momentum_c5",          cost_dataclass_He.tofc_extra_nuclear_momentum_c5)
    print("tofc_nested_boxes_c6",                    cost_dataclass_He.tofc_nested_boxes_c6)
    print("tofc_prep_unprep_nuclear_via_qrom_c7",    cost_dataclass_He.tofc_prep_unprep_nuclear_via_qrom_c7)
    print("tofc_add_subtract_momentum_for_select_c8",cost_dataclass_He.tofc_add_subtract_momentum_for_select_c8)
    print("tofc_phasing_by_structure_factor_c9",     cost_dataclass_He.tofc_phasing_by_structure_factor_c9)
    print("tofc_reflection_costs_cr",                cost_dataclass_He.tofc_reflection_costs_cr)

    print()
    print()
    for key, val in vars(cost_dataclass_He).items():
        if 'lambda' in key:
            if 'lambda_T' == key:
                print(key,"\t", "{:10.10f}".format(val), ((6 * num_elec * np.pi**2) / volume_bohr**(2/3)) * (4**(num_bits_momenta - 1)))
            elif 'lambda_Tn' == key:
                print(key,"\t", "{:10.10f}".format(val), 6. /projectile_mass * np.pi**2 / volume_bohr**(2/3) * (4**(num_bits_momenta + 2 - 1)))
            elif 'lambda_Tkmean' == key:
                print(key,"\t", "{:10.10f}".format(val), 6. /projectile_mass * np.pi**2 / volume_bohr**(2/3) * (4**(num_bits_momenta + 2 - 1)))
            else:
                print(key,"\t", "{:10.10f}".format(val))



    print("\n\n\n\n_________________________Hydrogen + Deuterium__________________")
    ase_cell = read_vasp(os.path.join(VASP_DATA, "D_POSCAR"))
    volume_ang = ase_cell.get_volume()
    print("Volume = {} A^3".format(volume_ang))
    
    # To compute rs parameter we need volume in Bohr
    volume_bohr = volume_ang / Bohr**3
    # and the number of valence electrons
    num_atoms = len(ase_cell.get_atomic_numbers()) 
    # There is 1 hydrogen atom in the cell. Is this just a proton?
    num_elec = num_atoms + 1
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
    
    blockencodingtoff, lambdaval, qubit, cost_dataclass_hd = pw_qubitization_with_projectile_costs_from_v5(
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

    print("tofc_inequality_c1",                      cost_dataclass_hd.tofc_inequality_c1)
    print("tofc_superposition_ij_c2",                cost_dataclass_hd.tofc_superposition_ij_c2)
    print("tofc_superposition_wrs_c3",               cost_dataclass_hd.tofc_superposition_wrs_c3)
    print("tofc_controlled_swaps_c4",                cost_dataclass_hd.tofc_controlled_swaps_c4)
    print("tofc_extra_nuclear_momentum_c5",          cost_dataclass_hd.tofc_extra_nuclear_momentum_c5)
    print("tofc_nested_boxes_c6",                    cost_dataclass_hd.tofc_nested_boxes_c6)
    print("tofc_prep_unprep_nuclear_via_qrom_c7",    cost_dataclass_hd.tofc_prep_unprep_nuclear_via_qrom_c7)
    print("tofc_add_subtract_momentum_for_select_c8",cost_dataclass_hd.tofc_add_subtract_momentum_for_select_c8)
    print("tofc_phasing_by_structure_factor_c9",     cost_dataclass_hd.tofc_phasing_by_structure_factor_c9)
    print("tofc_reflection_costs_cr",                cost_dataclass_hd.tofc_reflection_costs_cr)

    print()
    print()
    for key, val in vars(cost_dataclass_hd).items():
        if 'lambda' in key:
            if 'lambda_T' == key:
                print(key,"\t", "{:10.10f}".format(val), ((6 * num_elec * np.pi**2) / volume_bohr**(2/3)) * (4**(num_bits_momenta - 1)))
            elif 'lambda_Tn' == key:
                print(key,"\t", "{:10.10f}".format(val), 6. /projectile_mass * np.pi**2 / volume_bohr**(2/3) * (4**(num_bits_momenta + 2 - 1)))
            elif 'lambda_Tkmean' == key:
                print(key,"\t", "{:10.10f}".format(val), 6. /projectile_mass * np.pi**2 / volume_bohr**(2/3) * (4**(num_bits_momenta + 2 - 1)))
            else:
                print(key,"\t", "{:10.10f}".format(val))




    print("\n\n\n\n_________________________Hydrogen + CARBON __________________")
    ase_cell = read_vasp(os.path.join(VASP_DATA, "C_POSCAR_cubic.vasp"))
    volume_ang = ase_cell.get_volume()
    print("Volume = {} A^3".format(volume_ang))
    
    # To compute rs parameter we need volume in Bohr
    volume_bohr = volume_ang / Bohr**3
    # and the number of valence electrons
    num_atoms = len(ase_cell.get_atomic_numbers())  - 1 # -1 for hydrogen atom
    # There is 1 hydrogen atom in the cell. Is this just a proton?
    num_elec = sum(ase_cell.get_atomic_numbers())

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
    num_bits_nu = 6 # extra bits for nu 
    num_bits_nuc = 8 # extra bits for (quantum?) nuclear positions 
    num_nuclei = len(ase_cell.get_atomic_numbers()) - 1 # minus one for the projectile
    projectile_mass = 1836 
    projectile_charge = 1

    # calculate projectile wavenumer
    projectile_velocity = 4
    projectile_ke = 0.5 * projectile_mass * projectile_velocity**2
    projectile_wavenumber_au = np.sqrt(2 * projectile_ke / projectile_mass) * projectile_mass # p = m * v
    
    blockencodingtoff, lambdaval, qubit, cost_dataclass_C = pw_qubitization_with_projectile_costs_from_v5(
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

    print("tofc_inequality_c1",                      cost_dataclass_C.tofc_inequality_c1)
    print("tofc_superposition_ij_c2",                cost_dataclass_C.tofc_superposition_ij_c2)
    print("tofc_superposition_wrs_c3",               cost_dataclass_C.tofc_superposition_wrs_c3)
    print("tofc_controlled_swaps_c4",                cost_dataclass_C.tofc_controlled_swaps_c4)
    print("tofc_extra_nuclear_momentum_c5",          cost_dataclass_C.tofc_extra_nuclear_momentum_c5)
    print("tofc_nested_boxes_c6",                    cost_dataclass_C.tofc_nested_boxes_c6)
    print("tofc_prep_unprep_nuclear_via_qrom_c7",    cost_dataclass_C.tofc_prep_unprep_nuclear_via_qrom_c7)
    print("tofc_add_subtract_momentum_for_select_c8",cost_dataclass_C.tofc_add_subtract_momentum_for_select_c8)
    print("tofc_phasing_by_structure_factor_c9",     cost_dataclass_C.tofc_phasing_by_structure_factor_c9)
    print("tofc_reflection_costs_cr",                cost_dataclass_C.tofc_reflection_costs_cr)

    print()
    print()
    for key, val in vars(cost_dataclass_C).items():
        if 'lambda' in key:
            if 'lambda_T' == key:
                print(key,"\t", "{:10.10f}".format(val), ((6 * num_elec * np.pi**2) / volume_bohr**(2/3)) * (4**(num_bits_momenta - 1)))
            elif 'lambda_Tn' == key:
                print(key,"\t", "{:10.10f}".format(val), 6. /projectile_mass * np.pi**2 / volume_bohr**(2/3) * (4**(num_bits_momenta + 2 - 1)))
            elif 'lambda_Tkmean' == key:
                print(key,"\t", "{:10.10f}".format(val), 6. /projectile_mass * np.pi**2 / volume_bohr**(2/3) * (4**(num_bits_momenta + 2 - 1)))
            else:
                print(key,"\t", "{:10.10f}".format(val))

    ##### Create DF
    import pandas as pd
    df_dict = {"Toffoli Count": [], "System": [], "Subroutine": []}
    df_dict['Subroutine'] += ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CR"]
    df_dict['System'] += ['H-C'] * 10
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_inequality_c1)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_superposition_ij_c2)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_superposition_wrs_c3)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_controlled_swaps_c4)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_extra_nuclear_momentum_c5)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_nested_boxes_c6)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_prep_unprep_nuclear_via_qrom_c7)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_add_subtract_momentum_for_select_c8)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_phasing_by_structure_factor_c9)
    df_dict['Toffoli Count'].append(cost_dataclass_C.tofc_reflection_costs_cr)


    df_dict['Subroutine'] += ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CR"]
    df_dict['System'] += ['He-H'] * 10
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_inequality_c1)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_superposition_ij_c2)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_superposition_wrs_c3)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_controlled_swaps_c4)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_extra_nuclear_momentum_c5)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_nested_boxes_c6)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_prep_unprep_nuclear_via_qrom_c7)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_add_subtract_momentum_for_select_c8)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_phasing_by_structure_factor_c9)
    df_dict['Toffoli Count'].append(cost_dataclass_He.tofc_reflection_costs_cr)

    df_dict['Subroutine'] += ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CR"]
    df_dict['System'] += ['H-D'] * 10
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_inequality_c1)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_superposition_ij_c2)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_superposition_wrs_c3)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_controlled_swaps_c4)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_extra_nuclear_momentum_c5)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_nested_boxes_c6)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_prep_unprep_nuclear_via_qrom_c7)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_add_subtract_momentum_for_select_c8)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_phasing_by_structure_factor_c9)
    df_dict['Toffoli Count'].append(cost_dataclass_hd.tofc_reflection_costs_cr)


    df = pd.DataFrame().from_dict(df_dict)
    

    import seaborn as sns
    sns.set_context('paper')
    ax = sns.barplot(x='Subroutine', y='Toffoli Count', hue ='System', data = df,
                     edgecolor = 'w',
                     palette=colors[:3]) #[colors[0], colors[2], colors[3]]) # palette = 'Blues',
    ax.set_yscale("log")
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("Subroutine", fontsize=14)
    ax.set_ylabel(r"Toffolis", fontsize=14)
    ax.legend(loc='upper right', fontsize=14, ncol=1, frameon=False)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=90)

    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("seaborn_plt.png", format='PNG', dpi=300)
    plt.savefig("Subroutine_Costs.png", format='PNG', dpi=300)
    plt.savefig("Subroutine_Costs.pdf", format='PDF', dpi=300)
    plt.show()

 

if __name__ == "__main__":
    block_encoding_costs()