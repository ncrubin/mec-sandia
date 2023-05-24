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
    
    blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v5(
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
    
    blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v5(
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
    
    blockencodingtoff, lambdaval, qubit = pw_qubitization_with_projectile_costs_from_v5(
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



if __name__ == "__main__":
    block_encoding_costs()