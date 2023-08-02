"""
Calculate the number of Trotter steps and number of Toffoli assuming constant factors
"""

import numpy as np

def tau(N, eta, omega):
    return 3 * (np.pi**2) * N**(2/3) / (2 * omega**(2/3))

def nu(N, eta, omega):
    return (np.pi**(1/3)) * (3/4)**(2/3) * eta**(2/3) * N**(1/3) / omega**(1/3)

def num_trotter_steps(xi, tau, nu, epsilon, dt, order, eta):
    return dt**(1 + 1/order) * (tau + nu)**(1 - 1/order) * (xi * tau * nu * eta / epsilon)**(1/order)

def compute_num_toffolis(num_trotter_steps, eta):
    num_exponentials = num_trotter_steps * 17
    complexity_per_exponential = 2380 * eta * (eta - 1) / 2
    return num_exponentials * complexity_per_exponential

def big_o_value(tau, nu, eta, dt, order):
    return (tau + nu)**(order - 1) * tau * nu * eta * dt**(order + 1)

if __name__ == "__main__":
    omega = 2419.68282
    N = 53**3
    eta = 218
    xi = 5.0E-10# E-5# E-5 # 2.5E-5 # pedro's Number
    eps = 0.01
    dt = 10.
    print(f"{tau(N, eta, omega)=}")
    print(f"{nu(N, eta, omega)=}")
    tau_val = tau(N, eta, omega)
    nu_val = nu(N, eta, omega)
    print(f"{big_o_value(tau_val, nu_val, eta, dt, 8)=}")

    print(f"{(big_o_value(tau_val, nu_val, eta, dt, 8) * xi)=:3.3e}")
    print(f"{xi**(1/8)=}")
    num_steps = np.ceil(num_trotter_steps(xi, tau_val, nu_val, eps, dt, 8, eta))
    print(f"{num_steps=}")
    print("{: 2.3e}".format(compute_num_toffolis(num_steps, eta)))
    total_toffolis = compute_num_toffolis(num_steps, eta) # * 100 # x100 for sampling
    num_factories = 4
    print("num days ", total_toffolis * 2E-9 / num_factories) # roughly num_days based on Craigs / Matt's estimates


    # from mec_sandia.vasp_utils import read_vasp
    from mec_sandia.config import VASP_DATA, REPO_DIRECTORY
    import os
    from ase.io import read, write
    from ase.units import Bohr

    ########################
    #
    # Calculate properties
    #
    #########################
    fname = os.path.join(VASP_DATA, "H2_2eV_POSCAR_50_percent.vasp")
    os.chdir(os.path.join(REPO_DIRECTORY, 'vasp_data'))
    ase_atom = read(fname)
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


    omega = 302.4603526295446 
    N = 27**3
    eta = 29
    xi = 5E-10 # E-5# E-5 # 2.5E-5 # pedro's Number
    eps = 0.01
    dt = 1.
    print(f"{tau(N, eta, omega)=}")
    print(f"{nu(N, eta, omega)=}")
    tau_val = tau(N, eta, omega)
    nu_val = nu(N, eta, omega)
    print(f"{big_o_value(tau_val, nu_val, eta, dt, 8)=}")

    print(f"{(big_o_value(tau_val, nu_val, eta, dt, 8) * xi)=:3.3e}")
    print(f"{xi**(1/8)=}")
    num_steps = np.ceil(num_trotter_steps(xi, tau_val, nu_val, eps, dt, 8, eta))
    print(f"{num_steps=}")
    print("{: 2.3e}".format(compute_num_toffolis(num_steps, eta)))
    total_toffolis = compute_num_toffolis(num_steps, eta) # * 100 # x100 for sampling
    num_factories = 4
    print("num days ", total_toffolis * 2E-9 / num_factories) # roughly num_days based on Craigs / Matt's estimates