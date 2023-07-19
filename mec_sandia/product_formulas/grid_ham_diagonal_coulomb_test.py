"""
Test the time-evolution with full 2-electron integrals and Diagonal Coulomb. 

The Diagonal Coulomb should be faster
"""
import sys
import numpy as np

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid 
from mec_sandia.product_formulas.time_evolution_utility import apply_unitary_wrapper
from mec_sandia.product_formulas.spectral_norm_product import spectral_norm_fqe_power_iteration
from mec_sandia.product_formulas.strang import delta_action


def run_spectral_norm_comp(t: float, points_per_dim: int, eta: int, omega: float):
    L = omega**(1/3)
    rsg = RealSpaceGrid(L, points_per_dim)
    h1 = rsg.get_rspace_h1()
    eris = 2 * rsg.get_eris()
    of_eris = eris.transpose((0, 2, 3, 1))
    dc_mat = 2 * rsg.get_diagonal_coulomb_matrix()
    if np.isclose(eta % 2, 1):
        sz = 1
    else:
        sz = 0
    nelec = eta
    norb = rsg.norb

    fqe_ham = integrals_to_fqe_restricted(h1, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1,))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))
    fqe_ham_tb_dc = DiagonalCoulomb(np.einsum('ijlk', -0.5 * of_eris))
    fqe_ham_tb_dc_v2 = DiagonalCoulomb(0.5 * dc_mat)

     # initialize new wavefunction
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='random')
    x_wfn.normalize()

    import time
    # calculate spectral norm
    start_time = time.time()
    full_two_body_wfn = apply_unitary_wrapper(base=x_wfn,
                                     time=t,
                                     algo='taylor',
                                     ops=fqe_ham_tb,
                                     accuracy = 1.0E-20,
                                     expansion=300,
                                     verbose=True)
    end_time = time.time()
    print("full-two-body time {}".format(end_time - start_time))
    start_time = time.time()
    dc_wfn = x_wfn.time_evolve(t, fqe_ham_tb_dc)
    end_time = time.time()
    print("DiagonalCoulomb time {}".format(end_time - start_time))
    diff_wfn = full_two_body_wfn - dc_wfn
    print(f"{np.linalg.norm(diff_wfn.sector((nelec, sz)).coeff)=}")

    start_time = time.time()
    dc_wfn_v2 = x_wfn.time_evolve(t, fqe_ham_tb_dc_v2)
    end_time = time.time()
    print("DiagonalCoulomb-v2 time {}".format(end_time - start_time))
    diff_wfn = full_two_body_wfn - dc_wfn_v2
    print(f"{np.linalg.norm(diff_wfn.sector((nelec, sz)).coeff)=}")


if __name__ == "__main__":
    # python -u /home/nickrubin_google_com/trotter_scaling/grid/grid_ham_strang_spectral_norm_exe.py 0.65 2 4 5.0
    t = 0.25
    ppd = 2
    eta = 3
    omega = 3.0
    run_spectral_norm_comp(t, ppd, eta, omega)
