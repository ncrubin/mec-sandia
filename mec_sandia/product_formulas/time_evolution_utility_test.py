import os
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
import sys
import numpy as np
import time
import copy

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid 
from mec_sandia.product_formulas.time_evolution_utility import apply_unitary_wrapper, quad_and_diag_coulomb_apply_unitary_wrapper


def test_exact_ncr_exact_fqe():
    t = 3.0E-4 # 0.65
    points_per_dim = 3
    eta = 6
    omega = 5.

    L = omega**(1/3)
    rsg = RealSpaceGrid(L, points_per_dim)
    h1 = rsg.get_rspace_h1()
    eris = 2 * rsg.get_eris()
    dc_mat = 2 * rsg.get_diagonal_coulomb_matrix()
    of_eris = eris.transpose((0, 2, 3, 1))
    if np.isclose(eta % 2, 1):
        sz = 1
    else:
        sz = 0
    nelec = eta
    norb = rsg.norb

    fqe_ham = integrals_to_fqe_restricted(h1, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1,))
    fqe_ham_tb_rh = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))
    fqe_ham_tb = DiagonalCoulomb(0.5 * dc_mat)

     # initialize new wavefunction
    fqe.settings.use_accelerated_code = True
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='random')
    x_wfn.normalize()
    # print(x_wfn.sector((nelec, sz)).coeff.shape)
    # print(x_wfn.sector((nelec, sz))._low_thresh)

    test_final_wfn_1 = copy.deepcopy(x_wfn).apply(fqe_ham_ob)
    test_final_wfn_2 = copy.deepcopy(x_wfn).apply(fqe_ham_tb) 
    test_final_wfn = test_final_wfn_1 + test_final_wfn_2
    # test_final_wfn.print_wfn()
    true_final_wfn = copy.deepcopy(x_wfn).apply(fqe_ham)
    # true_final_wfn.print_wfn()
    print(np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - true_final_wfn.sector((nelec, sz)).coeff))
    assert np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - true_final_wfn.sector((nelec, sz)).coeff) < 2.0E-14

    print("ncr applyy generated unitary")
    start_time = time.time()
    test_final_wfn = quad_and_diag_coulomb_apply_unitary_wrapper(copy.deepcopy(x_wfn), t, quad_ham=fqe_ham_ob, diag_coulomb=fqe_ham_tb, verbose=True, debug=True)
    end_time = time.time()
    print("ncr_apply time ", end_time - start_time)
    print()
    print("sandia apply unitary")

    start_time = time.time()
    exact_final_wfn = apply_unitary_wrapper(copy.deepcopy(x_wfn), t, 'taylor', fqe_ham, verbose=True, debug=True)
    end_time = time.time()
    print("exact_apply time ", end_time - start_time)

    print(np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - exact_final_wfn.sector((nelec, sz)).coeff))
    assert np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - exact_final_wfn.sector((nelec, sz)).coeff) < 1.0E-14

def test_exact_partitioned_fqe():
    t = 5E-3 # 0.65
    points_per_dim = 3
    eta = 4
    omega = 5.

    L = omega**(1/3)
    rsg = RealSpaceGrid(L, points_per_dim)
    h1 = rsg.get_rspace_h1()
    eris = 2 * rsg.get_eris()
    dc_mat = 2 * rsg.get_diagonal_coulomb_matrix()
    of_eris = eris.transpose((0, 2, 3, 1))
    if np.isclose(eta % 2, 1):
        sz = 1
    else:
        sz = 0
    nelec = eta
    norb = rsg.norb

    fqe_ham = integrals_to_fqe_restricted(h1, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1,))
    fqe_ham_tb_rh = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))
    fqe_ham_tb = DiagonalCoulomb(0.5 * dc_mat)

     # initialize new wavefunction
    fqe.settings.use_accelerated_code = True
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='random')
    x_wfn.normalize()
    # print(x_wfn.sector((nelec, sz)).coeff.shape)
    # print(x_wfn.sector((nelec, sz))._low_thresh)

    test_final_wfn_1 = copy.deepcopy(x_wfn).apply(fqe_ham_ob)
    test_final_wfn_2 = copy.deepcopy(x_wfn).apply(fqe_ham_tb) 
    test_final_wfn = test_final_wfn_1 + test_final_wfn_2
    # test_final_wfn.print_wfn()
    true_final_wfn = copy.deepcopy(x_wfn).apply(fqe_ham)
    # true_final_wfn.print_wfn()
    print(np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - true_final_wfn.sector((nelec, sz)).coeff))
    assert np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - true_final_wfn.sector((nelec, sz)).coeff) < 2.0E-14

    print("ncr applyy generated unitary")
    start_time = time.time()
    test_final_wfn = quad_and_diag_coulomb_apply_unitary_wrapper(copy.deepcopy(x_wfn), t, quad_ham=fqe_ham_ob, diag_coulomb=fqe_ham_tb, verbose=True, debug=True)
    end_time = time.time()
    print("ncr_apply time ", end_time - start_time)
    print()
    print("sandia apply unitary")

    start_time = time.time()
    exact_final_wfn = apply_unitary_wrapper(copy.deepcopy(x_wfn), t, 'taylor', fqe_ham, verbose=True, debug=True)
    end_time = time.time()
    print("exact_apply time ", end_time - start_time)

    print(np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - exact_final_wfn.sector((nelec, sz)).coeff))
    assert np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - exact_final_wfn.sector((nelec, sz)).coeff) < 1.0E-14

def test_smallest_timeslice_factor():
    t = 0.2
    points_per_dim = 3
    eta = 6
    omega = 5.
    smallest_time_slice = 100

    L = omega**(1/3)
    rsg = RealSpaceGrid(L, points_per_dim)
    h1 = rsg.get_rspace_h1()
    eris = 2 * rsg.get_eris()
    dc_mat = 2 * rsg.get_diagonal_coulomb_matrix()
    of_eris = eris.transpose((0, 2, 3, 1))
    if np.isclose(eta % 2, 1):
        sz = 1
    else:
        sz = 0
    nelec = eta
    norb = rsg.norb

    fqe_ham = integrals_to_fqe_restricted(h1, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1,))
    fqe_ham_tb_rh = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))
    fqe_ham_tb = DiagonalCoulomb(0.5 * dc_mat)

     # initialize new wavefunction
    fqe.settings.use_accelerated_code = True
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='random')
    x_wfn.normalize()
    # print(x_wfn.sector((nelec, sz)).coeff.shape)
    # print(x_wfn.sector((nelec, sz))._low_thresh)

    test_final_wfn_1 = copy.deepcopy(x_wfn).apply(fqe_ham_ob)
    print(f"{test_final_wfn_1.sector((nelec, sz)).coeff.nbytes/(1024**3)=}")
    test_final_wfn_2 = copy.deepcopy(x_wfn).apply(fqe_ham_tb) 
    test_final_wfn = test_final_wfn_1 + test_final_wfn_2
    # test_final_wfn.print_wfn()
    true_final_wfn = copy.deepcopy(x_wfn).apply(fqe_ham)
    # exit()
    # # true_final_wfn.print_wfn()
    print(np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - true_final_wfn.sector((nelec, sz)).coeff))
    assert np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - true_final_wfn.sector((nelec, sz)).coeff) < 2.0E-13
    # exit()

    print("ncr applyy generated unitary")
    start_time = time.time()
    test_final_wfn = quad_and_diag_coulomb_apply_unitary_wrapper(copy.deepcopy(x_wfn), t, quad_ham=fqe_ham_ob, diag_coulomb=fqe_ham_tb, verbose=True, debug=True,
                                                                 smallest_time_slice=smallest_time_slice)
    end_time = time.time()
    print("ncr_apply time ", end_time - start_time)
    print()
    print("sandia apply unitary")

    start_time = time.time()
    exact_final_wfn = apply_unitary_wrapper(copy.deepcopy(x_wfn), t, 'taylor', fqe_ham, verbose=True, debug=True,
                                            smallest_time_slice=smallest_time_slice)
    end_time = time.time()
    print("exact_apply time ", end_time - start_time)

    print(np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - exact_final_wfn.sector((nelec, sz)).coeff))
    assert np.linalg.norm(test_final_wfn.sector((nelec, sz)).coeff - exact_final_wfn.sector((nelec, sz)).coeff) < 1.0E-13



if __name__ == "__main__":
    # test_exact_ncr_exact_fqe()
    # test_exact_partitioned_fqe()
    test_smallest_timeslice_factor()