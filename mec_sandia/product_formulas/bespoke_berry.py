import os
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'

import copy
import numpy as np
import time

from pyscf import gto, scf, ao2mo
from pyscf.fci.cistring import make_strings

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

import openfermion as of
from openfermion import MolecularData
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial

from mec_sandia.product_formulas.pyscf_utility import get_spectrum, pyscf_to_fqe_wf

def u_s2_trotter_cirq(t: float,
                      h0,
                      h1):
    from scipy.linalg import expm
    u = np.eye(h0.shape[0])
    u = u @ expm(-1j * t * 0.5 * h0)
    u = u @ expm(-1j * t * h1)
    u = u @ expm(-1j * t * 0.5 * h0)
    return u

def u_berry_bespoke_cirq(
                  t: float,
                  h0,
                  h1):
    """S2(w_10 t) S2(w_9 t) .... S2(w_2 t) S2(w_1 t) S2(w_0 t) S2(w_1 t) S2(w_2t) .... S2(w_9 t) S2(w_10 t)
    S2(t) =  exp(iAt/2) exp(iBt) exp(iAt/2)
    """
    wvals = [0.59358060400850625863463274318848,
             -0.46916012347004197296246963141914, 0.27435664258984679072234958738716,
              0.1719387948465677305996164814603, 0.23439874482541384415374265697566,
             -0.48616424480326193899633138022874, 0.49617367388114660354885678908755,
             -0.32660218948439130114486683568042, 0.23271679349369857679469542295681,
              0.098249557414708533273496706507094] 
    w_0 = 1 - 2 * sum(wvals) 
    u = np.eye(h0.shape[0])
    for ii in range(9, -1, -1):
        u = u @ u_s2_trotter_cirq(wvals[ii] * t, h0, h1)
    u = u @ u_s2_trotter_cirq(w_0 * t, h0, h1)
    for ii in range(10):
        u = u @ u_s2_trotter_cirq(wvals[ii] * t, h0, h1)
    return u



def evolve_s2_trotter(work: fqe.Wavefunction,
                      t: float,
                      h0: RestrictedHamiltonian,
                      h1: RestrictedHamiltonian):
    work = work.time_evolve(t * 0.5, h0)
    work = work.apply_generated_unitary(time=t,
                                        algo='taylor',
                                        ops=h1,
                                        accuracy=1.0E-20,
                                        expansion=200,
                                        verbose=False)
    work = work.time_evolve(t * 0.5, h0)
    return work

def berry_bespoke(work: fqe.Wavefunction,
                  t: float,
                  h0: RestrictedHamiltonian,
                  h1: RestrictedHamiltonian):
    """S2(w_10 t) S2(w_9 t) .... S2(w_2 t) S2(w_1 t) S2(w_0 t) S2(w_1 t) S2(w_2t) .... S2(w_9 t) S2(w_10 t)
    S2(t) =  exp(iAt/2) exp(iBt) exp(iAt/2)
    """
    wvals = [0.59358060400850625863463274318848,
             -0.46916012347004197296246963141914, 0.27435664258984679072234958738716,
              0.1719387948465677305996164814603, 0.23439874482541384415374265697566,
             -0.48616424480326193899633138022874, 0.49617367388114660354885678908755,
             -0.32660218948439130114486683568042, 0.23271679349369857679469542295681,
              0.098249557414708533273496706507094] 
    w_0 = 1 - 2 * sum(wvals) 
    for ii in range(9, -1, -1):
        work = evolve_s2_trotter(work, wvals[ii] * t, h0, h1)
    work = evolve_s2_trotter(work, w_0 * t, h0, h1)
    for ii in range(10):
        work = evolve_s2_trotter(work, wvals[ii] * t, h0, h1)
    return work

def exact_then_berry_u_inverse(work: fqe.Wavefunction,
                               t: float,
                               full_ham: RestrictedHamiltonian,
                               h0: RestrictedHamiltonian,
                               h1: RestrictedHamiltonian):
    """U_{berry}^ U_exact"""
    # note: we do not use time_evolve because
    # we need to customize the expansion rank
    work = work.apply_generated_unitary(time=t,
                                        algo='taylor',
                                        ops=full_ham,
                                        accuracy=1.0E-20,
                                        expansion=200,
                                        verbose=False)
    work = berry_bespoke(work, -t, h0, h1)
    return work

def berry_u_then_exact_inverse(work: fqe.Wavefunction,
                               t: float,
                               full_ham: RestrictedHamiltonian,
                               h0: RestrictedHamiltonian,
                               h1: RestrictedHamiltonian):
    """U_{exact}^U_{berry}"""
    work = berry_bespoke(work, t, h0, h1)
    work = work.apply_generated_unitary(time=-t,
                                        algo='taylor',
                                        ops=full_ham,
                                        accuracy=1.0E-20,
                                        expansion=200,
                                        verbose=False)
    return work

def berry_deltadagdelta_action(work: fqe.Wavefunction,
                         t: float,
                         full_ham: RestrictedHamiltonian,
                         h0: RestrictedHamiltonian,
                         h1: RestrictedHamiltonian):
    og_work = copy.deepcopy(work)
    w1 = exact_then_berry_u_inverse(work, t, full_ham, h0, h1)
    w2 = berry_u_then_exact_inverse(work, t, full_ham, h0, h1)
    og_work.scale(2.)
    return og_work - w1 - w2

def spectral_norm_fqe_power_iteration(work: fqe.Wavefunction,
                        t: float,
                        full_ham: RestrictedHamiltonian,
                        h0: RestrictedHamiltonian,
                        h1: RestrictedHamiltonian,
                        verbose=True,
                        stop_eps=1.0E-8):
    """Return spectral norm of the difference between product formula unitary and not"""
    prev_sqrt_lam_max = np.inf
    delta_sqrt_lam_max = np.inf
    iter_val = 0
    work.normalize()
    while delta_sqrt_lam_max > stop_eps:
        start_time = time.time()
        work = berry_deltadagdelta_action(work, t, full_ham, h0, h1)
        rnorm = work.norm()
        work.scale(1./rnorm) 
        sqrt_lam_max = np.sqrt(
            np.abs(
            fqe.vdot(work, berry_deltadagdelta_action(work, t, full_ham, h0, h1))
            ))
        end_time = time.time()
        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}", "iter_time = {}".format((end_time - start_time)))
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    return sqrt_lam_max

if __name__ == "__main__":
    from strang_spectral_norm_test import lih_molecule, heh_molecule
    # mol_mf = lih_molecule(basis='sto-3g')
    mol_mf = heh_molecule()
    nelec = mol_mf.mol.nelectron
    nalpha = nelec // 2
    nbeta = nelec // 2
    norb = mol_mf.mo_coeff.shape[0]
    print(f"{nelec=}", f"{norb=}")
    sz = 0
    of_eris = mol_mf._eri.transpose((0, 2, 3, 1))
    fqe_ham = integrals_to_fqe_restricted(mol_mf.get_hcore(), of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((mol_mf.get_hcore(), ))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(mol_mf.get_hcore()), np.einsum('ijlk', -0.5 * of_eris)))

    # set up cirq hamiltonians
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(mol_mf.get_hcore(), of_eris)
    molecular_hamiltonian = InteractionOperator(0, one_body_coefficients, 1 / 2 * two_body_coefficients)
    sparse_ham = of.get_sparse_operator(molecular_hamiltonian).todense()
    molecular_hamiltonian_ob = InteractionOperator(0, one_body_coefficients, np.zeros_like(1 / 2 * two_body_coefficients))
    molecular_hamiltonian_tb = InteractionOperator(0, np.zeros_like(one_body_coefficients), 1 / 2 * two_body_coefficients)
    sparse_ham_ob = of.get_sparse_operator(molecular_hamiltonian_ob).todense()
    sparse_ham_tb = of.get_sparse_operator(molecular_hamiltonian_tb).todense()

    from scipy.linalg import expm
    t = 5 # 0.78
    exact_u = expm(-1j * t * sparse_ham)
    from mec_sandia.product_formulas.bespoke_berry import u_berry_bespoke_cirq
    from mec_sandia.product_formulas.strang_spectral_norm import spectral_norm_power_method as spectral_norm_power_method_cirq, spectral_norm_svd

    np.random.seed(50)
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    x_cirq = fqe.to_cirq(x_wfn)

    print("Starting Berry construction")
    berry_u = u_berry_bespoke_cirq(t, sparse_ham_ob, sparse_ham_tb) 
    diff_u = berry_u - exact_u
    diff_u_expanded = 2 * np.eye(berry_u.shape[0]) - berry_u.conj().T @ exact_u - exact_u.conj().T @ berry_u
    cirq_spectral_norm, max_vec = spectral_norm_power_method_cirq(diff_u, x_cirq, verbose=True, stop_eps=1.0E-10, return_vec=True)
    print()
    print(f"{cirq_spectral_norm=}")
    print()
    true_spec_norm = spectral_norm_svd(diff_u)
    print(f"{true_spec_norm=}")

    max_vec_wfn = fqe.from_cirq(max_vec, thresh=1.0E-14)
    max_vec_wfn.print_wfn()
    fqe_spectral_norm = np.sqrt(
            np.abs(
            fqe.vdot(max_vec_wfn, berry_deltadagdelta_action(max_vec_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb))
            ))
    # fqe_spectral_norm = spectral_norm_fqe_power_iteration(max_vec_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-10)
    print(f"{ fqe_spectral_norm=}")
    print(f"{cirq_spectral_norm=}")

    tst_sqrt = np.sqrt(np.abs(max_vec.conj().T @ diff_u_expanded @ max_vec))
    print(f"{tst_sqrt=}")
    assert np.isclose(tst_sqrt, true_spec_norm)

    # up
    berry_u_action_cirq = berry_u @ max_vec
    berry_u_action_cirq = fqe.from_cirq(berry_u_action_cirq, thresh=1.0E-14)
    test_berry_u_action_fqe = berry_bespoke(max_vec_wfn, t, fqe_ham_ob, fqe_ham_tb)
    assert np.isclose(fqe.vdot(test_berry_u_action_fqe, berry_u_action_cirq), 1)

    # u
    exact_u_action_cirq = exact_u @ max_vec
    exact_u_action_cirq = fqe.from_cirq(exact_u_action_cirq, thresh=1.0E-14)
    test_exact_u_action_fqe = max_vec_wfn.apply_generated_unitary(time=t,
                                        algo='taylor',
                                        ops=fqe_ham,
                                        accuracy=1.0E-20,
                                        expansion=200,
                                        verbose=True)
    assert np.isclose(fqe.vdot(test_exact_u_action_fqe, exact_u_action_cirq), 1)

    # up^u
    part_delta_cirq = berry_u.conj().T @ exact_u @ max_vec
    part_delta_cirq = fqe.from_cirq(part_delta_cirq, thresh=1.0E-14)
    test_part_delta_fqe = max_vec_wfn.apply_generated_unitary(time=t,
                                        algo='taylor',
                                        ops=fqe_ham,
                                        accuracy=1.0E-20,
                                        expansion=200,
                                        verbose=True)
    test_part_delta_fqe = berry_bespoke(test_part_delta_fqe, -t, fqe_ham_ob, fqe_ham_tb)
    assert np.isclose(fqe.vdot(test_part_delta_fqe, part_delta_cirq), 1)

    # (up^u + u^ up)
    part_delta_cirq = (berry_u.conj().T @ exact_u + exact_u.conj().T @ berry_u) @ max_vec
    part_delta_cirq = fqe.from_cirq(part_delta_cirq, thresh=1.0E-14)
    test_part_delta_fqe1 = max_vec_wfn.apply_generated_unitary(time=t,
                                        algo='taylor',
                                        ops=fqe_ham,
                                        accuracy=1.0E-20,
                                        expansion=200,
                                        verbose=True)
    test_part_delta_fqe1 = berry_bespoke(test_part_delta_fqe1, -t, fqe_ham_ob, fqe_ham_tb)

    test_part_delta_fqe2 = berry_bespoke(max_vec_wfn, t, fqe_ham_ob, fqe_ham_tb)
    test_part_delta_fqe2 = test_part_delta_fqe2.apply_generated_unitary(time=-t,
                                        algo='taylor',
                                        ops=fqe_ham,
                                        accuracy=1.0E-20,
                                        expansion=200,
                                        verbose=True)
    test_part_delta_fqe = test_part_delta_fqe1 +  test_part_delta_fqe2
    assert np.allclose(test_part_delta_fqe.sector((nelec, 0)).coeff, part_delta_cirq.sector((nelec, 0)).coeff)
    assert np.isclose(test_part_delta_fqe.norm(), part_delta_cirq.norm())

    # 2 - (up^u + u^ up)
    full_delta_cirq = 2 * max_vec - (berry_u.conj().T @ exact_u + exact_u.conj().T @ berry_u) @ max_vec
    full_delta_cirq = fqe.from_cirq(full_delta_cirq, thresh=1.0E-14)

    new_wf = copy.deepcopy(max_vec_wfn)
    new_wf.scale(2)
    test_full_delta_fqe = new_wf - test_part_delta_fqe
    assert np.allclose(test_full_delta_fqe.sector((nelec, 0)).coeff, full_delta_cirq.sector((nelec, 0)).coeff)

    test_out = berry_deltadagdelta_action(max_vec_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb)
    assert np.allclose(test_out.sector((nelec, 0)).coeff, full_delta_cirq.sector((nelec, 0)).coeff)

    print("Test out")
    test_out.print_wfn(threshold=1.E-15)
    print("full_delta_cirq")
    full_delta_cirq.print_wfn(threshold=1.0E-15)
    print(full_delta_cirq.sector((nelec, 0)).coeff)
    print(test_out.sector((nelec, 0)).coeff)
    exit()
    tsqrt_fqe = fqe.vdot(max_vec_wfn, test_out)
    print(tsqrt_fqe)
    print(max_vec.conj().T @ fqe.to_cirq(test_out))
    print(max_vec.conj().T @ fqe.to_cirq(full_delta_cirq))
    print(np.sqrt(np.abs(max_vec.conj().T @ fqe.to_cirq(full_delta_cirq))))


