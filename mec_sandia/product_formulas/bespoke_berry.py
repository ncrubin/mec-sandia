import os
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'
import copy
import openfermion as of
import numpy as np
import fqe
from pyscf import gto, scf, ao2mo
from pyscf.fci.cistring import make_strings
from openfermion import MolecularData

from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf_utility import get_spectrum, pyscf_to_fqe_wf
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
import time


def evolve_s2_trotter(work: fqe.Wavefunction,
                      t: float,
                      h0: RestrictedHamiltonian,
                      h1: RestrictedHamiltonian):
    work = work.time_evolve(t * 0.5, h0)
    work = work.time_evolve(t, h1)
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
    work = work.time_evolve(t, full_ham)
    work = berry_bespoke(work, -t, h0, h1)
    return work

def berry_u_then_exact_inverse(work: fqe.Wavefunction,
                               t: float,
                               full_ham: RestrictedHamiltonian,
                               h0: RestrictedHamiltonian,
                               h1: RestrictedHamiltonian):
    """U_{exact}^U_{berry}"""
    work = berry_bespoke(work, t, h0, h1)
    work = work.time_evolve(-t, full_ham)
    return work

def berry_deltadagdelta_action(work: fqe.Wavefunction,
                         t: float,
                         full_ham: RestrictedHamiltonian,
                         h0: RestrictedHamiltonian,
                         h1: RestrictedHamiltonian):
    og_work = copy.deepcopy(work)
    w1 = exact_then_berry_u_inverse(work, t, full_ham, h0, h1) + berry_u_then_exact_inverse(work, t, full_ham, h0, h1)
    og_work.scale(2)
    work = og_work - w1
    return work

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
        work.scale(1/rnorm) 
        sqrt_lam_max = np.sqrt(
            np.abs(
            fqe.vdot(work, berry_deltadagdelta_action(work, t, full_ham, h0, h1))
            ))
        end_time = time.time()
        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}", f"{(end_time - start_time)=}")
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    return sqrt_lam_max

if __name__ == "__main__":
    from strang_spectral_norm_test import lih_molecule
    mol_mf = lih_molecule(basis='3-21g')
    # mol_mf = heh_molecule()
    nelec = mol_mf.mol.nelectron
    norb = mol_mf.mo_coeff.shape[0]
    print(f"{nelec=}", f"{norb=}")
    sz = 0
    of_eris = mol_mf._eri.transpose((0, 2, 3, 1))
    fqe_ham = integrals_to_fqe_restricted(mol_mf.get_hcore(), of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((mol_mf.get_hcore(), ))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(mol_mf.get_hcore()), np.einsum('ijlk', -0.5 * of_eris)))

    np.random.seed(50)
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()

    t = 0.78
    fqe_spectral_norm = spectral_norm_fqe_power_iteration(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8)
    print(f"{ fqe_spectral_norm=}")


