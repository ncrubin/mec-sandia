"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
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


def spectral_norm_power_method(A, x, verbose=False, stop_eps=1.0E-8):
    prev_sqrt_lam_max = np.inf
    delta_sqrt_lam_max = np.inf
    iter_val = 0
    x /= np.linalg.norm(x)
    AdA = A.conj().T @ A
    while delta_sqrt_lam_max > stop_eps:
        r = AdA @ x
        x = r / np.linalg.norm(r)
        sqrt_lam_max = np.sqrt((x.conj().T @ AdA @ x).real)

        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}")
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    return sqrt_lam_max


def spectral_norm_svd(A):
    _, sigma, _ = np.linalg.svd(A)
    return np.max(sigma)


def strang_u(work: fqe.Wavefunction, t: float, fqe_ham_ob: RestrictedHamiltonian, fqe_ham_tb: RestrictedHamiltonian ):
    """Strang split-operator evolution"""
    work = work.time_evolve(t * 0.5, fqe_ham_ob)
    work = work.time_evolve(t, fqe_ham_tb)
    work = work.time_evolve(t * 0.5, fqe_ham_ob)
    return work

def exact_then_strang_u_inverse(work: fqe.Wavefunction,
                                t: float,
                                full_ham: RestrictedHamiltonian,
                                h0: RestrictedHamiltonian,
                                h1: RestrictedHamiltonian):
    """U_{strang}^{\dagger}U_{exact}
    """
    work = work.time_evolve(t, full_ham)
    work = work.time_evolve(-t * 0.5, h0)
    work = work.time_evolve(-t, h1)
    work = work.time_evolve(-t * 0.5, h0)
    return work

def strang_u_then_exact_inverse(work: fqe.Wavefunction,
                                t: float,
                                full_ham: RestrictedHamiltonian,
                                h0: RestrictedHamiltonian,
                                h1: RestrictedHamiltonian):
    """U_{exact}^{\dagger}U_{strang}
    """
    work = work.time_evolve(t * 0.5, h0)
    # work = work.apply_generated_unitary(t * 0.5, algo='taylor', ops=h0)
    work = work.time_evolve(t, h1)
    work = work.time_evolve(t * 0.5, h0)
    work = work.time_evolve(-t, full_ham)
    return work

def deltadagdelta_action(work: fqe.Wavefunction,
                         t: float,
                         full_ham: RestrictedHamiltonian,
                         h0: RestrictedHamiltonian,
                         h1: RestrictedHamiltonian):
    og_work = copy.deepcopy(work)
    w1 = exact_then_strang_u_inverse(work, t, full_ham, h0, h1) + strang_u_then_exact_inverse(work, t, full_ham, h0, h1)
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
        work = deltadagdelta_action(work, t, full_ham, h0, h1)
        rnorm = work.norm()
        work.scale(1/rnorm) 
        sqrt_lam_max = np.sqrt(
            np.abs(
            fqe.vdot(work, deltadagdelta_action(work, t, full_ham, h0, h1))
            )
            )
        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}")
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    return sqrt_lam_max

