"""
Calculate the spectral norm of the difference between a product formula and the exact unitary.
"""
from typing import Callable, Union
import time
import numpy as np
import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

def spectral_norm_power_method(A, x, verbose=False, stop_eps=1.0E-8, return_vec=False):
    """
    Compute spectral norm of a matrix A (not necessarily Hermitain).  
    This is accomplished by finding the largest eigenvalue of A\dag A
    using the power method and then taking the square root.
    """
    prev_spec_norm_estimate = np.inf
    delta_spec_norm_estimate = np.inf
    iter_val = 0
    x /= np.linalg.norm(x)
    Ad = A.conj().T 
    while delta_spec_norm_estimate > stop_eps:
        x = A @ x
        spec_norm_estimate = np.linalg.norm(x)
        x = Ad @ x
        x /= np.linalg.norm(x)
        delta_spec_norm_estimate = np.abs(prev_spec_norm_estimate - spec_norm_estimate)
        if verbose:
            print(iter_val, f"{spec_norm_estimate=}", f"{delta_spec_norm_estimate=}")
        prev_spec_norm_estimate = spec_norm_estimate
        iter_val += 1

    if return_vec:
        return spec_norm_estimate, x
    else:
        return spec_norm_estimate

def spectral_norm_power_method_hermitian_mat(A, x, verbose=False, stop_eps=1.0E-8, return_vec=False):
    """
    Compute spectral norm of a Hermitian matrix A.  This is accomplished by finding the largest
    eigenvalue by the power method and then taking the square root.
    """
    prev_sqrt_lam_max = np.inf
    delta_sqrt_lam_max = np.inf
    iter_val = 0
    x /= np.linalg.norm(x)
    while delta_sqrt_lam_max > stop_eps:
        r = A @ x
        x = r / np.linalg.norm(r)
        sqrt_lam_max = np.sqrt((x.conj().T @ A @ x).real)

        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}")
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    if return_vec:
        return sqrt_lam_max, x
    else:
        return sqrt_lam_max


def spectral_norm_svd(A: np.ndarray) -> float:
    """Compute the spectral norm of a matrix by SVD decomposition"""
    _, sigma, _ = np.linalg.svd(A)
    return np.max(sigma)


def spectral_norm_fqe_power_iteration(work: fqe.Wavefunction,
                        t: float,
                        full_ham: RestrictedHamiltonian,
                        h0: RestrictedHamiltonian,
                        h1: Union[RestrictedHamiltonian, DiagonalCoulomb],
                        delta_action: Callable,
                        verbose=True,
                        stop_eps=1.0E-8) -> float:
    """Return spectral norm of the difference between product formula unitary and exact unitary
    ||U_{p} - U_{exact}|| 

    :param work: fqe.Wavefunction to initialize the power method
    :param t: time Evolution
    :param full_ham: Full Hamiltonian
    :param h0: the one-body (quadratic) part of the Hamiltonian
    :param h1: the two-body part of the Hamiltonian
    """
    prev_spec_norm_estimate = np.inf
    delta_spec_norm_estimate = np.inf
    iter_val = 0
    while delta_spec_norm_estimate > stop_eps:
        start_time = time.time()
        work = delta_action(work, t, full_ham, h0, h1)
        spec_norm_estimate = work.norm()
        work.scale(1./spec_norm_estimate) # .scale(spec_norm_estimate)
        work = delta_action(work, -t, full_ham, h0, h1) # this will error out with non-normalized wavefunctions
        work.scale(spec_norm_estimate) # cancels the 1./spec norm from 2 lines ago
        rnorm = work.norm()
        work.scale(1./rnorm)
        end_time = time.time()
        delta_spec_norm_estimate = np.abs(prev_spec_norm_estimate - spec_norm_estimate)
        if verbose:
            print(iter_val, f"{spec_norm_estimate=}", f"{delta_spec_norm_estimate=}", "iter_time = {}".format((end_time - start_time)))
        prev_spec_norm_estimate = spec_norm_estimate
        iter_val += 1

    return spec_norm_estimate

