"""
Calculate the spectral norm of the difference between a product formula and the exact unitary.
"""
import time
import numpy as np
import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

def spectral_norm_power_method(A, x, verbose=False, stop_eps=1.0E-8, return_vec=False):
    """
    Compute spectral norm of a matrix A (not necessarily Hermitain).  
    This is accomplished by finding the largest eigenvalue of A\dag A
    using the power method and then taking the square root.
    """
    prev_sqrt_lam_max = np.inf
    delta_sqrt_lam_max = np.inf
    iter_val = 0
    x /= np.linalg.norm(x)
    AdA = A.conj().T @ A
    while delta_sqrt_lam_max > stop_eps:
        r = AdA @ x
        rnorm = np.linalg.norm(r)
        x = r / rnorm
        sqrt_lam_max = np.sqrt((x.conj().T @ AdA @ x).real)

        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}", f"{np.sqrt(rnorm)=}", f"{(np.sqrt(rnorm) - sqrt_lam_max)=}")
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    if return_vec:
        return sqrt_lam_max, x
    else:
        return sqrt_lam_max

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
                        h1: RestrictedHamiltonian,
                        deltadag_delta_action,
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
    prev_sqrt_lam_max = np.inf
    delta_sqrt_lam_max = np.inf
    iter_val = 0
    work.normalize()
    while delta_sqrt_lam_max > stop_eps:
        start_time = time.time()
        work = deltadag_delta_action(work, t, full_ham, h0, h1)
        rnorm = work.norm()
        work.scale(1./rnorm) 
        # sqrt_lam_max = np.sqrt(
        #     np.abs(
        #     fqe.vdot(work, deltadag_delta_action(work, t, full_ham, h0, h1))
        #     ))

        # since A^A v approx lambda_max v.  We use the norm of the vector
        # which should converge to the eigenvalue. This saves applying the 
        # A^A twice per iteration
        sqrt_lam_max = np.sqrt(rnorm)         
        end_time = time.time()
        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}", "iter_time = {}".format((end_time - start_time)))
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    return sqrt_lam_max

