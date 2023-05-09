import math
from re import I
from typing import Tuple, Union

import numpy as np
import scipy.optimize
from pyscf.lib.numpy_helper import cartesian_prod


def estimate_error_kinetic_energy(kcut: float, sigma: float) -> float:
    a = kcut
    b = 2 * sigma**2.0
    t1 = 2 * a * np.exp(-(a**2.0) / b)
    t2 = np.sqrt(np.pi * b) * scipy.special.erfc(a / (b**0.5))
    prefactor = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    return 0.25 * prefactor * b * (t1 + t2)


def estimate_kinetic_energy(kcut: float, sigma: float) -> float:
    a = kcut
    b = 2 * sigma**2.0
    t1 = 2 * a * np.exp(-(a**2.0) / b)
    t2 = np.sqrt(np.pi * b) * scipy.special.erf(a / (b**0.5))
    prefactor = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    return 0.25 * prefactor * b * (t2 + t1)


def estimate_energy_cutoff(target_precision: float, sigma: float) -> float:
    objective = (
        lambda x, sigma: (estimate_error_kinetic_energy(x, sigma) - target_precision)
        ** 2.0
    )
    brackets = scipy.optimize.bracket(objective, xa=10, xb=2 * sigma, args=(sigma,))
    x0 = scipy.optimize.bisect(
        lambda x, sigma: estimate_error_kinetic_energy(x, sigma) - target_precision,
        brackets[0],
        brackets[2],
        args=(sigma,),
    )
    return 0.5 * x0**2.0


def estimate_energy_cutoff_sum(target_precision, sigma):
    return 4 * estimate_energy_cutoff(target_precision, sigma)


def get_ngmax(ecut, box_length):
    ng_max = math.ceil(np.sqrt(2 * ecut) / (2 * np.pi / box_length))
    return ng_max


def _build_gaussian(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
    mu: Union[np.ndarray, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if mu is None:
        mu = np.zeros(
            ndim,
        )
    nmax = get_ngmax(ecut_hartree, box_length)
    grid = (np.arange(-nmax / 2, nmax / 2 + 1, dtype=int),) * ndim
    grid_spacing = 2 * np.pi / box_length
    kgrid = grid_spacing * cartesian_prod(grid)
    gaussian = np.exp(
        -(np.sum((kgrid - mu[None, :]) ** 2.0, axis=-1)) / (4 * sigma**2.0)
    )
    return gaussian, kgrid


def discrete_gaussian_wavepacket(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""Build a discrete Gaussian wavepacket.

    This function builds the amplitudes associated with
    $
        |\psi\rangle &= \frac{1}{\mathcal{N}} \sum_\mathbf{k} e^{\mathbf{k}^2/(4\sigma_k^2)} |\mathbf{k}\rangle \\
                     &= \sum_\mathbf{k} \sqrt{p_k} |\mathbf{k}\rangle
    $
    
    Args:
        ecut_hartree: Cutoff energy in Hartree. This defines the k-grid size for
            the wavepacket. See get_ngmax.
        box_length: The boxlength in bohr.
        sigma: The reciprocal space wavepacket standard deviation. sigma^2 =
            variance.
        ndim: The number of dimensions.

    Returns:
        sqrt_pk: The wavefunction amplitudes defined above ($\sqrt{p_k}$).
        kgrid: The k-space grid.
        normalization: The normalization factor for the Gaussian.
    """
    assert ndim > 0
    gaussian, kgrid = _build_gaussian(ecut_hartree, box_length, sigma, ndim=ndim)
    normalization = np.sqrt(np.sum(gaussian**2.0))
    return gaussian / normalization, kgrid, normalization


def _calc_kinetic_energy(kgrid: np.ndarray, kproj: np.ndarray, p_k: np.ndarray):
    kinetic_energy = np.sum(
        0.5 * np.sum((kgrid + kproj[None, :]) ** 2.0, axis=-1) * p_k
    )
    return kinetic_energy


def kinetic_energy(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
    kproj: Union[np.ndarray, None] = None,
) -> float:
    """kproj needs to be in units of a0^{-1}"""
    if kproj is None:
        kproj = np.zeros(
            ndim,
        )
    gaussian, kgrid, _ = discrete_gaussian_wavepacket(
        ecut_hartree, box_length, sigma, ndim=ndim
    )
    ke = _calc_kinetic_energy(kgrid, kproj, gaussian**2.0)
    return ke


def estimate_kinetic_energy_sampling(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
    num_samples: int = 1000,
    kproj: Union[np.ndarray, None] = None,
    shift_by_constant=False,
) -> Tuple[float, float]:
    if kproj is None:
        kproj = np.zeros(
            ndim,
        )
    # Builds |psi> = N^{-1/2} e^{-k^2/(4sigma^2)}
    gaussian, kgrid = _build_gaussian(ecut_hartree, box_length, sigma, ndim=ndim)
    norm = np.sum(gaussian**2.0)
    indx_k = np.random.choice(np.arange(len(kgrid)), num_samples, p=gaussian**2.0 / norm)
    k_select = kgrid[indx_k]
    kinetic_samples = 0.5 * np.sum((k_select + kproj[None, :]) ** 2.0, axis=-1)
    if shift_by_constant:
        kinetic_samples -= 0.5 * np.dot(kproj, kproj)
    return (
        np.mean(kinetic_samples),
        np.std(kinetic_samples, ddof=1) / num_samples**0.5,
    )


def estimate_kinetic_energy_importance_sampling(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    importance_function: np.ndarray,
    ndim: int = 1,
    num_samples: int = 1000,
    kproj: Union[np.ndarray, None] = None,
) -> Tuple[float, float]:
    if kproj is None:
        kproj = np.zeros(
            ndim,
        )
    gaussian, kgrid = _build_gaussian(ecut_hartree, box_length, sigma, ndim=ndim)
    p_x = gaussian**2.0 / sum(gaussian**2.0)
    q_x = importance_function
    indx_k = np.random.choice(np.arange(len(kgrid)), num_samples, p=q_x)
    k_select = kgrid[indx_k]
    kinetic_samples = (
        0.5
        * np.sum((k_select + kproj[None, :]) ** 2.0, axis=-1)
        * (p_x[indx_k] / q_x[indx_k])
    )
    return (
        np.mean(kinetic_samples),
        np.std(kinetic_samples, ddof=1) / num_samples**0.5,
    )
