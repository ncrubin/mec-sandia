"""Helper function for discrete and continuous Gaussian wavepackets."""
import math
from typing import Tuple, Union

import numpy as np
import scipy.optimize
from pyscf.lib.numpy_helper import cartesian_prod


def estimate_error_kinetic_energy(
    kcut: float,
    sigma: float,
    kproj: float = 0.0,
) -> float:
    r"""Estimate the error in the kinetic energy of 1D Gaussian wavepacket.

    The error in the kinetic energy incurred when limiting the maximum wavevector is given by

    $$
        \epsilon = 2 * (1/(2N_{k_c} )) \int_{0}^{k_c} (k^2-k_p^2) e^{-k^2/2\sigma^2}
                 - \frac{1}{2 N} \int_\infty^\infty (k^2 - k_p^2) e^{-k^2/2 \sigma^2}
    $$

    Note that the cross term from 2 k_p k vanishes as the integrand is odd. In the above we use

    $$
        N_{k_c}= 2 \int_{k_c}^\infty e^{-k^2/2\sigma^2}
    $$

    is the approximate normalization factor and $N = N_{\infty}$.

    Args:
        kcut: Maximum wavevector to include in the integral. This is the lower
            bound for the integral. This corresponds to $k_c$ in the docstring.
        sigma: Gaussian's standard deviation.
        kproj: Kinetic energy of the projectile.

    Returns:
        ke_err_apprx: Approximate kinetic energy
    """
    assert isinstance(kproj, float)
    # N_{k_c} in the docstring. Approximate normalization for this value of k_c.
    norm = (
        2 * np.sqrt(np.pi / 2) * sigma * scipy.special.erf(kcut / (np.sqrt(2) * sigma))
    )
    # Write approximate integral as term_a + term_b + term_c, terb_b is zero by symmetry.
    term_a = np.sqrt(np.pi / 2) * sigma**3.0 * scipy.special.erf(
        kcut / (np.sqrt(2) * sigma)
    ) - kcut * sigma**2.0 * np.exp(-(kcut**2.0) / (2 * sigma**2.0))
    term_c = 0.5 * norm * kproj**2.0
    ke_inf = (sigma**2.0 + kproj**2.0) / 2
    return np.abs((term_a + term_c) / norm - ke_inf)


def estimate_kinetic_energy(kcut: float, sigma: float, kproj: float = 0.0) -> float:
    r"""Estimate the error in the kinetic energy of 1D Gaussian wavepacket.

    The approximate kinetic energy given a maximum wavevector kcut ($k_c$).

    $$
        KE = 2 * (1/(2N_{k_c} )) \int_{0}^{k_c} (k^2-k_p^2) e^{-k^2/2\sigma^2}
    $$

    Note that the cross term from 2 k_p k vanishes as the integrand is odd. In the above we use

    $$
        N_{k_c}= 2 \int_{k_c}^\infty e^{-k^2/2\sigma^2}
    $$

    is the approximate normalization factor.

    Args:
        kcut: Maximum wavevector to include in the integral. This is the lower
            bound for the integral. This corresponds to $k_c$ in the docstring.
        sigma: Gaussian's standard deviation.
        kproj: Initial kinetic energy of the projectile. $k_p$ above.

    Returns:
        ke_err_apprx: Approximate kinetic energy
    """
    # Just add expected value to error with $k_c = \infty$.
    ke_int_inf = (sigma**2.0 + kproj**2.0) / 2
    return estimate_error_kinetic_energy(kcut, sigma, kproj) + ke_int_inf


def estimate_energy_cutoff(
    target_precision: float, sigma: float, kproj: float = 0.0
) -> float:
    """Estimate cutoff required using integral expressions for 1D Gaussian wavepacket.

    Args:
        sigma: Gaussian's standard deviation.
        kproj: Initial kinetic energy of the projectile. $k_p$ above.

    Returns:
        ecut: Approximate kinetic energy cutoff required.
    """

    def objective(kcut, sigma):
        return (
            estimate_error_kinetic_energy(kcut, sigma, kproj) - target_precision
        ) ** 2.0

    brackets = scipy.optimize.bracket(objective, xa=10, xb=2 * sigma, args=(sigma,))
    kcut_opt = scipy.optimize.bisect(
        lambda x, sigma: estimate_error_kinetic_energy(x, sigma) - target_precision,
        brackets[0],
        brackets[2],
        args=(sigma,),
    )
    # Factor of 1/2 because we consider range [-nmax/2, nmax/2]
    # k_cut_opt = 0.5 * (2*Ecut**2.0)**0.5
    # ecut = 0.5 * (2 kcut_opt)**2
    return 0.5 * (2 * kcut_opt) ** 2.0


def get_ngmax(ecut: float, box_length: float) -> int:
    r"""Get max value for n (wavenumber index) given cutoff in Hartree.

    Given $k = 2 \pi n / L$ find nmax given Ecut assuming a spherical kinetic energy cutoff.

    $$
        1/2 k_{cut}^2 = E_cut \rightarrow n_{max} = \sqrt{(2 E_cut)}
    $$
    Args:
        ecut: Spherical kinetic energy cutoff.
        box_length: $L$ in bohr.

    Returns:
        nmax: Maximum integer associated with $k_{cut}$ (i.e. $n_{max}$ in the docstring.)
    """
    ng_max = math.ceil(np.sqrt(2 * ecut) / (2 * np.pi / box_length))
    return ng_max


def _build_gaussian(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
    mu: Union[np.ndarray, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    This function builds the unnormalized amplitudes associated with a Gaussian wavepacket

    Args:
        ecut_hartree: Cutoff energy in Hartree. This defines the k-grid size for
            the wavepacket. See get_ngmax.
        box_length: The boxlength in bohr.
        sigma: The reciprocal space wavepacket standard deviation. sigma^2 =
            variance.
        ndim: The number of dimensions.
        mu: mean value for the Gaussian.

    Returns:
        gaussian: The un-normalized wavefunction amplitudes defined above.
        kgrid: The kspace grid the Gaussian is evaluated on.
    """
    if mu is None:
        mu = np.zeros(
            ndim,
        )
    nmax = get_ngmax(ecut_hartree, box_length)
    limit = nmax // 2
    grid = (np.arange(-limit, limit + 1, dtype=int),) * ndim
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
    """Helper function to compute Gaussian kinetic energy."""
    kinetic_energy = np.sum(
        0.5 * np.sum((kgrid - kproj[None, :]) ** 2.0, axis=-1) * p_k
    )
    return kinetic_energy


def kinetic_energy(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
    kproj: Union[np.ndarray, None] = None,
) -> float:
    r"""Compute Gaussian Kinetic Energy for a discrete Gaussian wavepacket.

    Args:
        ecut_hartree: Cutoff energy in Hartree. This defines the k-grid size for
            the wavepacket. See get_ngmax.
        box_length: The boxlength in bohr.
        sigma: The reciprocal space wavepacket standard deviation. sigma^2 =
            variance.
        ndim: The number of dimensions.
        kproj: Kinetic energy of the projectile.

    Returns:
        ke: The kinetic energy..
    """
    if kproj is None:
        kproj = np.zeros(
            ndim,
        )
    assert kproj.shape[0] == ndim
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
    num_samples: int = 1_000,
    kproj: Union[np.ndarray, None] = None,
    shift_by_constant=False,
) -> Tuple[float, float]:
    r"""Compute Gaussian kinetic energy for a discrete Gaussian wavepacket
    through sampling the discrete probability distibution.

    Args:
        ecut_hartree: Cutoff energy in Hartree. This defines the k-grid size for
            the wavepacket. See get_ngmax.
        box_length: The boxlength in bohr.
        sigma: The reciprocal space wavepacket standard deviation. sigma^2 =
            variance.
        ndim: The number of dimensions.
        num_samples: Number of samples to use.
        kproj: Kinetic energy of the projectile.
        shift_by_constant: Subtract known kproj mean value term from the result.

    Returns:
        ke: The kinetic energy..
        ke_err: The standard error in the mean of the kinetic energy.
    """
    if kproj is None:
        kproj = np.zeros(
            ndim,
        )
    # Builds |psi> = N^{-1/2} e^{-k^2/(4sigma^2)}
    gaussian, kgrid = _build_gaussian(ecut_hartree, box_length, sigma, ndim=ndim)
    # Need to square to get a probabilty distribution.
    norm = np.sum(gaussian**2.0)
    # Choose k-points based on the Guassian wavepacket.
    indx_k = np.random.choice(
        np.arange(len(kgrid)), num_samples, p=gaussian**2.0 / norm
    )
    k_select = kgrid[indx_k]
    kinetic_samples = 0.5 * np.sum((k_select + kproj[None, :]) ** 2.0, axis=-1)
    if shift_by_constant:
        kinetic_samples -= 0.5 * np.dot(kproj, kproj)
    if num_samples == 1:
        return kinetic_samples[0], 0.0
    else:
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
    r"""Compute Gaussian kinetic energy for a discrete Gaussian wavepacket
    through sampling the discrete probability distibution. Accepts an importance
    function. Mostly for playing around.

    Args:
        ecut_hartree: Cutoff energy in Hartree. This defines the k-grid size for
            the wavepacket. See get_ngmax.
        box_length: The boxlength in bohr.
        sigma: The reciprocal space wavepacket standard deviation. sigma^2 =
            variance.
        importance_function: Values of the user provided importance function
        evaluated on the same grid as the discrete gaussian wavefpacket.
        ndim: The number of dimensions.
        num_samples: Number of samples to use.
        kproj: Kinetic energy of the projectile.
        shift_by_constant: Subtract known kproj mean value term from the result.

    Returns:
        ke: The kinetic energy..
        ke_err: The standard error in the mean of the kinetic energy.
    """
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
