from dataclasses import dataclass
from sys import float_info
import numpy as np
from typing import Tuple, Union
import math
from pyscf.lib.numpy_helper import cartesian_prod
import scipy.optimize


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
    grid = (np.arange(-nmax / 2, nmax / 2, dtype=int),) * ndim
    grid_spacing = 2 * np.pi / box_length
    kgrid = grid_spacing * cartesian_prod(grid)
    gaussian = np.exp(
        -(np.sum((kgrid - mu[None, :]) ** 2.0, axis=-1)) / (2 * sigma**2.0)
    )
    return gaussian, kgrid


def discrete_gaussian_wavepacket(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    assert ndim > 0
    gaussian, kgrid = _build_gaussian(ecut_hartree, box_length, sigma, ndim=ndim)
    normalization = np.sum(gaussian)
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
    gaussian, kgrid, norm = discrete_gaussian_wavepacket(
        ecut_hartree, box_length, sigma, ndim=ndim
    )
    kinetic_energy = _calc_kinetic_energy(kgrid, kproj, gaussian)
    return kinetic_energy


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
    gaussian, kgrid = _build_gaussian(ecut_hartree, box_length, sigma, ndim=ndim)
    norm = np.sum(gaussian)
    indx_k = np.random.choice(np.arange(len(kgrid)), num_samples, p=gaussian / norm)
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
    p_x = gaussian / sum(gaussian)
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


def sigma_time(time, sigma, stopping_deriv, mass_proj):
    return sigma * (1 + stopping_deriv * time / mass_proj)


@dataclass
class StoppingPowerData:
    stopping: float
    stopping_err: float
    kinetic: np.ndarray
    kinetic_err: np.ndarray
    num_samples: int


def compute_stopping_power(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    time_values: np.ndarray,
    kproj_vals: np.ndarray,
    stopping_deriv: float,
    mass_proj: float,
    ndim: int = 1,
    num_samples: int = 10_000,
) -> StoppingPowerData:

    def _fit_linear(x, a, b):
        return a * x + b

    sigma_tvals = sigma_time(time_values, sigma, stopping_deriv, mass_proj)
    func = lambda x, k: estimate_kinetic_energy_sampling(
        ecut_hartree, box_length, x, ndim=3, num_samples=num_samples, kproj=k
    )
    values = [func(sigma_t, kproj) for (sigma_t, kproj) in zip(sigma_tvals, kproj_vals)]
    yvals, errs = zip(*values)
    yvals = np.array(yvals) / mass_proj
    errs = np.array(errs) / mass_proj
    popt, pcov = scipy.optimize.curve_fit(
        _fit_linear, time_values, yvals, sigma=errs, absolute_sigma=True
    )
    slope, incpt = popt
    slope_err = np.sqrt(pcov[0, 0])
    data = StoppingPowerData(
        stopping=slope,
        stopping_err=slope_err,
        kinetic=yvals,
        kinetic_err=errs,
        num_samples=num_samples,
    )
    return data
