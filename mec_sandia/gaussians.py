from matplotlib.backend_managers import ToolManagerMessageEvent
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
    return 0.5*x0**2.0


def estimate_energy_cutoff_sum(target_precision, sigma):
    return 4 * estimate_energy_cutoff(target_precision, sigma)


def get_ngmax(ecut, box_length):
    ng_max = math.ceil(np.sqrt(2 * ecut) / (2 * np.pi / box_length))
    return ng_max


def discrete_gaussian_wavepacket(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    assert ndim > 0
    nmax = get_ngmax(ecut_hartree, box_length)
    grid = (np.arange(-nmax / 2, nmax / 2, dtype=int),) * ndim
    grid_spacing = 2 * np.pi / box_length
    kgrid = grid_spacing * cartesian_prod(grid)
    gaussian = np.exp(-(np.sum(kgrid**2.0, axis=-1)) / (2 * sigma**2.0))
    normalization = np.sum(gaussian)
    return gaussian / normalization, kgrid, normalization


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
    kinetic_energy = np.sum(
        0.5 * np.sum((kgrid + kproj[None, :]) ** 2.0, axis=-1) * gaussian
    )
    return kinetic_energy
