import numpy as np
from typing import Tuple, Union
import math
from pyscf.lib.numpy_helper import cartesian_prod

def estimate_error_kinetic_energy(kcut: float, sigma: float) -> float:
    a = kcut 
    b = (2*sigma**2.0)
    t1 = 2 * a * np.exp(-a**2.0/b)
    t2 = np.sqrt(np.pi*b) * scipy.special.erfc(a/(b**0.5))
    return 0.25*b*(t1 + t2)

def estimate_cutoff():
    pass

def get_ngmax(ecut, box_length):
    ng_max = math.ceil(np.sqrt(2 * ecut) / (2 * np.pi / box_length))
    return ng_max


def discrete_gaussian_wavepacket(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int=1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    assert ndim > 0
    nmax = get_ngmax(ecut_hartree, box_length)
    grid = (np.arange(-nmax / 2, nmax / 2),)* ndim
    grid_spacing = 2 * np.pi / box_length
    kgrid = grid_spacing * cartesian_prod(grid)
    gaussian = np.exp(-(np.sum(kgrid**2.0, axis=-1)) / (2 * sigma**2.0))
    normalization = np.sum(gaussian)
    return gaussian / normalization, kgrid, normalization


def kinetic_energy(
    ecut_hartree: float,
    box_length: float,
    sigma: float,
    ndim: int=1,
    kproj: Union[np.ndarray, None]=None 
) -> float:
    if kproj is None:
        kproj = np.zeros(ndim,)
    gaussian, kgrid, norm = discrete_gaussian_wavepacket(ecut_hartree, box_length, sigma, ndim=ndim)
    kinetic_energy = np.sum(0.5 * np.sum((kgrid + kproj[None,:])**2.0, axis=-1) * gaussian)
    return kinetic_energy