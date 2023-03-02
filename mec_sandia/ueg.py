from __future__ import annotations
import itertools
import math
from dataclasses import dataclass

import numpy as np


def calc_fermi_energy(rs: float) -> float:
    """Fermi energy of unpolarised free Fermi gas.
    
    :param rs: Wigner-Seitz radius.
    :returns EF: Fermi energy.
    """
    return 0.5 * (9.0 * np.pi / 4.0) ** (2.0 / 3.0) * rs ** (-2.0)


def calc_beta_from_theta(theta: float, rs: float) -> float:
    """Compute beta = 1 / T from theta = T / TF. TF = Fermi temperature

    :param theta: T/TF.
    :param rs: Wigner-Seitz radius.
    :returns beta: Inverse temperature in au.
    """
    ef = calc_fermi_energy(rs)
    T = ef * theta
    return 1.0 / T


def calc_theta_from_beta(beta: float, rs: float) -> float:
    """Compute theta = T / TF from beta = 1 / T, where TF = Fermi temperature

    :param theta: T/TF.
    :param rs: Wigner-Seitz radius.
    :returns beta: Inverse temperature in au.
    """
    ef = calc_fermi_energy(rs)
    T = 1.0 / beta
    theta = T / ef
    return theta


@dataclass
class UEG:
    """Basic finite-sized 3D uniform electron gas (free fermions) system class

    :param rs: Wigner-Seitz radius
    :param num_elec: Number of electrons
    :param box_length: Box length L.
    :param volum: Box Volume.
    :param eigenvalues: Single-particle eigenvalues (1/2 k^2).
    :param cutoff: Dimensionless kinetic energy cutoff in units of (2 pi / L)^2.
    """
    rs: float
    num_elec: int
    box_length: float
    volume: float
    eigenvalues: np.ndarray
    cutoff: float

    @staticmethod
    def build(num_elec: int, rs: float, cutoff: float) -> UEG:
        """Build 3D UEG helper class."""
        volume = (rs**3.0) * (4.0 * np.pi * num_elec / 3)
        box_length = volume ** (1.0 / 3.0)
        nmax = int(math.ceil(np.sqrt((2 * cutoff))))
        spval = []
        factor = 2 * np.pi / box_length
        for ni, nj, nk in itertools.product(range(-nmax, nmax + 1), repeat=3):
            spe = 0.5 * (ni**2 + nj**2 + nk**2)
            if spe <= cutoff:
                # Reintroduce 2 \pi / L factor.
                spval.append(spe * factor**2.0)

        # Sort the arrays in terms of increasing energy.
        eigs = np.sort(np.array(spval))

        return UEG(
            rs=rs,
            num_elec=num_elec,
            box_length=box_length,
            volume=volume,
            eigenvalues=eigs,
            cutoff=cutoff,
        )

    @property
    def fermi_energy(self) -> float:
        """Unpolarized Fermi gas"""
        return calc_fermi_energy(self.rs)

    def calc_beta_from_theta(self, theta: float) -> float:
        """Compute (inverse) temperature from target reduced temperature thetat = T/TF where TF is the Fermi temperature"""
        return calc_beta_from_theta(theta, self.rs)
