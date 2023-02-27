import itertools
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class UEG:
    rs: float
    num_elec: int
    box_length: float
    volume: float
    eigenvalues: np.array 
    cutoff: float

    @staticmethod
    def build(num_elec: int, rs: float, cutoff: float):
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
        return 0.5 * (9.0*np.pi/4.0)**(2.0/3.0) * self.rs**(-2.0)

    def calc_beta_from_theta(self, theta) -> float:
        """Compute (inverse) temperature from target reduced temperature thetat = T/TF where TF is the Fermi temperature"""
        ef = self.fermi_energy
        T = ef * theta

        return 1.0 / T