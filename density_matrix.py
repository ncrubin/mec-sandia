from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Union

import scipy.optimize


def fermi_factor(ek, mu, beta):
    return 1.0 / (np.exp(beta * (ek - mu)) + 1)


def compute_electron_number(mu, eigs, beta):
    return sum(fermi_factor(eigs, mu, beta))


def cost_function(mu, eigs, beta, target_electron_number):
    return compute_electron_number(mu, eigs, beta) - target_electron_number


def find_chemical_potential(eigs, beta, target_num_elec, mu0=0.0):
    r"""Find solution of :math:`<N> = \sum_i f(e_i, mu)`"""
    return scipy.optimize.fsolve(
        cost_function, mu0, args=(eigs, beta, target_num_elec)
    )[0]


@dataclass
class DensityMatrix:
    num_basis: int
    occ_strings: List[np.ndarray]
    weights: np.ndarray

    @staticmethod
    def build_grand_canonical(fermi_occupations: np.ndarray, num_samples: int):
        occ_string = []
        orb_indices = np.arange(len(fermi_occupations), dtype=np.int32)
        random_numbers = np.random.random(
            size=num_samples * len(fermi_occupations)
        ).reshape((num_samples, len(fermi_occupations)))
        for isample in range(num_samples):
            pi = random_numbers[isample]
            occ_str = orb_indices[np.where(fermi_occupations > pi)]
            occ_string.append(occ_str)

        return DensityMatrix(
            num_basis=len(fermi_occupations),
            occ_strings=occ_string,
            weights=np.ones(num_samples),
        )

    @property
    def num_samples(self) -> int:
        return len(self.occ_strings)

    def compute_occupations(self) -> Tuple[float, float]:
        """Compute average electron number from statistical sample of density matrix"""
        occupations = np.zeros((self.num_samples, self.num_basis))
        for isample in range(self.num_samples):
            occupations[isample, self.occ_strings[isample]] += 1.0 * self.weights[isample]
        return (
            np.mean(occupations, axis=0),
            np.std(occupations, ddof=1, axis=0) / (self.num_samples**0.5),
        )

    def contract_diagonal_one_body(
        self, matrix_elements: np.ndarray
    ) -> Tuple[float, float]:
        """Compute average electron number from statistical sample of density matrix"""
        one_body = np.zeros((self.num_samples,))
        for isample in range(self.num_samples):
            one_body[isample] += sum(matrix_elements[self.occ_strings[isample]]*self.weights[isample])
        return float(np.mean(one_body)), np.std(one_body, ddof=1) / (
            self.num_samples**0.5
        )

    def compute_electron_number(self) -> Tuple[float, float]:
        """Compute average electron number from statistical sample of density matrix"""
        nav = [len(occ_str)*weights for occ_str, weight in zip(self.occ_strings, self.weights)]
        return float(np.mean(nav)), np.std(nav, ddof=1) / (self.num_samples**0.5)
