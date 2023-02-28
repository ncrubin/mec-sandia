from dataclasses import dataclass
import itertools
import numpy as np
from typing import List, Tuple, Union

import scipy.optimize


def fermi_factor(ek, mu, beta):
    return 1.0 / (np.exp(beta * (ek - mu)) + 1)


def compute_electron_number(mu, eigs, beta):
    return 2 * sum(fermi_factor(eigs, mu, beta))


def cost_function(mu, eigs, beta, target_electron_number):
    return compute_electron_number(mu, eigs, beta) - target_electron_number


def find_chemical_potential(eigs, beta, target_num_elec, mu0=0.0):
    r"""Find solution of :math:`<N> = \sum_i f(e_i, mu)`"""
    return scipy.optimize.fsolve(
        cost_function, mu0, args=(eigs, beta, target_num_elec)
    )[0]


@dataclass
class DensityMatrix:
    num_spin_orbs: int
    occ_strings: List[np.ndarray]
    weights: np.ndarray

    @staticmethod
    def build_grand_canonical(fermi_occupations: np.ndarray, num_samples: int):
        occ_string = []
        # Use spin orbitals abab ordering
        num_spin_orbs = 2 * len(fermi_occupations)
        orb_indices = np.arange(num_spin_orbs, dtype=np.int32)
        random_numbers = np.random.random(size=num_samples * num_spin_orbs).reshape(
            (num_samples, num_spin_orbs)
        )
        fermi_spin_orb = np.zeros(num_spin_orbs)
        fermi_spin_orb[::2] = fermi_occupations
        fermi_spin_orb[1::2] = fermi_occupations
        for isample in range(num_samples):
            pi = random_numbers[isample]
            occ_str = orb_indices[np.where(fermi_spin_orb > pi)]
            occ_string.append(occ_str)

        return DensityMatrix(
            num_spin_orbs=num_spin_orbs,
            occ_strings=occ_string,
            weights=np.ones(num_samples),
        )

    @staticmethod
    def build_canonical(fermi_occupations: np.ndarray, num_samples: int, target_num_elec: int):
        occ_string = []
        # Use spin orbitals abab ordering
        num_spin_orbs = 2 * len(fermi_occupations)
        orb_indices = np.arange(num_spin_orbs, dtype=np.int32)
        random_numbers = np.random.random(size=num_samples * num_spin_orbs).reshape(
            (num_samples, num_spin_orbs)
        )
        fermi_spin_orb = np.zeros(num_spin_orbs)
        fermi_spin_orb[::2] = fermi_occupations
        fermi_spin_orb[1::2] = fermi_occupations
        for isample in range(num_samples):
            pi = random_numbers[isample]
            occ_str = orb_indices[np.where(fermi_spin_orb > pi)]
            if len(occ_str) == target_num_elec:
                occ_string.append(occ_str)

        return DensityMatrix(
            num_spin_orbs=num_spin_orbs,
            occ_strings=occ_string,
            weights=np.ones(num_samples),
        )

    @staticmethod
    def build_grand_canonical_exact(eigenvalues: np.ndarray, mu: float, beta: float):
        occ_string = []
        weights = []
        # Use spin orbitals abab ordering
        num_spin_orbs = 2 * len(eigenvalues)
        orb_indices = np.arange(num_spin_orbs, dtype=np.int32)
        spin_eig = np.zeros(num_spin_orbs)
        spin_eig[::2] = eigenvalues
        spin_eig[1::2] = eigenvalues
        Zgc = 0
        for nelec in range(0, num_spin_orbs+1):
            for occ_str in itertools.combinations(range(0, num_spin_orbs), nelec):
                boltzmann = np.exp(-beta*(sum(spin_eig[list(occ_str)])-mu*nelec))
                Zgc += boltzmann
                weights.append(np.array(boltzmann))
                occ_string.append(np.array(occ_str))

        return DensityMatrix(
            num_spin_orbs=num_spin_orbs,
            occ_strings=occ_string,
            weights=len(occ_string)*np.array(weights)/Zgc,
        )

    @staticmethod
    def build_canonical_exact(eigenvalues: np.ndarray, beta: float, num_elec: int):
        occ_string = []
        weights = []
        # Use spin orbitals abab ordering
        num_spin_orbs = 2 * len(eigenvalues)
        orb_indices = np.arange(num_spin_orbs, dtype=np.int32)
        spin_eig = np.zeros(num_spin_orbs)
        spin_eig[::2] = eigenvalues
        spin_eig[1::2] = eigenvalues
        Z = 0
        for occ_str in itertools.combinations(range(0, num_spin_orbs), num_elec):
            boltzmann = np.exp(-beta*(sum(spin_eig[list(occ_str)])))
            Z += boltzmann
            weights.append(np.array(boltzmann))
            occ_string.append(np.array(occ_str))

        return DensityMatrix(
            num_spin_orbs=num_spin_orbs,
            occ_strings=occ_string,
            weights=len(occ_string)*np.array(weights)/Z,
        )

    @property
    def num_samples(self) -> int:
        return len(self.occ_strings)

    def compute_occupations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute average electron number from statistical sample of density matrix"""
        occupations = np.zeros((self.num_samples, self.num_spin_orbs))
        for isample in range(self.num_samples):
            if len(self.occ_strings[isample]) == 0:
                continue
            occupations[isample, self.occ_strings[isample]] += (
                1.0 * self.weights[isample]
            )
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
            if len(self.occ_strings[isample]) == 0:
                continue
            one_body[isample] += sum(
                matrix_elements[self.occ_strings[isample] // 2] * self.weights[isample]
            )
        return float(np.mean(one_body)), np.std(one_body, ddof=1) / (
            self.num_samples**0.5
        )

    def compute_electron_number(self) -> Tuple[float, float]:
        """Compute average electron number from statistical sample of density matrix"""
        nav = [
            len(occ_str) * weight
            for occ_str, weight in zip(self.occ_strings, self.weights)
        ]
        return float(np.mean(nav)), np.std(nav, ddof=1) / (self.num_samples**0.5)