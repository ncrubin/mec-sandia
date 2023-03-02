from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from typing import List, Tuple, Union

import scipy.optimize


def fermi_factor(
    ek: Union[float, np.ndarray],
    mu: float,
    beta: float,
) -> Union[float, np.ndarray]:
    """Fermi factor

    :param ek: eigenvalue(s) to compute Fermi factor for.
    :param mu: chemical potential.
    :param beta: inverse temperature.

    :returns f(ek): Fermi factor
    """
    return 1.0 / (np.exp(beta * (ek - mu)) + 1)


def compute_electron_number(
    mu: float,
    eigs: np.ndarray,
    beta: float,
) -> float:
    """Compute average electron number

    :param mu: chemical potential.
    :param eigs: eigenvalues to compute Fermi factor for.
    :param beta: inverse temperature.

    :returns <N>_0: Average number of electrons. 
    """
    return 2 * sum(fermi_factor(eigs, mu, beta))


def _chem_pot_cost_function(
    mu: float,
    eigs: np.ndarray,
    beta: float,
    target_electron_number: int,
) -> float:
    """Cost function for finding chemical potential.

    :param mu: chemical potential.
    :param eigs: eigenvalues to compute Fermi factor for.
    :param beta: inverse temperature.
    :param target_electron_number: Target number of electrons.

    :returns <N>_0-target_num_electrons: cost function value. 
    """
    return compute_electron_number(mu, eigs, beta) - target_electron_number


def find_chemical_potential(
    eigs: np.ndarray,
    beta: float,
    target_num_elec: int,
    mu0=0.0,
) -> float:
    r"""Find solution of :math:`<N> = \sum_i f(e_i, mu)`

    :param eigs: eigenvalues to compute Fermi factor for.
    :param beta: inverse temperature.
    :param target_electron_number: Target number of electrons.
    :param mu0: Initial guess for chemical potential. May be useful at low T (~HOMO).
    """
    return scipy.optimize.fsolve(
        _chem_pot_cost_function, mu0, args=(eigs, beta, target_num_elec)
    )[0]


@dataclass
class DensityMatrix:
    """Small wrapper to represent a density matrix given a set of occupation strings, and weights.

    Write rho = sum_i w_i s_i, where s_i = list of occupied spin orbitals, and w_i is an optional weight.

    Note this is complete overkill as for sampling we never need to store the
    configurations, but it's helpful for testing.

    :param num_spin_orbs: Number of spin orbitals.
    :param occ_strings: List of occupation number strings representing (diagonal) density matrix.
    :param weights: Optional weights for density matrix.
    """
    num_spin_orbs: int
    occ_strings: List[np.ndarray]
    weights: np.ndarray

    @staticmethod
    def build_grand_canonical(
        fermi_occupations: np.ndarray, num_samples: int
    ) -> DensityMatrix:
        r"""Build statistitical representation of Grand Canonical density matrix for free fermions.

        rho = 1/Z sum_N sum_{occ_N} e^{-beta\sum_{i_occ} (eps_i-mu)} |occ><occ|

        Here we sample occupations with propability given by Boltzmann factors.
        This can be achieved by occupying orbitals with probability given by the
        fermi factors. This works because f_i gives the probability that orbital
        i is occupied in this ensemble and there is no correlation between
        orbitals so the probability of a given occupation is just the product of
        the orbital probabilities.

        Note in this case the occupation strings are drawn with probability
        proportional to the Boltzmann factor so no weights are necessary.
    
        :param fermi_occupations: Fermi factors for every orbital.
        :param num_samples: Number of samples to take when constructing density matrix.
        """
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
    def build_canonical(
        fermi_occupations: np.ndarray, num_samples: int, target_num_elec: int
    ) -> DensityMatrix:
        r"""Build statistitical representation of Canonical density matrix for free fermions.

        rho = 1/Z sum_{occ_N} e^{-beta\sum_{i_occ} (eps_i-mu)} |occ><occ|

        Here we sample occupations with propability given by Boltzmann factors.
        This can be achieved by occupying orbitals with probability given by the
        fermi factors and discarding states with the incorrect number of electrons.

        Note in this case the occupation strings are drawn with probability
        proportional to the Boltzmann factor so no weights are necessary.
    
        :param fermi_occupations: Fermi factors for every orbital.
        :param num_samples: Number of samples to take when constructing density matrix.
        """
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
    def build_grand_canonical_exact(
        eigenvalues: np.ndarray, mu: float, beta: float
    ) -> DensityMatrix:
        r"""Build exact representation of Grand Canonical density matrix for free fermions.

        rho = 1/Z sum_N sum_{occ_N} e^{-beta\sum_{i_occ} (eps_i-mu)} |occ><occ|

        Here literally enumerate all 2^N occupation strings and set the weights to be the Boltzmann factors.

        Naturally will only work for very small number of obitals. 

        :param eigenvalues: Single-particle eigenvalues.
        :param mu: Chemical potential.
        :param beta: Inverse temperature.
        """
        occ_string = []
        weights = []
        # Use spin orbitals abab ordering
        num_spin_orbs = 2 * len(eigenvalues)
        orb_indices = np.arange(num_spin_orbs, dtype=np.int32)
        spin_eig = np.zeros(num_spin_orbs)
        spin_eig[::2] = eigenvalues
        spin_eig[1::2] = eigenvalues
        Zgc = 0
        for nelec in range(0, num_spin_orbs + 1):
            for occ_str in itertools.combinations(range(0, num_spin_orbs), nelec):
                boltzmann = np.exp(-beta * (sum(spin_eig[list(occ_str)]) - mu * nelec))
                Zgc += boltzmann
                weights.append(np.array(boltzmann))
                occ_string.append(np.array(occ_str))

        # Note for computing average properties we slightly abusing weights here to include factors of dim(H).
        return DensityMatrix(
            num_spin_orbs=num_spin_orbs,
            occ_strings=occ_string,
            weights=len(occ_string) * np.array(weights) / Zgc,
        )

    @staticmethod
    def build_canonical_exact(
        eigenvalues: np.ndarray, beta: float, num_elec: int
    ) -> DensityMatrix:
        r"""Build exact representation of Canonical density matrix for free fermions.

        rho = 1/Z sum_{occ_N} e^{-beta\sum_{i_occ} eps_i} |occ><occ|

        Here literally enumerate all (N Choose eta) occupation strings and set
        the weights to be the Boltzmann factors.

        Naturally will only work for very small number of obitals. 

        :param eigenvalues: Single-particle eigenvalues.
        :param mu: Chemical potential.
        :param beta: Inverse temperature.
        """
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
            boltzmann = np.exp(-beta * (sum(spin_eig[list(occ_str)])))
            Z += boltzmann
            weights.append(np.array(boltzmann))
            occ_string.append(np.array(occ_str))

        # Note for computing average properties we slightly abusing weights here to include factors of dim(H).
        return DensityMatrix(
            num_spin_orbs=num_spin_orbs,
            occ_strings=occ_string,
            weights=len(occ_string) * np.array(weights) / Z,
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
        """Contact diagonal one-body operator with density matrix

        :param matrix_elements: 1D array of spatial orbital matrix elements of diagonal operator. 
        """
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
        """Compute average electron number from density matrix"""
        nav = [
            len(occ_str) * weight
            for occ_str, weight in zip(self.occ_strings, self.weights)
        ]
        return float(np.mean(nav)), np.std(nav, ddof=1) / (self.num_samples**0.5)

    def histogram_electron_counts(self) -> np.ndarray:
        """Build histrogram of electron numbers."""
        hist = np.zeros(self.num_spin_orbs)
        for isample in range(self.num_samples):
            hist[len(self.occ_strings[isample])] += 1
        return hist
