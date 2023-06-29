from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy


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
    gvecs: np.ndarray
    cutoff: float

    @staticmethod
    def build(num_elec: int, rs: float, cutoff: float) -> UEG:
        """Build 3D UEG helper class."""
        volume = (rs**3.0) * (4.0 * np.pi * num_elec / 3)
        box_length = volume ** (1.0 / 3.0)
        nmax = int(math.ceil(np.sqrt((2 * cutoff))))
        spval = []
        gvecs = []
        factor = 2 * np.pi / box_length
        for ni, nj, nk in itertools.product(range(-nmax, nmax + 1), repeat=3):
            spe = 0.5 * (ni**2 + nj**2 + nk**2)
            if spe <= cutoff:
                # Reintroduce 2 \pi / L factor.
                spval.append(spe * factor**2.0)
                gvecs.append([ni, nj, nk])

        # Sort the arrays in terms of increasing energy.
        ix = np.argsort(spval)
        eigs = np.array(spval)[ix]
        gvecs_sorted = np.array(gvecs)[ix]

        return UEG(
            rs=rs,
            num_elec=num_elec,
            box_length=box_length,
            volume=volume,
            eigenvalues=eigs,
            gvecs=gvecs_sorted,
            cutoff=cutoff,
        )


# pruned from pauxy / ipie
class UEGTMP:
    def __init__(self, nelec: Tuple[int, int], rs: float, ecut: float):
        # core energy
        nel = sum(nelec)
        # Box Length.
        self.rs = rs
        self.nelec = nel
        self.L = self.rs * (4.0 * nel * math.pi / 3.0) ** (1 / 3.0)
        # Volume
        self.vol = self.L**3.0
        # k-space grid spacing.
        # self.kfac = 2*math.pi/self.L
        self.kfac = 2 * math.pi / self.L
        self.ecore = 0.5 * nel * self.madelung(nel, rs)
        self.ecut = ecut

        self.vol = self.L**3
        # Single particle eigenvalues and corresponding kvectors
        (self.sp_eigv, self.basis, self.nmax) = self.sp_energies(self.kfac, self.ecut)

        self.nbasis = len(self.sp_eigv)
        # Allowed momentum transfers (4*ecut)
        (eigs, qvecs, self.qnmax) = self.sp_energies(self.kfac, 4 * self.ecut)
        # Omit Q = 0 term.
        self.qvecs = np.copy(qvecs[1:])
        self.vqvec = np.array([self.vq(self.kfac * q) for q in self.qvecs])
        # Number of momentum transfer vectors / auxiliary fields.
        # Can reduce by symmetry but be stupid for the moment.
        self.nchol = len(self.qvecs)

        self.imax_sq = np.dot(self.basis[-1], self.basis[-1])
        self.shifted_nmax = 2 * self.nmax
        self.create_lookup_table()
        for i, k in enumerate(self.basis):
            assert i == self.lookup_basis(k)

        nlimit = max(nelec)
        self.ikpq_i = []
        self.ikpq_kpq = []
        for iq, q in enumerate(self.qvecs):
            idxkpq_list_i = []
            idxkpq_list_kpq = []
            for i, k in enumerate(self.basis[0:nlimit]):
                kpq = k + q
                idxkpq = self.lookup_basis(kpq)
                if idxkpq is not None:
                    idxkpq_list_i += [i]
                    idxkpq_list_kpq += [idxkpq]
            self.ikpq_i += [idxkpq_list_i]
            self.ikpq_kpq += [idxkpq_list_kpq]

        self.ipmq_i = []
        self.ipmq_pmq = []
        for iq, q in enumerate(self.qvecs):
            idxpmq_list_i = []
            idxpmq_list_pmq = []
            for i, p in enumerate(self.basis[0:nlimit]):
                pmq = p - q
                idxpmq = self.lookup_basis(pmq)
                if idxpmq is not None:
                    idxpmq_list_i += [i]
                    idxpmq_list_pmq += [idxpmq]
            self.ipmq_i += [idxpmq_list_i]
            self.ipmq_pmq += [idxpmq_list_pmq]

        for iq, q in enumerate(self.qvecs):
            self.ikpq_i[iq] = np.array(self.ikpq_i[iq], dtype=np.int64)
            self.ikpq_kpq[iq] = np.array(self.ikpq_kpq[iq], dtype=np.int64)
            self.ipmq_i[iq] = np.array(self.ipmq_i[iq], dtype=np.int64)
            self.ipmq_pmq[iq] = np.array(self.ipmq_pmq[iq], dtype=np.int64)

        (self.chol_vecs, self.iA, self.iB) = self.two_body_potentials_incore()

    def sp_energies(self, kfac, ecut):
        # Scaled Units to match with HANDE.
        # So ecut is measured in units of 1/kfac^2.
        nmax = int(math.ceil(np.sqrt((2 * ecut))))

        spval = []
        vec = []
        kval = []

        for ni in range(-nmax, nmax + 1):
            for nj in range(-nmax, nmax + 1):
                for nk in range(-nmax, nmax + 1):
                    spe = 0.5 * (ni**2 + nj**2 + nk**2)
                    if spe <= ecut:
                        kijk = [ni, nj, nk]
                        kval.append(kijk)
                        # Reintroduce 2 \pi / L factor.
                        ek = 0.5 * np.dot(np.array(kijk), np.array(kijk))
                        spval.append(kfac**2 * ek)

        # Sort the arrays in terms of increasing energy.
        spval = np.array(spval)
        ix = np.argsort(spval, kind="mergesort")
        spval = spval[ix]
        kval = np.array(kval)[ix]

        return (spval, kval, nmax)

    def create_lookup_table(self):
        basis_ix = []
        for k in self.basis:
            basis_ix.append(self.map_basis_to_index(k))
        self.lookup = np.zeros(max(basis_ix) + 1, dtype=int)
        for i, b in enumerate(basis_ix):
            self.lookup[b] = i
        self.max_ix = max(basis_ix)

    def lookup_basis(self, vec):
        if np.dot(vec, vec) <= self.imax_sq:
            ix = self.map_basis_to_index(vec)
            if ix >= len(self.lookup):
                ib = None
            else:
                ib = self.lookup[ix]
            return ib
        else:
            ib = None

    def map_basis_to_index(self, k):
        return (
            (k[0] + self.nmax)
            + self.shifted_nmax * (k[1] + self.nmax)
            + self.shifted_nmax * self.shifted_nmax * (k[2] + self.nmax)
        )

    def madelung(self, nel, rs):
        """Use expression in Schoof et al. (PhysRevLett.115.130402) for the
        Madelung contribution to the total energy fitted to L.M. Fraser et al.
        Phys. Rev. B 53, 1814.
        Parameters
        ----------
        rs : float
            Wigner-Seitz radius.
        ne : int
            Number of electrons.
        Returns
        -------
        v_M: float
            Madelung potential (in Hartrees).
        """
        c1 = -2.837297
        c2 = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        return c1 * c2 / (nel ** (1.0 / 3.0) * rs)

    def vq(self, q):
        """The typical 3D Coulomb kernel
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        v_M: float
            3D Coulomb kernel (in Hartrees)
        """
        return 4 * math.pi / np.dot(q, q)

    def density_operator(self, iq):
        """Density operator as defined in Eq.(6) of PRB(75)245123
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        rho_q: float
            density operator
        """
        nnz = self.rho_ikpq_kpq[iq].shape[0]  # Number of non-zeros
        ones = np.ones((nnz), dtype=np.complex128)
        rho_q = scipy.sparse.csc_matrix(
            (ones, (self.rho_ikpq_kpq[iq], self.rho_ikpq_i[iq])),
            shape=(self.nbasis, self.nbasis),
            dtype=np.complex128,
        )
        return rho_q

    def scaled_density_operator_incore(self, transpose):
        """Density operator as defined in Eq.(6) of PRB(75)245123
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        rho_q: float
            density operator
        """
        rho_ikpq_i = []
        rho_ikpq_kpq = []
        for iq, q in enumerate(self.qvecs):
            idxkpq_list_i = []
            idxkpq_list_kpq = []
            for i, k in enumerate(self.basis):
                kpq = k + q
                idxkpq = self.lookup_basis(kpq)
                if idxkpq is not None:
                    idxkpq_list_i += [i]
                    idxkpq_list_kpq += [idxkpq]
            rho_ikpq_i += [idxkpq_list_i]
            rho_ikpq_kpq += [idxkpq_list_kpq]

        for iq, q in enumerate(self.qvecs):
            rho_ikpq_i[iq] = np.array(rho_ikpq_i[iq], dtype=np.int64)
            rho_ikpq_kpq[iq] = np.array(rho_ikpq_kpq[iq], dtype=np.int64)

        nq = len(self.qvecs)
        nnz = 0
        for iq in range(nq):
            nnz += rho_ikpq_kpq[iq].shape[0]

        col_index = []
        row_index = []

        values = []

        if transpose:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
                piovol = math.pi / (self.vol)
                factor = (piovol / np.dot(qscaled, qscaled)) ** 0.5

                for innz, kpq in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [
                        rho_ikpq_kpq[iq][innz] + rho_ikpq_i[iq][innz] * self.nbasis
                    ]
                    col_index += [iq]
                    values += [factor]
        else:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
                piovol = math.pi / (self.vol)
                factor = (piovol / np.dot(qscaled, qscaled)) ** 0.5

                for innz, kpq in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [
                        rho_ikpq_kpq[iq][innz] * self.nbasis + rho_ikpq_i[iq][innz]
                    ]
                    col_index += [iq]
                    values += [factor]

        rho_q = scipy.sparse.csc_matrix(
            (values, (row_index, col_index)),
            shape=(self.nbasis * self.nbasis, nq),
            dtype=np.complex128,
        )

        return rho_q

    def two_body_potentials_incore(self):
        """Calculatate A and B of Eq.(13) of PRB(75)245123 for a given plane-wave vector q
        Parameters
        ----------
        system :
            system class
        q : float
            a plane-wave vector
        Returns
        -------
        iA : np array
            Eq.(13a)
        iB : np array
            Eq.(13b)
        """
        # qscaled = self.kfac * self.qvecs

        # # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol

        rho_q = self.scaled_density_operator_incore(False)
        rho_qH = self.scaled_density_operator_incore(True)

        iA = 1j * (rho_q + rho_qH)
        iB = -(rho_q - rho_qH)

        return (rho_q, iA, iB)

    def hijkl(self, i, j, k, l):
        """Compute <ij|kl> = (ik|jl) = 1/Omega * 4pi/(kk-ki)**2

        Checks for momentum conservation k_i + k_j = k_k + k_k, or
        k_k - k_i = k_j - k_l.

        Parameters
        ----------
        i, j, k, l : int
            Orbital indices for integral (ik|jl) = <ij|kl>.

        Returns
        -------
        integral : float
            (ik|jl)
        """
        q1 = self.basis[k] - self.basis[i]
        q2 = self.basis[j] - self.basis[l]
        if np.dot(q1, q1) > 1e-12 and np.dot(q1 - q2, q1 - q2) < 1e-12:
            return 1.0 / self.vol * self.vq(self.kfac * q1)
        else:
            return 0.0

    def compute_real_transformation(self):
        U22 = np.zeros((2, 2), dtype=np.complex128)
        U22[0, 0] = 1.0 / np.sqrt(2.0)
        U22[0, 1] = 1.0 / np.sqrt(2.0)
        U22[1, 0] = -1.0j / np.sqrt(2.0)
        U22[1, 1] = 1.0j / np.sqrt(2.0)

        U = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)

        for i, b in enumerate(self.basis):
            if np.sum(b * b) == 0:
                U[i, i] = 1.0
            else:
                mb = -b
                diff = np.einsum("ij->i", (self.basis - mb) ** 2)
                idx = np.argwhere(diff == 0)
                assert idx.ravel().shape[0] == 1
                if i < idx:
                    idx = idx.ravel()[0]
                    U[i, i] = U22[0, 0]
                    U[i, idx] = U22[0, 1]
                    U[idx, i] = U22[1, 0]
                    U[idx, idx] = U22[1, 1]
                else:
                    continue

        U = U.T.copy()
        return U

    def eri_4(self):
        eri_chol = 4 * self.chol_vecs.dot(self.chol_vecs.T)
        eri_chol = (
            eri_chol.toarray()
            .reshape((self.nbasis, self.nbasis, self.nbasis, self.nbasis))
            .real
        )
        eri_chol = eri_chol.transpose(0, 1, 3, 2)
        return eri_chol

    # Compute 8-fold symmetric integrals. Useful for running standard quantum chemistry methods
    def eri_8(self):
        eri = self.eri_4()
        U = self.compute_real_transformation()

        eri0 = np.einsum("mp,mnls->pnls", U.conj(), eri, optimize=True)
        eri1 = np.einsum("nq,pnls->pqls", U, eri0, optimize=True)
        eri2 = np.einsum("lr,pqls->pqrs", U.conj(), eri1, optimize=True)
        eri3 = np.einsum("st,pqrs->pqrt", U, eri2, optimize=True).real

        return eri3
