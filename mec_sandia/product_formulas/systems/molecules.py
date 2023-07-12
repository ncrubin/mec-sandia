import os
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'

import numpy as np

from pyscf import gto, scf, ao2mo

from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

from mec_sandia.product_formulas.pyscf_utility import compute_integrals

def heh_molecule():
    mol = gto.M()
    mol.atom = [['He', 0, 0, 0], ['H', 0, 0, 1.4]]
    mol.charge = +1
    mol.basis = 'sto-6g'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    obi, tbi = compute_integrals(mf.mol, mf)
    ecore = mf.energy_nuc()
    norb = obi.shape[0]

    nmol = gto.M()
    nmol.nelectron = mol.nelectron
    nmf = scf.RHF(mol)
    nmf.mo_coeff = np.eye(norb)
    nmf.get_hcore = lambda *args: obi
    nmf.get_ovlp = lambda *args: np.eye(norb)
    nmf._eri = tbi.transpose((0, 3, 1, 2))
    nmf.energy_nuc = lambda *args: ecore
    nmf.kernel()

    # check if integrals in chem ordering
    two_electron_compressed = ao2mo.kernel(mf.mol,
                                           mf.mo_coeff)
    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, norb)
    assert np.allclose(two_electron_integrals, nmf._eri)


    return nmf


def lih_molecule(basis='sto-6g'):
    mol = gto.M()
    mol.atom = [['Li', 0, 0, 0], ['H', 0, 0, 1.4]]
    mol.basis = basis
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    obi, tbi = compute_integrals(mf.mol, mf)
    ecore = mf.energy_nuc()
    norb = obi.shape[0]

    nmol = gto.M()
    nmol.nelectron = mol.nelectron
    nmf = scf.RHF(mol)
    nmf.mo_coeff = np.eye(norb)
    nmf.get_hcore = lambda *args: obi
    nmf.get_ovlp = lambda *args: np.eye(norb)
    nmf._eri = tbi.transpose((0, 3, 1, 2))
    nmf.energy_nuc = lambda *args: ecore
    nmf.kernel()

    # check if integrals in chem ordering
    two_electron_compressed = ao2mo.kernel(mf.mol,
                                           mf.mo_coeff)
    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, norb)
    assert np.allclose(two_electron_integrals, nmf._eri)


    return nmf

