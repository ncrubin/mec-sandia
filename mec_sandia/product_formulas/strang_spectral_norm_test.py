"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import copy
import openfermion as of
import numpy as np
from scipy.linalg import expm
import fqe
from pyscf import gto, scf, ao2mo
from pyscf.fci.cistring import make_strings
from openfermionpyscf._run_pyscf import compute_integrals
from openfermion import MolecularData
from openfermionpyscf import PyscfMolecularData

from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf_utility import get_spectrum, pyscf_to_fqe_wf
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from strang_spectral_norm import spectral_norm_power_method, spectral_norm_svd, spectral_norm_fqe_power_iteration


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

 

def power_method_test():
    np.random.seed(10)
    A = np.random.randn(64).reshape((8, 8))
    _, sigma, _ = np.linalg.svd(A)
    print(np.max(sigma))

    x = np.random.randn(8)
    prev_sqrt_lam_max = np.inf
    delta_sqrt_lam_max = np.inf
    while delta_sqrt_lam_max > 1.0E-8:
        r = A.T @ A @ x
        x = r / np.linalg.norm(r)
        sqrt_lam_max = np.sqrt(x.T @ A.T @ A @ x)
        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        print(f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}")
        prev_sqrt_lam_max = sqrt_lam_max


def test_cirq_spectral_norm():
    # mol_mf = lih_molecule()
    mol_mf = heh_molecule()
    nelec = mol_mf.mol.nelectron
    norb = mol_mf.mo_coeff.shape[0]
    sz = 0
    of_eris = mol_mf._eri.transpose((0, 2, 3, 1))
    fqe_ham = integrals_to_fqe_restricted(mol_mf.get_hcore(), of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((mol_mf.get_hcore(), ))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(mol_mf.get_hcore()), np.einsum('ijlk', -0.5 * of_eris)))
    roots, wfns = get_spectrum(mol_mf, num_roots=1)
    gs_e, gs_wfn = roots, pyscf_to_fqe_wf(wfns, pyscf_mf=mol_mf)
    assert np.allclose(gs_e, gs_wfn.expectationValue(fqe_ham).real + mol_mf.energy_nuc())
    assert np.allclose(gs_e, gs_wfn.expectationValue(fqe_ham_ob).real + gs_wfn.expectationValue(fqe_ham_tb).real + mol_mf.energy_nuc())

    np.random.seed(50)
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    x_cirq = fqe.to_cirq(x_wfn)

    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(mol_mf.get_hcore(), of_eris)
    molecular_hamiltonian = InteractionOperator(0, one_body_coefficients, 1 / 2 * two_body_coefficients)
    sparse_ham = of.get_sparse_operator(molecular_hamiltonian).todense()
    molecular_hamiltonian_ob = InteractionOperator(0, one_body_coefficients, np.zeros_like(1 / 2 * two_body_coefficients))
    molecular_hamiltonian_tb = InteractionOperator(0, np.zeros_like(one_body_coefficients), 1 / 2 * two_body_coefficients)

    sparse_ham_ob = of.get_sparse_operator(molecular_hamiltonian_ob).todense()
    sparse_ham_tb = of.get_sparse_operator(molecular_hamiltonian_tb).todense()

    cirq_wf = fqe.to_cirq(gs_wfn).reshape((-1, 1))
    cirq_e = (cirq_wf.conj().T @ sparse_ham @ cirq_wf + mol_mf.energy_nuc())[0, 0].real
    assert np.allclose(cirq_e, gs_e)

    t = 0.78
    exact_u = expm(-1j * t * sparse_ham)
    strang_u = expm(-1j * t * 0.5 * sparse_ham_ob) @ expm(-1j * t * 1 * sparse_ham_tb) @ expm(-1j * t * 0.5 * sparse_ham_ob)
    diff_u = strang_u - exact_u
    # assert np.allclose(strang_u.conj().T, expm(1j * t * 0.5 * sparse_ham_ob) @ expm(1j * t * 1 * sparse_ham_tb) @ expm(1j * t * 0.5 * sparse_ham_ob))
    # assert np.allclose(diff_u.conj().T, strang_u.conj().T - exact_u.conj().T)
    # assert np.allclose(diff_u.conj().T @ diff_u, (strang_u.conj().T - exact_u.conj().T) @ (strang_u - exact_u))

    _, sigma_true, _ = np.linalg.svd(diff_u)
    true_spectral_norm = np.max(sigma_true)
    low_level_spectrum = sigma_true[:10]
    print(f"{low_level_spectrum=}")

    # w, v = np.linalg.eigh(diff_u.conj().T @ diff_u)
    # w = w[::-1]
    # v = v[:, ::-1]
    # eigfunc_overlaps = v.conj().T @ x_cirq
    # print(eigfunc_overlaps[:10])
    # max_eigfunc = fqe.from_cirq(v[:, 0].flatten(), thresh=1.E-12)
    # max_eigfunc.print_wfn()
    # exit()

    # # print(spectral_norm_svd(diff_u))
    # # exit()
    cirq_spectral_norm = spectral_norm_power_method(diff_u, x_cirq, verbose=True, stop_eps=1.0E-10)
    print(f"{true_spectral_norm=}")
    print(f"{cirq_spectral_norm=}")

    # this is not necessarily true
    # assert np.isclose(true_spectral_norm, cirq_spectral_norm)

    fqe_spectral_norm = spectral_norm_fqe_power_iteration(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-10)
    print(f"{true_spectral_norm=}")
    print(f"{cirq_spectral_norm=}")
    print(f"{ fqe_spectral_norm=}")



 

if __name__ == "__main__":
    test_cirq_spectral_norm()