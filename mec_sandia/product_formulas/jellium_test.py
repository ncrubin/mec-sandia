"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import itertools
import copy
import openfermion as of
import numpy as np
from scipy.linalg import expm

from pyscf import gto, scf, ao2mo
from pyscf.fci.cistring import make_strings

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf_utility import get_spectrum, pyscf_to_fqe_wf, get_fqe_wfns

from openfermion import MolecularData, InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.resource_estimates.molecule.pyscf_utils import cas_to_pyscf, pyscf_to_cas

from mec_sandia.ueg import UEG, UEGTMP


def test_eris():
    ueg = UEGTMP(nelec=(7, 7), rs=1.0, ecut=1)
    assert ueg.nbasis == 19
    ueg = UEGTMP(nelec=(7, 7), rs=1.0, ecut=1)
    for ecutval in np.linspace(0, 2, 10):
        ueg = UEGTMP(nelec=(1, 1), rs=1.0, ecut=ecutval) # kfac ~ rs * nelec**1/3
        print(f"{ueg.nelec=}", f"{ecutval=}", f"{ueg.nbasis=}", f"{ueg.kfac=}")
    print()
    for ecutval in np.linspace(0, 2, 10):
        ueg = UEGTMP(nelec=(2, 2), rs=1.0, ecut=ecutval) # kfac ~ rs * nelec**1/3
        print(f"{ueg.nelec=}", f"{ecutval=}", f"{ueg.nbasis=}", f"{ueg.kfac=}")

    ueg = UEGTMP(nelec=(7, 7), rs=1.0, ecut=1)
    eris_4 = ueg.eri_4() # chemist notation (1'1|2'2)
    assert np.allclose(eris_4, eris_4.transpose(1, 0, 3, 2))
    assert np.allclose(eris_4, eris_4.transpose(2, 3, 0, 1))
    assert not np.allclose(eris_4, eris_4.transpose(0, 1, 3, 2))
    eris_8 = ueg.eri_8() # chemist notation (1'1|2'2)
    assert np.allclose(eris_8.imag, 0)
    assert np.allclose(eris_8, eris_8.transpose(1, 0, 2, 3))
    assert np.allclose(eris_8, eris_8.transpose(0, 1, 3, 2))
    assert eris_4.shape == (19,) * 4

    for i, j, k, l in itertools.product(range(ueg.nbasis), repeat=4):
        # momentum conservation
        q1 = ueg.basis[k] - ueg.basis[i]
        q2 = ueg.basis[j] - ueg.basis[l]
        if np.allclose(q1, q2) and np.dot(q1, q1) > 0:
            assert np.isclose(
                eris_4[i, k, j, l],
                1.0 / ueg.vol * ueg.vq(ueg.kfac * (ueg.basis[k] - ueg.basis[i])),
            )
        else:
            assert eris_4[i, k, j, l] == 0
        assert np.allclose(eris_4[i, k, j, l], ueg.hijkl(i, j, k, l)) #hijkl is phys <1'2'|12> and outputs (ik|jl)

    h1e = np.diag(ueg.sp_eigv)
    assert h1e.shape == (ueg.nbasis,) * 2
    # planewaves are MOs for UEG so DM = 2 * I in the oo block
    e1b = 2 * np.sum(h1e[:7, :7])
    e2b = 2 * np.einsum("ppqq->", eris_4[:7, :7, :7, :7])
    e2b -= np.einsum("pqqp->", eris_4[:7, :7, :7, :7])
    etot = e1b + e2b
    assert np.isclose(etot, 13.60355734)  # HF energy from HANDE


def small_system():
    ueg = UEGTMP(nelec=(2, 2), rs=1.0, ecut=0.7) # kfac ~ rs * nelec**1/3
    eris_8 = ueg.eri_8() # chemist notation (1'1|2'2)
    h1e = np.diag(ueg.sp_eigv)
    print(h1e.shape)
    print(eris_8.shape)
    nelec = ueg.nelec
    nalpha = nelec // 2
    nbeta = nelec // 2
    norb = eris_8.shape[0]
    occ = nalpha


    e1b = 2 * np.sum(h1e[:nalpha, :nalpha])
    e2b = 2 * np.einsum("ppqq->", eris_8[:nalpha, :nalpha, :nalpha, :nalpha])
    e2b -= np.einsum("pqqp->", eris_8[:nalpha, :nalpha, :nalpha, :nalpha])
    escf = e1b + e2b
    print(f"{escf=}")


    of_eris = eris_8.transpose((0, 2, 3, 1))
    fqe_ham = integrals_to_fqe_restricted(h1e, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1e, ))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(h1e), np.einsum('ijlk', -0.5 * of_eris)))

    # one_body_coefficients, two_body_coefficients = spinorb_from_spatial(h1e, of_eris)
    # molecular_hamiltonian = InteractionOperator(0, one_body_coefficients, 0.5 * two_body_coefficients)
    # sparse_ham = of.get_sparse_operator(molecular_hamiltonian)

    _, mf = cas_to_pyscf(h1=h1e, eri=eris_8, ecore=0, num_alpha=nalpha, num_beta=nbeta)
    th1, teri, tecore, tnum_alpha, tnum_beta = pyscf_to_cas(mf)
    assert np.allclose(th1, h1e)
    assert np.allclose(teri, eris_8)
    assert np.allclose(tnum_alpha, nalpha)
    assert np.allclose(tnum_beta, nbeta)

    hf_wf = fqe.Wavefunction([[nelec, 0, norb]])
    hf_wf.set_wfn(strategy='hartree-fock')
    hf_wf.print_wfn()
    fqe_hf_energy = hf_wf.expectationValue(fqe_ham).real
    print(f"{fqe_hf_energy=}")
    assert np.isclose(fqe_hf_energy, escf)
    assert np.isclose(mf.energy_tot(), escf)


    roots, wfns = get_spectrum(mf, num_roots=3)
    print(roots)
    gs_e, gs_wfn = roots, pyscf_to_fqe_wf(wfns[0], pyscf_mf=mf)
    print(f"{gs_e[0]=}")
    print(f"{roots[0]=}")

    print(f"{gs_wfn.expectationValue(fqe_ham).real=}")
    print(f"{(gs_wfn.expectationValue(fqe_ham_ob).real + gs_wfn.expectationValue(fqe_ham_tb).real)=}")



if __name__ == "__main__":
    test_eris()
    small_system()
