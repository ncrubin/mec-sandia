"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import itertools
import copy
import numpy as np
from scipy.linalg import expm

from pyscf import gto, scf, ao2mo
from pyscf.fci.cistring import make_strings

import openfermion as of
from openfermion import MolecularData, InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.resource_estimates.molecule.pyscf_utils import cas_to_pyscf, pyscf_to_cas

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

from mec_sandia.ueg import UEG, UEGTMP
from mec_sandia.product_formulas.pyscf_utility import get_spectrum, pyscf_to_fqe_wf, get_fqe_wfns
from mec_sandia.product_formulas.bespoke_berry import spectral_norm_fqe_power_iteration as berry_spectral_norm_fqe
from mec_sandia.product_formulas.strang_spectral_norm import spectral_norm_fqe_power_iteration as strang_spectral_norm_fqe
from mec_sandia.product_formulas.suzuki_spectral_norms import spectral_norm_fqe_power_iteration as suzuki_spectral_norm_fqe
from mec_sandia.product_formulas.strang_spectral_norm import spectral_norm_power_method as spectral_norm_power_method_cirq



def small_system():
    ueg = UEGTMP(nelec=(2, 2), rs=1.0, ecut=0.7) # kfac ~ rs * nelec**1/3
    eris_8 = ueg.eri_8() # chemist notation (1'1|2'2)
    h1e = np.diag(ueg.sp_eigv)
    nelec = ueg.nelec
    nalpha = nelec // 2
    nbeta = nelec // 2
    norb = eris_8.shape[0]
    sz = nalpha - nbeta
    occ = nalpha

    of_eris = eris_8.transpose((0, 2, 3, 1))
    fqe_ham = integrals_to_fqe_restricted(h1e, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1e, ))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(h1e), np.einsum('ijlk', -0.5 * of_eris)))

    # set up cirq hamiltonians
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(h1e, of_eris)
    molecular_hamiltonian = InteractionOperator(0, one_body_coefficients, 1 / 2 * two_body_coefficients)
    sparse_ham = of.get_sparse_operator(molecular_hamiltonian).todense()
    molecular_hamiltonian_ob = InteractionOperator(0, one_body_coefficients, np.zeros_like(1 / 2 * two_body_coefficients))
    molecular_hamiltonian_tb = InteractionOperator(0, np.zeros_like(one_body_coefficients), 1 / 2 * two_body_coefficients)
    sparse_ham_ob = of.get_sparse_operator(molecular_hamiltonian_ob).todense()
    sparse_ham_tb = of.get_sparse_operator(molecular_hamiltonian_tb).todense()


    tvals = np.logspace(0.0, -0.1, 5)
    berry_spectral_norms = []
    for t in tvals:
        # initialize new wavefunction
        x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
        x_wfn.set_wfn(strategy='ones')
        x_wfn.normalize()
        x_cirq = fqe.to_cirq(x_wfn)

        # # compute spectral norm
        # fqe_spectral_norm = berry_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-9)
        # print(f"{ fqe_spectral_norm=}")
        # berry_spectral_norms.append(fqe_spectral_norm)

        # compute via power iteration
        exact_u = expm(-1j * t * sparse_ham)
        exit()
        from mec_sandia.product_formulas.bespoke_berry import u_berry_bespoke_cirq
        berry_u = u_berry_bespoke_cirq(t, sparse_ham_ob, sparse_ham_tb) 
        diff_u = berry_u - exact_u
        cirq_spectral_norm = spectral_norm_power_method_cirq(diff_u, x_cirq, verbose=True, stop_eps=1.0E-10)
        print(f"{cirq_spectral_norm=}")
        print(f"{ fqe_spectral_norm=}")
        exit()



    exit()

    np.save("berry_spectral_norms", berry_spectral_norms)

    # tvals = np.logspace(0.0, -0.1, 5)
    # strang_spectral_norms = []
    # for t in tvals:
    #     # initialize new wavefunction
    #     x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    #     x_wfn.set_wfn(strategy='ones')
    #     x_wfn.normalize()

    #     # compute spectral norm
    #     fqe_spectral_norm = strang_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8)
    #     print(f"{ fqe_spectral_norm=}")
    #     strang_spectral_norms.append(fqe_spectral_norm)
    # np.save("strang_spectral_norms", strang_spectral_norms)
    # suzuki_4_spectral_norms = []
    # for t in tvals:
    #     # initialize new wavefunction
    #     x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    #     x_wfn.set_wfn(strategy='ones')
    #     x_wfn.normalize()

    #     # compute spectral norm
    #     fqe_spectral_norm = suzuki_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8, suzuki_order=4)
    #     print(f"{ fqe_spectral_norm=}")
    #     suzuki_4_spectral_norms.append(fqe_spectral_norm)
    # np.save("suzuki_4_spectral_norms", suzuki_4_spectral_norms)
    # suzuki_6_spectral_norms = []
    # for t in tvals:
    #     # initialize new wavefunction
    #     x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    #     x_wfn.set_wfn(strategy='ones')
    #     x_wfn.normalize()

    #     # compute spectral norm
    #     fqe_spectral_norm = suzuki_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8, suzuki_order=6)
    #     print(f"{ fqe_spectral_norm=}")
    #     suzuki_6_spectral_norms.append(fqe_spectral_norm)
    # np.save("suzuki_6_spectral_norms", suzuki_6_spectral_norms)


if __name__ == "__main__":
    small_system()
