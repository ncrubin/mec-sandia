"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import os
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'

import itertools
import copy
import openfermion as of
import numpy as np
import fqe
from pyscf import gto, scf, ao2mo
from pyscf.fci.cistring import make_strings
from openfermion import MolecularData

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from pyscf_utility import get_spectrum, pyscf_to_fqe_wf, get_fqe_wfns
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial

from openfermion.resource_estimates.molecule.pyscf_utils import cas_to_pyscf, pyscf_to_cas

from mec_sandia.ueg import UEG, UEGTMP

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

    # calculate SCF energy
    _, mf = cas_to_pyscf(h1=h1e, eri=eris_8, ecore=0, num_alpha=nalpha, num_beta=nbeta)

    of_eris = eris_8.transpose((0, 2, 3, 1))
    fqe_ham = integrals_to_fqe_restricted(h1e, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1e, ))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(h1e), np.einsum('ijlk', -0.5 * of_eris)))

    from bespoke_berry import spectral_norm_fqe_power_iteration as berry_spectral_norm_fqe
    from strang_spectral_norm import spectral_norm_fqe_power_iteration as strang_spectral_norm_fqe
    from suzuki_spectral_norms import spectral_norm_fqe_power_iteration as suzuki_spectral_norm_fqe
    tvals = np.logspace(-0.5, -2, 10)
    berry_spectral_norms = []
    for t in tvals:
        # initialize new wavefunction
        x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
        x_wfn.set_wfn(strategy='ones')
        x_wfn.normalize()

        # compute spectral norm
        fqe_spectral_norm = berry_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8)
        print(f"{ fqe_spectral_norm=}")
        berry_spectral_norms.append(fqe_spectral_norm)
    np.save("berry_spectral_norms", berry_spectral_norms)
    strang_spectral_norms = []
    for t in tvals:
        # initialize new wavefunction
        x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
        x_wfn.set_wfn(strategy='ones')
        x_wfn.normalize()

        # compute spectral norm
        fqe_spectral_norm = strang_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8)
        print(f"{ fqe_spectral_norm=}")
        strang_spectral_norms.append(fqe_spectral_norm)
    np.save("strang_spectral_norms", strang_spectral_norms)
    suzuki_4_spectral_norms = []
    for t in tvals:
        # initialize new wavefunction
        x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
        x_wfn.set_wfn(strategy='ones')
        x_wfn.normalize()

        # compute spectral norm
        fqe_spectral_norm = suzuki_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8, suzuki_order=4)
        print(f"{ fqe_spectral_norm=}")
        suzuki_4_spectral_norms.append(fqe_spectral_norm)
    np.save("suzuki_4_spectral_norms", suzuki_4_spectral_norms)
    suzuki_6_spectral_norms = []
    for t in tvals:
        # initialize new wavefunction
        x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
        x_wfn.set_wfn(strategy='ones')
        x_wfn.normalize()

        # compute spectral norm
        fqe_spectral_norm = suzuki_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8, suzuki_order=6)
        print(f"{ fqe_spectral_norm=}")
        suzuki_6_spectral_norms.append(fqe_spectral_norm)
    np.save("suzuki_6_spectral_norms", suzuki_6_spectral_norms)


if __name__ == "__main__":
    # test_eris()
    small_system()
