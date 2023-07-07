"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import sys
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

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
from mec_sandia.product_formulas.pyscf_utility import get_spectrum, pyscf_to_fqe_wf, get_fqe_wfns
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial

from openfermion.resource_estimates.molecule.pyscf_utils import cas_to_pyscf, pyscf_to_cas

from mec_sandia.ueg import UEG, UEGTMP

def small_system(nel: int, nmo: int, rs: float, ecut_max: float=2):
    # pick ecut= 2 which gives 33 planewaves and just truncate
    nalpha = nel // 2
    nbeta = nel // 2
    ueg = UEGTMP(nelec=(nalpha, nbeta), rs=rs, ecut=ecut_max) # kfac ~ rs * nelec**1/3
    eris_8 = ueg.eri_8() # chemist notation (1'1|2'2)
    assert nmo < eris_8.shape[-1]
    h1e = np.diag(ueg.sp_eigv)[:nmo, :nmo]
    eris_8 = eris_8[:nmo, :nmo, :nmo, :nmo]
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

    from mec_sandia.product_formulas.bespoke_berry import spectral_norm_fqe_power_iteration as berry_spectral_norm_fqe
    berry_spectral_norms = []
    t = 0.25
    # initialize new wavefunction
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    # compute spectral norm
    fqe_spectral_norm = berry_spectral_norm_fqe(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, verbose=True, stop_eps=1.0E-8)
    print(f"{ fqe_spectral_norm=}")
    berry_spectral_norms.append(fqe_spectral_norm)
    np.save("berry_spectral_norms", berry_spectral_norms)


if __name__ == "__main__":
    nel = int(sys.argv[1])
    nmo = int(sys.argv[2])
    rs = float(sys.argv[3])
    small_system(nel, nmo, rs)
