"""
Grid Hamiltonian spectral norm computation executable
"""
import os
os.environ['MKL_NUM_THREADS'] = '4'
import sys
import numpy as np
import itertools

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

def run_spectral_norm_comp(t: float, points_per_dim: int, eta: int, omega: float):
    norb = 64
    sz = 0
    nelec = 2
    h1 = np.random.randn(64**2).reshape((norb, norb))
    h1 = h1.T + h1

    h2 = np.random.randn(64**2).reshape((norb, norb))
    h2 = h2.T + h2
    eris_chem = np.zeros((norb, norb, norb, norb))
    for ll, mm in itertools.product(range(norb), repeat=2):
        eris_chem[ll, ll, mm, mm] = h2[ll, mm]
    of_eris = eris_chem.transpose((0, 2, 3, 1))

    fqe_ham = integrals_to_fqe_restricted(h1, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1,))
    fqe_ham_tb = DiagonalCoulomb(h2)

    # initialize new wavefunction
    fqe.settings.use_accelerated_code = False
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    print(x_wfn.sector((nelec, sz)).coeff.shape)
    print(x_wfn.sector((nelec, sz)).coeff.nbytes / 1000**3)
    fqe.settings.use_accelerated_code = True

    # fqe.settings.use_accelerated_code = False
    diag, vij = fqe_ham_tb.iht(t)
    work = x_wfn._evolve_diagonal_coulomb_inplace(diag, vij)
    print("PASSED")

if __name__ == "__main__":
    # 0.65 3 4 5.0
    t = 0.65 # float(sys.argv[1])
    ppd = 4 # int(sys.argv[2])
    eta = 2 # int(sys.argv[3])
    omega = 5. # float(sys.argv[4])
    run_spectral_norm_comp(t, ppd, eta, omega)