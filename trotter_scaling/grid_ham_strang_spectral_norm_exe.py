"""
Grid Hamiltonian spectral norm computation executable
"""
import os
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
import sys
import numpy as np

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid 
from mec_sandia.product_formulas.spectral_norm_product import spectral_norm_fqe_power_iteration
from mec_sandia.product_formulas.strang import delta_action


def run_spectral_norm_comp(t: float, points_per_dim: int, eta: int, omega: float):
    L = omega**(1/3)
    rsg = RealSpaceGrid(L, points_per_dim)
    h1 = rsg.get_rspace_h1()
    eris = 2 * rsg.get_eris()
    dc_mat = 2 * rsg.get_diagonal_coulomb_matrix()
    of_eris = eris.transpose((0, 2, 3, 1))
    if np.isclose(eta % 2, 1):
        sz = 1
    else:
        sz = 0
    nelec = eta
    norb = rsg.norb

    fqe_ham = integrals_to_fqe_restricted(h1, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1,))
    # fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))
    fqe_ham_tb = DiagonalCoulomb(0.5 * dc_mat)

     # initialize new wavefunction
    fqe.settings.use_accelerated_code = True
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    print(x_wfn.sector((nelec, sz)).coeff.shape)
    print(x_wfn.sector((nelec, sz)).coeff.nbytes / 1000**3)

    diag, vij = fqe_ham_tb.iht(t)

    work = x_wfn._evolve_diagonal_coulomb_inplace(diag, vij)
    # work = x_wfn.time_evolve(t, fqe_ham_tb)

    new_wfn = delta_action(x_wfn, t, full_ham=fqe_ham, h0=fqe_ham_ob, h1=fqe_ham_tb)
    exit(0)

    # calculate spectral norm
    spectral_norm = spectral_norm_fqe_power_iteration(work=x_wfn,
                                                      t=t,
                                                      full_ham=fqe_ham,
                                                      h0=fqe_ham_ob,
                                                      h1=fqe_ham_tb,
                                                      delta_action=delta_action,
                                                      verbose=True,
                                                      stop_eps=1.0E-10)
    np.save("spectral_norm.npy", np.array(spectral_norm))

if __name__ == "__main__":
    # 0.65 3 4 5.0
    t = 0.65 # float(sys.argv[1])
    ppd = 3 # int(sys.argv[2])
    eta = 2 # int(sys.argv[3])
    omega = 5. # float(sys.argv[4])
    run_spectral_norm_comp(t, ppd, eta, omega)
