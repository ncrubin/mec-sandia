"""
Grid Hamiltonian spectral norm computation executable
"""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
import sys
import numpy as np

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb


from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid
from mec_sandia.product_formulas.spectral_norm_product import spectral_norm_fqe_power_iteration
from mec_sandia.product_formulas.suzuki import delta_action_6 as delta_action


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
    # fqe_ham_tb_rh = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))
    fqe_ham_tb = DiagonalCoulomb(0.5 * dc_mat)


    # initialize new wavefunction
    if norb == 64:
        fqe.settings.use_accelerated_code = False
    else:
        fqe.settings.use_accelerated_code = True
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    if norb == 64:
        fqe.settings.use_accelerated_code = True
    
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    print(x_wfn.norm() - 1.)

    # calculate spectral norm
    spectral_norm = spectral_norm_fqe_power_iteration(work=x_wfn,
                                                      t=t,
                                                      full_ham=fqe_ham,
                                                      h0=fqe_ham_ob,
                                                      h1=fqe_ham_tb,
                                                      delta_action=delta_action,
                                                      verbose=True,
                                                      stop_eps=0.5E-5)
    np.save("spectral_norm_suzuki_6.npy", np.array(spectral_norm))

if __name__ == "__main__":
    t = float(sys.argv[1])
    ppd = int(sys.argv[2])
    eta = int(sys.argv[3])
    omega = float(sys.argv[4])
    run_spectral_norm_comp(t, ppd, eta, omega)