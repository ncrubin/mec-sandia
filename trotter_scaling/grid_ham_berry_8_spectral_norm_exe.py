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
from mec_sandia.product_formulas.bespoke_berry import berry_delta_action as delta_action


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
    fqe_ham_tb_rh = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))
    fqe_ham_tb = DiagonalCoulomb(0.5 * dc_mat)

    fqe.settings.use_accelerated_code = True
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    print(x_wfn.norm() - 1.)
    
    # calculate spectral norm
    apply_generated_unitary_kwargs = {"smallest_time_slice": 10, "verbose": True, "debug": True}
    import time
    start_time = time.time()
    spectral_norm1 = spectral_norm_fqe_power_iteration(work=x_wfn,
                                                      t=t,
                                                      full_ham=fqe_ham,
                                                      h0=fqe_ham_ob,
                                                      h1=fqe_ham_tb_rh,
                                                      delta_action=delta_action,
                                                      delta_action_kwargs=apply_generated_unitary_kwargs,
                                                      stop_eps=1.0E-3,
                                                      verbose=True,
                                                      )
    end_time = time.time()
    specnorm1_time = end_time - start_time

    start_time = time.time() 
    spectral_norm2 = spectral_norm_fqe_power_iteration(work=x_wfn,
                                                      t=t,
                                                      full_ham=fqe_ham,
                                                      h0=fqe_ham_ob,
                                                      h1=fqe_ham_tb,
                                                      delta_action=delta_action,
                                                      delta_action_kwargs=apply_generated_unitary_kwargs,
                                                      verbose=True,
                                                      stop_eps=1.0E-3)
    end_time = time.time()
    specnorm2_time = end_time - start_time
    print(f"{spectral_norm1=}")
    print(f"{spectral_norm2=}")
    print(f"{specnorm1_time=}")
    print(f"{specnorm2_time=}")
    exit()
    np.save("spectral_norm_berry_8.npy", np.array(spectral_norm))

if __name__ == "__main__":
    t = 0.65 # float(sys.argv[1])
    ppd = 2 # int(sys.argv[2])
    eta = 3 # int(sys.argv[3])
    omega = 5. # float(sys.argv[4])
    run_spectral_norm_comp(t, ppd, eta, omega)

    # t = float(sys.argv[1])
    # ppd = int(sys.argv[2])
    # eta = int(sys.argv[3])
    # omega = float(sys.argv[4])
    # run_spectral_norm_comp(t, ppd, eta, omega)