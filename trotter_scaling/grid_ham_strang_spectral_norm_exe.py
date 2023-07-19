"""
Grid Hamiltonian spectral norm computation executable
"""
import sys
import numpy as np

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid 
from mec_sandia.product_formulas.spectral_norm_product import spectral_norm_fqe_power_iteration
from mec_sandia.product_formulas.strang import delta_action


def run_spectral_norm_comp(t: float, points_per_dim: int, eta: int, omega: float):
    L = omega**(1/3)
    rsg = RealSpaceGrid(L, points_per_dim)
    h1 = rsg.get_rspace_h1()
    eris = 2 * rsg.get_eris()
    of_eris = eris.transpose((0, 2, 3, 1))
    if np.isclose(eta % 2, 1):
        sz = 1
    else:
        sz = 0
    nelec = eta
    norb = rsg.norb

    fqe_ham = integrals_to_fqe_restricted(h1, of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((h1,))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(h1), np.einsum('ijlk', -0.5 * of_eris)))

     # initialize new wavefunction
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()

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
    t = float(sys.argv[1])
    ppd = int(sys.argv[2])
    eta = int(sys.argv[3])
    omega = float(sys.argv[4])
    run_spectral_norm_comp(t, ppd, eta, omega)
