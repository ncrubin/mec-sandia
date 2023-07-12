import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

import openfermion as of
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial

from mec_sandia.product_formulas.systems.molecules import lih_molecule, heh_molecule
from mec_sandia.product_formulas.spectral_norm_product import spectral_norm_fqe_power_iteration

def test_berry_eight_order():
    # mol_mf = lih_molecule(basis='sto-3g')
    mol_mf = heh_molecule()
    nelec = mol_mf.mol.nelectron
    nalpha = nelec // 2
    nbeta = nelec // 2
    norb = mol_mf.mo_coeff.shape[0]
    print(f"{nelec=}", f"{norb=}")
    sz = 0
    of_eris = np.array(mol_mf._eri.transpose((0, 2, 3, 1)))
    fqe_ham = integrals_to_fqe_restricted(mol_mf.get_hcore(), of_eris)    
    fqe_ham_ob = RestrictedHamiltonian((mol_mf.get_hcore(), ))
    fqe_ham_tb = RestrictedHamiltonian((np.zeros_like(mol_mf.get_hcore()), np.einsum('ijlk', -0.5 * of_eris)))

    # set up cirq hamiltonians
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(mol_mf.get_hcore(), of_eris)
    molecular_hamiltonian = InteractionOperator(0, one_body_coefficients, 1 / 2 * two_body_coefficients)
    sparse_ham = of.get_sparse_operator(molecular_hamiltonian).todense()
    molecular_hamiltonian_ob = InteractionOperator(0, one_body_coefficients, np.zeros_like(1 / 2 * two_body_coefficients))
    molecular_hamiltonian_tb = InteractionOperator(0, np.zeros_like(one_body_coefficients), 1 / 2 * two_body_coefficients)
    sparse_ham_ob = of.get_sparse_operator(molecular_hamiltonian_ob).todense()
    sparse_ham_tb = of.get_sparse_operator(molecular_hamiltonian_tb).todense()

    from scipy.linalg import expm
    t = 5.011872336272722
    exact_u = expm(-1.j * t * sparse_ham)
    np.random.seed(50)
    x_wfn = fqe.Wavefunction([[nelec, sz, norb]])
    x_wfn.set_wfn(strategy='ones')
    x_wfn.normalize()
    x_cirq = fqe.to_cirq(x_wfn)
    # 0.0016929495953134997
    # 0.001692949592699579
    # 0.0016929495203156786
    # 0.0016929496163362457
    from mec_sandia.product_formulas.bespoke_berry import berry_deltadagdelta_action
    fqe_spectral_norm = spectral_norm_fqe_power_iteration(x_wfn, t, fqe_ham, fqe_ham_ob, fqe_ham_tb, 
                                                          berry_deltadagdelta_action,
                                                          verbose=True, stop_eps=1.0E-10)
    
if __name__ == "__main__":
    test_berry_eight_order()