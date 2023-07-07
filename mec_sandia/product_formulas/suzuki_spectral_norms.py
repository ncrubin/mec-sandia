"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import copy
import numpy as np

from pyscf import gto, scf, ao2mo
from pyscf.fci.cistring import make_strings

import openfermion as of
from openfermion import MolecularData, InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

from mec_sandia.product_formulas.pyscf_utility import get_spectrum, pyscf_to_fqe_wf


def suzuki_trotter_fourth_order_u(work: fqe.Wavefunction, t: float, h0: RestrictedHamiltonian, h1: RestrictedHamiltonian ):
    """Suzuki-fourth order split-operator evolution
    
     indices: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
     coeffs:  [0.20724538589718786, 0.4144907717943757, 0.4144907717943757, 0.4144907717943757, -0.12173615769156357, -0.6579630871775028, -0.12173615769156357, 0.4144907717943757, 0.4144907717943757, 0.4144907717943757, 0.20724538589718786] 
    """
    indices = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    coeffs = [0.20724538589718786, 
              0.4144907717943757, 
              0.4144907717943757, 
              0.4144907717943757, 
              -0.12173615769156357, 
              -0.6579630871775028, 
              -0.12173615769156357, 
              0.4144907717943757, 
              0.4144907717943757, 
              0.4144907717943757, 
              0.20724538589718786]
    for ii in range(len(indices)):
        if indices[ii] == 0:
            work = work.time_evolve(t * coeffs[ii], h0)
        elif indices[ii] == 1:
            work = work.time_evolve(t * coeffs[ii], h1)
        else:
            raise ValueError("The impossible has happened")
    return work


def suzuki_trotter_sixth_order_u(work: fqe.Wavefunction, t: float, h0: RestrictedHamiltonian, h1: RestrictedHamiltonian ):
    """Suzuki-fourth order split-operator evolution
    indices: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    coeffs:  [0.07731617143363592, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, -0.04541560043427138, -0.24546354373581464, -0.04541560043427138, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, -0.04541560043427138, -0.24546354373581464, -0.04541560043427138, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, -0.024703128403719937, -0.20403859967471172, -0.20403859967471172, -0.20403859967471172, 0.05992624404552197, 0.32389108776575565, 0.05992624404552197, -0.20403859967471172, -0.20403859967471172, -0.20403859967471172, -0.024703128403719937, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, -0.04541560043427138, -0.24546354373581464, -0.04541560043427138, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, -0.04541560043427138, -0.24546354373581464, -0.04541560043427138, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.07731617143363592] 
    """
    indices = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    coeffs = [0.07731617143363592, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 
             -0.04541560043427138, -0.24546354373581464, -0.04541560043427138, 0.15463234286727184, 
              0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 
              0.15463234286727184, 0.15463234286727184, -0.04541560043427138, -0.24546354373581464, 
             -0.04541560043427138, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 
             -0.024703128403719937, -0.20403859967471172, -0.20403859967471172, -0.20403859967471172,
              0.05992624404552197, 0.32389108776575565, 0.05992624404552197, -0.20403859967471172, 
             -0.20403859967471172, -0.20403859967471172, -0.024703128403719937, 0.15463234286727184, 
              0.15463234286727184, 0.15463234286727184, -0.04541560043427138, -0.24546354373581464,
             -0.04541560043427138, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 
              0.15463234286727184, 0.15463234286727184, 0.15463234286727184, 0.15463234286727184,
             -0.04541560043427138, -0.24546354373581464, -0.04541560043427138, 0.15463234286727184, 
             0.15463234286727184, 0.15463234286727184, 0.07731617143363592] 

    for ii in range(len(indices)):
        if indices[ii] == 0:
            work = work.time_evolve(t * coeffs[ii], h0)
        elif indices[ii] == 1:
            work = work.time_evolve(t * coeffs[ii], h1)
        else:
            raise ValueError("The impossible has happened")
    return work


def exact_then_suzuki_u_inverse(work: fqe.Wavefunction,
                                t: float,
                                full_ham: RestrictedHamiltonian,
                                h0: RestrictedHamiltonian,
                                h1: RestrictedHamiltonian,
                                suzuki_order=4):
    """U_{product-formula}^{\dagger}U_{exact}
    """
    work = work.time_evolve(t, full_ham)
    if suzuki_order == 4:
        work = suzuki_trotter_fourth_order_u(work, -t, h0, h1)
    elif suzuki_order == 6:
        work = suzuki_trotter_sixth_order_u(work, -t, h0, h1)
    else:
        raise ValueError("Suzuki Order {} not coded".format(suzuki_order))
    return work

def suzuki_u_then_exact_inverse(work: fqe.Wavefunction,
                                t: float,
                                full_ham: RestrictedHamiltonian,
                                h0: RestrictedHamiltonian,
                                h1: RestrictedHamiltonian, 
                                suzuki_order=4):
    """U_{exact}^{\dagger}U_{strang}
    """
    if suzuki_order == 4:
        work = suzuki_trotter_fourth_order_u(work, t, h0, h1)
    elif suzuki_order == 6:
        work = suzuki_trotter_sixth_order_u(work, t, h0, h1)
    else:
        raise ValueError("Suzuki Order {} not coded".format(suzuki_order))
    work = work.time_evolve(-t, full_ham)
    return work

def deltadagdelta_action(work: fqe.Wavefunction,
                         t: float,
                         full_ham: RestrictedHamiltonian,
                         h0: RestrictedHamiltonian,
                         h1: RestrictedHamiltonian,
                         suzuki_order='4'):
    og_work = copy.deepcopy(work)
    w1 = exact_then_suzuki_u_inverse(work, t, full_ham, h0, h1, suzuki_order=suzuki_order) + \
        suzuki_u_then_exact_inverse(work, t, full_ham, h0, h1, suzuki_order=suzuki_order)
    og_work.scale(2)
    work = og_work - w1
    return work


def spectral_norm_fqe_power_iteration(work: fqe.Wavefunction,
                        t: float,
                        full_ham: RestrictedHamiltonian,
                        h0: RestrictedHamiltonian,
                        h1: RestrictedHamiltonian,
                        verbose=True,
                        stop_eps=1.0E-8,
                        suzuki_order=4):
    """Return spectral norm of the difference between product formula unitary and not"""
    prev_sqrt_lam_max = np.inf
    delta_sqrt_lam_max = np.inf
    iter_val = 0
    work.normalize()
    while delta_sqrt_lam_max > stop_eps:
        work = deltadagdelta_action(work, t, full_ham, h0, h1, suzuki_order=suzuki_order)
        rnorm = work.norm()
        work.scale(1/rnorm) 
        sqrt_lam_max = np.sqrt(
            np.abs(
            fqe.vdot(work, deltadagdelta_action(work, t, full_ham, h0, h1, suzuki_order=suzuki_order))
            )
            )
        delta_sqrt_lam_max = np.abs(prev_sqrt_lam_max - sqrt_lam_max)
        if verbose:
            print(iter_val, f"{sqrt_lam_max=}", f"{delta_sqrt_lam_max=}")
        prev_sqrt_lam_max = sqrt_lam_max
        iter_val += 1

    return sqrt_lam_max

