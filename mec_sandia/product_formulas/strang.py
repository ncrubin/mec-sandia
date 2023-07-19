"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
from typing import Union
import copy
import time

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

from mec_sandia.product_formulas.time_evolution_utility import apply_unitary_wrapper

MAX_EXPANSION_LIMIT = 200
NORM_ERROR_RESOLUTION = 1.0E-13


def strang_u(work: fqe.Wavefunction, t: float, fqe_ham_ob: RestrictedHamiltonian, fqe_ham_tb: Union[RestrictedHamiltonian, DiagonalCoulomb] ):
    """Strang split-operator evolution"""
    work = work.time_evolve(t * 0.5, fqe_ham_ob)
    if isinstance(fqe_ham_tb, DiagonalCoulomb):
        print("Using diagonal coulomb evolution")
        work = work.time_evolve(t, fqe_ham_tb)
    elif isinstance(fqe_ham_tb, RestrictedHamiltonian):
        work = apply_unitary_wrapper(base=work,
                                     time=t,
                                     algo='taylor',
                                     ops=fqe_ham_tb,
                                     accuracy=1.0E-20,
                                     expansion=MAX_EXPANSION_LIMIT,
                                     verbose=False)
    else:
        raise TypeError("The two-body Hamiltonian does not have an allowed type {}".format(type(fqe_ham_tb)))
    work = work.time_evolve(t * 0.5, fqe_ham_ob)
    return work

def exact_then_strang_u_inverse(work: fqe.Wavefunction,
                                t: float,
                                full_ham: RestrictedHamiltonian,
                                h0: RestrictedHamiltonian,
                                h1: RestrictedHamiltonian):
    """U_{strang}^{\dagger}U_{exact}
    """
    work = apply_unitary_wrapper(base=work,
                                 time=t,
                                 algo='taylor',
                                 ops=full_ham,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False
                                 )
    work = work.time_evolve(-t * 0.5, h0)
    work = apply_unitary_wrapper(base=work,
                                 time=-t,
                                 algo='taylor',
                                 ops=h1,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False
                                 )
    work = work.time_evolve(-t * 0.5, h0)
    return work

def strang_u_then_exact_inverse(work: fqe.Wavefunction,
                                t: float,
                                full_ham: RestrictedHamiltonian,
                                h0: RestrictedHamiltonian,
                                h1: RestrictedHamiltonian):
    """U_{exact}^{\dagger}U_{strang}
    """
    work = work.time_evolve(t * 0.5, h0)
    work = apply_unitary_wrapper(base=work,
                                 time=t,
                                 algo='taylor',
                                 ops=h1,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False
                                 )
    work = work.time_evolve(t * 0.5, h0)
    work = apply_unitary_wrapper(base=work,
                                 time=-t,
                                 algo='taylor',
                                 ops=full_ham,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False
                                 )
    if work.norm() - 1. > 1.0E-14:
        raise RuntimeError("exact evolution did not converge")
    return work

def delta_action(work: fqe.Wavefunction,
                 t: float,
                 full_ham: RestrictedHamiltonian,
                 h0: RestrictedHamiltonian,
                 h1: Union[RestrictedHamiltonian, DiagonalCoulomb]):
    if work.norm() - 1. > NORM_ERROR_RESOLUTION:
        print(f"{work.norm()=}", f"{(work.norm() - 1.)=}")
        raise RuntimeError("Input wavefunction wrong norm")

    start_time = time.time()
    product_wf = strang_u(copy.deepcopy(work), t, h0, h1)
    end_time = time.time()

    print("Strang u time ", end_time - start_time)

    if product_wf.norm() - 1. >  NORM_ERROR_RESOLUTION:
        print(f"{product_wf.norm()=}", f"{(product_wf.norm() - 1.)=}")
        raise RuntimeError("Evolution did not converge")
    
    start_time = time.time()
    exact_wf = apply_unitary_wrapper(base=work,
                                     time=t,
                                     algo='taylor',
                                     ops=full_ham,
                                     accuracy = 1.0E-20,
                                     expansion=MAX_EXPANSION_LIMIT,
                                     verbose=False,
                                     debug=False)
    end_time = time.time()
    print("exact u time ", end_time - start_time)
    return product_wf - exact_wf


def deltadagdelta_action(work: fqe.Wavefunction,
                         t: float,
                         full_ham: RestrictedHamiltonian,
                         h0: RestrictedHamiltonian,
                         h1: RestrictedHamiltonian):
    og_work = copy.deepcopy(work)
    w1 = exact_then_strang_u_inverse(work, t, full_ham, h0, h1) + strang_u_then_exact_inverse(work, t, full_ham, h0, h1)
    og_work.scale(2)
    work = og_work - w1
    return work

