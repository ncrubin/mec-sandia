"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import copy
import time

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

from mec_sandia.product_formulas.time_evolution_utility import apply_unitary_wrapper, quad_and_diag_coulomb_apply_unitary_wrapper

MAX_EXPANSION_LIMIT = 200
NORM_ERROR_RESOLUTION = 1.0E-12


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
    assert h0.quadratic() == True
    for ii in range(len(indices)):
        if indices[ii] == 0:
            work = work.time_evolve(t * coeffs[ii], h0)
        elif indices[ii] == 1:
            if isinstance(h1, DiagonalCoulomb):
                work = work.time_evolve(t * coeffs[ii], h1)
            elif isinstance(h1, RestrictedHamiltonian):
                work = apply_unitary_wrapper(base=work,
                                             time=t * coeffs[ii],
                                             algo='taylor',
                                             ops=h1,
                                             accuracy=1.0E-20,
                                             expansion=MAX_EXPANSION_LIMIT,
                                             verbose=False)
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
            if isinstance(h1, DiagonalCoulomb):
                work = work.time_evolve(t * coeffs[ii], h1)
            elif isinstance(h1, RestrictedHamiltonian):
                work = apply_unitary_wrapper(base=work,
                                             time=t * coeffs[ii],
                                             algo='taylor',
                                             ops=h1,
                                             accuracy=1.0E-20,
                                             expansion=MAX_EXPANSION_LIMIT,
                                             verbose=False)
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
    work = apply_unitary_wrapper(base=work,
                                 time=-t,
                                 algo='taylor',
                                 ops=full_ham,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False)

    return work

def delta_action_4(work: fqe.Wavefunction,
                   t: float,
                   full_ham: RestrictedHamiltonian,
                   h0: RestrictedHamiltonian,
                   h1: RestrictedHamiltonian,
                   **apply_unitary_kwargs):
    return delta_action(work, t, full_ham, h0, h1, suzuki_order=4, **apply_unitary_kwargs)


def delta_action_6(work: fqe.Wavefunction,
                   t: float,
                   full_ham: RestrictedHamiltonian,
                   h0: RestrictedHamiltonian,
                   h1: RestrictedHamiltonian,
                   **apply_unitary_kwargs):
    return delta_action(work, t, full_ham, h0, h1, suzuki_order=6, **apply_unitary_kwargs)



def delta_action(work: fqe.Wavefunction,
                 t: float,
                 full_ham: RestrictedHamiltonian,
                 h0: RestrictedHamiltonian,
                 h1: RestrictedHamiltonian,
                 suzuki_order=4,
                **apply_unitary_kwargs):

    if work.norm() - 1. > NORM_ERROR_RESOLUTION:
        print(f"{work.norm()=}", f"{(work.norm() - 1.)=}")
        raise RuntimeError("Input wavefunction wrong norm")

    start_time = time.time()
    if suzuki_order == 4:
        product_wf = suzuki_trotter_fourth_order_u(work, t, h0, h1)
    elif suzuki_order == 6:
        product_wf = suzuki_trotter_sixth_order_u(work, t, h0, h1)
    else:
        raise ValueError("Suzuki Order {} not coded".format(suzuki_order))
    end_time = time.time()
    print("Suzuki u time ", end_time - start_time)

    start_time = time.time()
    if h0.quadratic() and isinstance(h1, DiagonalCoulomb):
        exact_wf = quad_and_diag_coulomb_apply_unitary_wrapper(base=work,
                                         time=t,
                                         algo='taylor',
                                         quad_ham=h0,
                                         diag_coulomb=h1,
                                         accuracy = 1.0E-20,
                                         expansion=MAX_EXPANSION_LIMIT,
                                         **apply_unitary_kwargs
                                         )
    else:
        exact_wf = apply_unitary_wrapper(base=work,
                                         time=t,
                                         algo='taylor',
                                         ops=full_ham,
                                         accuracy = 1.0E-20,
                                         expansion=MAX_EXPANSION_LIMIT,
                                         **apply_unitary_kwargs)
    end_time = time.time()
    print("exact u time ", end_time - start_time)

    return product_wf - exact_wf


def deltadagdelta_action(work: fqe.Wavefunction,
                         t: float,
                         full_ham: RestrictedHamiltonian,
                         h0: RestrictedHamiltonian,
                         h1: RestrictedHamiltonian,
                         suzuki_order=4):
    og_work = copy.deepcopy(work)
    w1 = exact_then_suzuki_u_inverse(work, t, full_ham, h0, h1, suzuki_order=suzuki_order) + \
        suzuki_u_then_exact_inverse(work, t, full_ham, h0, h1, suzuki_order=suzuki_order)
    og_work.scale(2)
    work = og_work - w1
    return work
