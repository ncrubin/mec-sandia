from typing import Union
import copy
import numpy as np
import time

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

from mec_sandia.product_formulas.time_evolution_utility import apply_unitary_wrapper, quad_and_diag_coulomb_apply_unitary_wrapper

MAX_EXPANSION_LIMIT = 200
NORM_ERROR_RESOLUTION = 1.0E-10


def u_s2_trotter_cirq(t: float,
                      h0,
                      h1):
    from scipy.linalg import expm
    u = np.eye(h0.shape[0])
    u = u @ expm(-1j * t * 0.5 * h0)
    u = u @ expm(-1j * t * h1)
    u = u @ expm(-1j * t * 0.5 * h0)
    return u

def u_berry_bespoke_cirq(
                  t: float,
                  h0,
                  h1):
    """S2(w_10 t) S2(w_9 t) .... S2(w_2 t) S2(w_1 t) S2(w_0 t) S2(w_1 t) S2(w_2t) .... S2(w_9 t) S2(w_10 t)
    S2(t) =  exp(iAt/2) exp(iBt) exp(iAt/2)
    """
    wvals = [0.59358060400850625863463274318848,
             -0.46916012347004197296246963141914, 0.27435664258984679072234958738716,
              0.1719387948465677305996164814603, 0.23439874482541384415374265697566,
             -0.48616424480326193899633138022874, 0.49617367388114660354885678908755,
             -0.32660218948439130114486683568042, 0.23271679349369857679469542295681,
              0.098249557414708533273496706507094] 
    w_0 = 1 - 2 * sum(wvals) 
    u = np.eye(h0.shape[0])
    for ii in range(9, -1, -1):
        u = u @ u_s2_trotter_cirq(wvals[ii] * t, h0, h1)
    u = u @ u_s2_trotter_cirq(w_0 * t, h0, h1)
    for ii in range(10):
        u = u @ u_s2_trotter_cirq(wvals[ii] * t, h0, h1)
    return u


def evolve_s2_trotter(work: fqe.Wavefunction,
                      t: float,
                      h0: RestrictedHamiltonian,
                      h1: RestrictedHamiltonian):
    assert h0.quadratic() == True
    work = work.time_evolve(t * 0.5, h0) # this should be exact
    if isinstance(h1, DiagonalCoulomb):
        work = work.time_evolve(t, h1)
    elif isinstance(h1, RestrictedHamiltonian):
        work = apply_unitary_wrapper(base=work,
                                     time=t,
                                     algo='taylor',
                                     ops=h1,
                                     accuracy=1.0E-20,
                                     expansion=MAX_EXPANSION_LIMIT,
                                     verbose=False)
    else:
        raise TypeError("The two-body Hamiltonian does not have an allowed type {}".format(type(h1)))
    work = work.time_evolve(t * 0.5, h0)
    return work

def berry_bespoke(work: fqe.Wavefunction,
                  t: float,
                  h0: RestrictedHamiltonian,
                  h1: RestrictedHamiltonian):
    """S2(w_10 t) S2(w_9 t) .... S2(w_2 t) S2(w_1 t) S2(w_0 t) S2(w_1 t) S2(w_2t) .... S2(w_9 t) S2(w_10 t)
    S2(t) =  exp(iAt/2) exp(iBt) exp(iAt/2)
    """
    wvals = np.array([0.59358060400850625863463274318848, -0.46916012347004197296246963141914, 
                      0.27435664258984679072234958738716, 0.1719387948465677305996164814603, 
                      0.23439874482541384415374265697566, -0.48616424480326193899633138022874, 
                      0.49617367388114660354885678908755, -0.32660218948439130114486683568042, 
                      0.23271679349369857679469542295681, 0.098249557414708533273496706507094])
    w_0 = 1. - 2. * np.sum(wvals) 
    for ii in range(9, -1, -1):
        work = evolve_s2_trotter(work, wvals[ii] * t, h0, h1)
    work = evolve_s2_trotter(work, w_0 * t, h0, h1)
    for ii in range(10):
        work = evolve_s2_trotter(work, wvals[ii] * t, h0, h1)
    return work

def old_eigth_order(work: fqe.Wavefunction,
                    t: float,
                    h0: RestrictedHamiltonian,
                    h1: Union[RestrictedHamiltonian, DiagonalCoulomb]):
    wvals = np.array([[0.11699135019217642180722881433533, 0.12581718736176041804392391641587, 
                       0.12603912321825988140305670268365, 0.11892905625000350062692972283951, 
                       0.11317848435755633314700952515599, -0.24445266791528841269462171413216, 
                       -0.23341414023165082198780281128319, 0.35337821052654342419534541324080, 
                       0.10837408645835726397433410591546, 0.10647728984550031823931967854896]])
    w_0 = 1. - 2. * np.sum(wvals) 
    for ii in range(9, -1, -1):
        work = evolve_s2_trotter(work, wvals[ii] * t, h0, h1)
    work = evolve_s2_trotter(work, w_0 * t, h0, h1)
    for ii in range(10):
        work = evolve_s2_trotter(work, wvals[ii] * t, h0, h1)
    return work



def exact_then_berry_u_inverse(work: fqe.Wavefunction,
                               t: float,
                               full_ham: RestrictedHamiltonian,
                               h0: RestrictedHamiltonian,
                               h1: RestrictedHamiltonian):
    """U_{berry}^ U_exact"""
    # note: we do not use time_evolve because
    # we need to customize the expansion rank
    work = apply_unitary_wrapper(base=work,
                                 time=t,
                                 algo='taylor',
                                 ops=full_ham,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False
                                 )
    work = berry_bespoke(work, -t, h0, h1)
    return work

def berry_u_then_exact_inverse(work: fqe.Wavefunction,
                               t: float,
                               full_ham: RestrictedHamiltonian,
                               h0: RestrictedHamiltonian,
                               h1: RestrictedHamiltonian):
    """U_{exact}^U_{berry}"""
    work = berry_bespoke(work, t, h0, h1)
    work = apply_unitary_wrapper(base=work,
                                 time=-t,
                                 algo='taylor',
                                 ops=full_ham,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False
                                 )
    return work

def berry_delta_action(work: fqe.Wavefunction,
                       t: float,
                       full_ham: RestrictedHamiltonian,
                       h0: RestrictedHamiltonian,
                       h1: RestrictedHamiltonian,
                       **apply_unitary_kwargs
                       ):
    
    if work.norm() - 1. > NORM_ERROR_RESOLUTION:
        print(f"{work.norm()=}", f"{(work.norm() - 1.)=}")
        raise RuntimeError("Input wavefunction wrong norm")

    start_time = time.time()
    product_wf = berry_bespoke(work, t, h0, h1)
    end_time = time.time()
    print("Berry-8 u time ", end_time - start_time)

    if product_wf.norm() - 1. >  NORM_ERROR_RESOLUTION:
        print(f"{product_wf.norm()=}", f"{(product_wf.norm() - 1.)=}")
        raise RuntimeError("Evolution did not converge")

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

def old_eight_order_delta_action(work: fqe.Wavefunction,
                       t: float,
                       full_ham: RestrictedHamiltonian,
                       h0: RestrictedHamiltonian,
                       h1: RestrictedHamiltonian,
                       **apply_unitary_kwargs
                       ):
    
    if work.norm() - 1. > NORM_ERROR_RESOLUTION:
        print(f"{work.norm()=}", f"{(work.norm() - 1.)=}")
        raise RuntimeError("Input wavefunction wrong norm")

    start_time = time.time()
    product_wf = old_eigth_order(work, t, h0, h1)
    end_time = time.time()
    print("old-8 u time ", end_time - start_time)

    if product_wf.norm() - 1. >  NORM_ERROR_RESOLUTION:
        print(f"{product_wf.norm()=}", f"{(product_wf.norm() - 1.)=}")
        raise RuntimeError("Evolution did not converge")

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


def berry_deltadagdelta_action(work: fqe.Wavefunction,
                         t: float,
                         full_ham: RestrictedHamiltonian,
                         h0: RestrictedHamiltonian,
                         h1: RestrictedHamiltonian):
    og_work = copy.deepcopy(work)
    w1 = exact_then_berry_u_inverse(work, t, full_ham, h0, h1)
    w2 = berry_u_then_exact_inverse(work, t, full_ham, h0, h1)
    og_work.scale(2.)
    return og_work - w1 - w2
