import os
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'

import copy
import numpy as np

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

from mec_sandia.product_formulas.time_evolution_utility import apply_unitary_wrapper

MAX_EXPANSION_LIMIT = 200

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
    # work = apply_unitary_wrapper(base=work,
    #                              time=t * 0.5,
    #                              algo='taylor',
    #                              ops=h0,
    #                              accuracy=1.0E-20,
    #                              expansion=MAX_EXPANSION_LIMIT,
    #                              verbose=False)

    work = apply_unitary_wrapper(base=work,
                                 time=t,
                                 algo='taylor',
                                 ops=h1,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False)
    work = work.time_evolve(t * 0.5, h0)
    # work = apply_unitary_wrapper(base=work,
    #                              time=t * 0.5,
    #                              algo='taylor',
    #                              ops=h0,
    #                              accuracy=1.0E-20,
    #                              expansion=MAX_EXPANSION_LIMIT,
    #                              verbose=False)
    return work

def berry_bespoke(work: fqe.Wavefunction,
                  t: float,
                  h0: RestrictedHamiltonian,
                  h1: RestrictedHamiltonian):
    """S2(w_10 t) S2(w_9 t) .... S2(w_2 t) S2(w_1 t) S2(w_0 t) S2(w_1 t) S2(w_2t) .... S2(w_9 t) S2(w_10 t)
    S2(t) =  exp(iAt/2) exp(iBt) exp(iAt/2)
    """
    wvals = np.array([0.59358060400850625863463274318848,
             -0.46916012347004197296246963141914, 0.27435664258984679072234958738716,
              0.1719387948465677305996164814603, 0.23439874482541384415374265697566,
             -0.48616424480326193899633138022874, 0.49617367388114660354885678908755,
             -0.32660218948439130114486683568042, 0.23271679349369857679469542295681,
              0.098249557414708533273496706507094])
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
                       h1: RestrictedHamiltonian):
    product_wf = berry_bespoke(work, t, h0, h1)
    exact_wf = apply_unitary_wrapper(base=work,
                                     time=t,
                                     algo='taylor',
                                     ops=full_ham,
                                     accuracy = 1.0E-20,
                                     expansion=MAX_EXPANSION_LIMIT,
                                     verbose=False)
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
