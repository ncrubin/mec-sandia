"""Get the Jellium Hamiltonian as an FQE-Hamiltonian"""
import copy
import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from mec_sandia.product_formulas.time_evolution_utility import apply_unitary_wrapper
MAX_EXPANSION_LIMIT = 200

def strang_u(work: fqe.Wavefunction, t: float, fqe_ham_ob: RestrictedHamiltonian, fqe_ham_tb: RestrictedHamiltonian ):
    """Strang split-operator evolution"""
    work = work.time_evolve(t * 0.5, fqe_ham_ob)
    work = apply_unitary_wrapper(base=work,
                                 time=t,
                                 algo='taylor',
                                 ops=fqe_ham_tb,
                                 accuracy=1.0E-20,
                                 expansion=MAX_EXPANSION_LIMIT,
                                 verbose=False)
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

