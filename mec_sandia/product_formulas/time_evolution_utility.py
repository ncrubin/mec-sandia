"""
Time Evolution utiltiy
"""
import copy
import numpy as np
from scipy.special import factorial

import fqe
from fqe.hamiltonians import hamiltonian
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb

NORM_ERROR_RESOLUTION = 1.0E-10

def apply_unitary_wrapper(base: fqe.Wavefunction, time: float, algo, ops: hamiltonian.Hamiltonian, 
                          accuracy=1.0E-20, expansion=200, verbose=True,
                          smallest_time_slice=0.5, debug=False) -> fqe.Wavefunction:
    upper_bound_norm = 0
    norm_of_coeffs = []
    for op_coeff_tensor in ops.iht(time):
        upper_bound_norm += np.sum(np.abs(op_coeff_tensor))
        norm_of_coeffs.append(np.linalg.norm(op_coeff_tensor))
    if debug:
        print(f"{upper_bound_norm=}")

    if upper_bound_norm >= smallest_time_slice:
        num_slices = int(np.ceil(upper_bound_norm / smallest_time_slice))
        time_evol = copy.deepcopy(base)
        if debug:
            print("Using {} slices to evolve for {} time".format(num_slices, time), norm_of_coeffs)
            if time_evol.norm() - 1. > NORM_ERROR_RESOLUTION:
                print("pre-slice-start", f"{time_evol.norm()=}", f"{(time_evol.norm() - 1.)=}")
                raise RuntimeError("Evolution did not converge")

        total_time = 0
        for mm in range(num_slices):
            if debug:
                print("Slice ", f"{mm=}")

            time_evol = time_evol.apply_generated_unitary(
                time=time / num_slices, algo=algo, ops=ops, accuracy=accuracy, expansion=expansion,
                                                verbose=verbose
                                                )
            total_time += time / num_slices
            if debug:
                if time_evol.norm() - 1. > NORM_ERROR_RESOLUTION:
                    print(f"{mm=}", f"{time_evol.norm()=}", f"{(time_evol.norm() - 1.)=}")
                    raise RuntimeError("Evolution did not converge")
        if debug:
            assert np.isclose(total_time, time)
    else:
        time_evol = base.apply_generated_unitary(
            time=time, algo=algo, ops=ops, accuracy=accuracy, expansion=expansion,
                                            verbose=verbose
                                            )
    if time_evol.norm() - 1. > NORM_ERROR_RESOLUTION:
        print(f"Post suceess run {time_evol.norm()=}", f"{(time_evol.norm() - 1.)=}")
        raise RuntimeError("Evolution did not converge")
    
    return time_evol

def quad_and_diag_coulomb_apply_unitary_wrapper(base: fqe.Wavefunction, 
                              time: float, 
                              quad_ham: RestrictedHamiltonian, 
                              diag_coulomb: DiagonalCoulomb,
                              algo='taylor', 
                              accuracy=1.0E-20, 
                              expansion=200, 
                              verbose=False,
                              debug=False,
                              smallest_time_slice=0.5) -> fqe.Wavefunction:
    """
    Exact time evolution when Hamiltonian is composed of a quadratic Hamiltonian and
    a diagonal coulomb Hamiltonian as is the case for grid based Hamiltonians.

    This function effectively stubs out 
    """
    upper_bound_norm = 0
    norm_of_coeffs = []
    quad_ham_array = quad_ham.iht(time)[0]
    dc_ham_array = diag_coulomb.iht(time)

    upper_bound_norm += np.sum(np.abs(quad_ham_array))
    upper_bound_norm += np.sum(np.abs(dc_ham_array[0]))
    upper_bound_norm += np.sum(np.abs(dc_ham_array[1]))
    if debug:
        print(f"{upper_bound_norm=}")

    norm_of_coeffs.append(np.linalg.norm(quad_ham_array))
    norm_of_coeffs.append(np.linalg.norm(dc_ham_array[0]) + np.linalg.norm(dc_ham_array[1]))

    if upper_bound_norm >= smallest_time_slice:
        num_slices = int(np.ceil(upper_bound_norm / smallest_time_slice))
        time_evol = copy.deepcopy(base)
        if debug:
            print("Using {} slices to evolve for {} time".format(num_slices, time), norm_of_coeffs)
            if time_evol.norm() - 1. > NORM_ERROR_RESOLUTION:
                print("pre-slice-start", f"{time_evol.norm()=}", f"{(time_evol.norm() - 1.)=}")
                raise RuntimeError("Evolution did not converge")

        total_time = 0
        for mm in range(num_slices):
            if debug:
                print("Slice ", f"{mm=}")

            time_evol = _apply_generated_unitary(base=time_evol,
                time=time / num_slices, algo=algo, quad_ham=quad_ham, diag_coulomb=diag_coulomb, accuracy=accuracy, expansion=expansion,
                                                verbose=verbose
                                                )
            total_time += time / num_slices
            if debug:
                if time_evol.norm() - 1. > NORM_ERROR_RESOLUTION:
                    print(f"{mm=}", f"{time_evol.norm()=}", f"{(time_evol.norm() - 1.)=}")
                    raise RuntimeError("Evolution did not converge")
        if debug:
            assert np.isclose(total_time, time)
    else:
        time_evol = _apply_generated_unitary(base=base,
            time=time, algo=algo, quad_ham=quad_ham, diag_coulomb=diag_coulomb, accuracy=accuracy, expansion=expansion,
                                            verbose=verbose
                                            )
    if time_evol.norm() - 1. > NORM_ERROR_RESOLUTION:
        print(f"Post suceess run {time_evol.norm()=}", f"{(time_evol.norm() - 1.)=}")
        raise RuntimeError("Evolution did not converge")
    
    return time_evol

def _apply_generated_unitary(base: fqe.Wavefunction, 
                             time: float, 
                             quad_ham: RestrictedHamiltonian, 
                             diag_coulomb: DiagonalCoulomb,
                             algo='taylor', 
                             accuracy=1.0E-20, 
                             expansion=200, 
                             verbose=False) -> fqe.Wavefunction:
    assert quad_ham.quadratic() == True
    assert isinstance(diag_coulomb, DiagonalCoulomb)
    if not isinstance(expansion, int):
        raise TypeError(
            "expansion must be an int. You provided {}".format(expansion))

    algo_avail = ['taylor']

    assert algo in algo_avail

    max_expansion = expansion

    time_evol = copy.deepcopy(base)
    work = copy.deepcopy(base)
    for order in range(1, max_expansion):
        work1 = work.apply(quad_ham)
        work2 = work.apply(diag_coulomb)
        work = work1 + work2
        coeff = (-1.j * time)**order / factorial(order, exact=True)
        time_evol.ax_plus_y(coeff, work)
        if verbose:
            print(f"{order=}", f"{(work.norm() * np.abs(coeff))=}", f"{accuracy=}", f"{time_evol.norm()=}")
        if work.norm() * np.abs(coeff) < accuracy:
            break
    else:
        raise RuntimeError("maximum taylor expansion limit reached")

    return time_evol
