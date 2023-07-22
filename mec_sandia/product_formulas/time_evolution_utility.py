"""
Time Evolution utiltiy
"""
import copy
import numpy as np
import fqe
from fqe.hamiltonians import hamiltonian
NORM_ERROR_RESOLUTION = 1.0E-12

def apply_unitary_wrapper(base: fqe.Wavefunction, time: float, algo, ops: hamiltonian.Hamiltonian, 
                          accuracy=1.0E-20, expansion=200, verbose=True,
                          smallest_time_slice=0.5, debug=False) -> fqe.Wavefunction:
    upper_bound_norm = 0
    norm_of_coeffs = []
    for op_coeff_tensor in ops.iht(time):
        upper_bound_norm += np.sum(np.abs(op_coeff_tensor))
        norm_of_coeffs.append(np.linalg.norm(op_coeff_tensor))

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

