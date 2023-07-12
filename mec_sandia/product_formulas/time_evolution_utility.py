"""
Time Evolution utiltiy
"""
import copy
import numpy as np
import fqe
from fqe.hamiltonians import hamiltonian

def apply_unitary_wrapper(base: fqe.Wavefunction, time: float, algo, ops: hamiltonian.Hamiltonian, 
                          accuracy=1.0E-20, expansion=200, verbose=True,
                          smallest_time_slice=0.5) -> fqe.Wavefunction:
    upper_bound_norm = 0
    for op_coeff_tensor in ops.iht(time):
        upper_bound_norm += np.sum(np.abs(op_coeff_tensor))
    if upper_bound_norm >= smallest_time_slice:
        num_slices = int(np.ceil(upper_bound_norm / smallest_time_slice))
        time_evol = copy.deepcopy(base)
        # print("Using {} slices to evolve for {} time".format(num_slices, time))
        for mm in range(num_slices):
            time_evol = time_evol.apply_generated_unitary(
                time=time / num_slices, algo=algo, ops=ops, accuracy=accuracy, expansion=expansion,
                                                verbose=verbose
                                                )
    else:
        time_evol = base.apply_generated_unitary(
            time=time, algo=algo, ops=ops, accuracy=accuracy, expansion=expansion,
                                            verbose=verbose
                                            )
    if time_evol.norm() - 1. > 1.0E-14:
        print(f"{time_evol.norm()=}", f"{(time_evol.norm() - 1.)=}")
        raise RuntimeError("Evolution did not converge")
    
    return time_evol

