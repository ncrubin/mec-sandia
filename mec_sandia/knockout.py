"""
Experiments with Robin's method

The goal is to make the unitary that Robin does QPE on 
to then confirm the assertions in the paper

This example we will do Ry(2 * theta) to produce sqrt(1 - p)|0> + sqrt(p)|1>
as the synthesizer

y(0) = 0, y(1) = 1
alpha(0) = 0, alpha(1) = -pi/2
"""

import numpy as np
import cirq
from cirq import ZPowGate

P0 = np.array([[1, 0], [0, 0]])
P1 = np.array([[0, 0], [0, 1]])

def build_ry_synthesizer(p, qubit):
    """
    :param float p: probability of measuring 1
    """
    theta = np.arccos(np.sqrt(1- p))
    return cirq.Circuit([cirq.ry(2 * theta).on(qubit)])

def main():
    qubits = cirq.LineQubit.range(2)
    prob_1 = 1/3
    synthesizer_circuit = build_ry_synthesizer(prob_1, qubits[0])
    wf = synthesizer_circuit.final_state_vector().reshape((2, -1))
    assert np.isclose(wf.conj().T @ P1 @ wf, prob_1)

    # one-qubit reflection around zero is Z gate
    refl_p = cirq.inverse(synthesizer_circuit) + cirq.Z.on(qubits[0]) + synthesizer_circuit

    # let's take y(0) = 0, y(1) = 1 which means automatically E[y^2] <= 1 is satisfied
    # alpha(0) = 0, alpha(1) = -pi/2
    # define rot_y|l> = e^{i alpha_l}|l>
    # rot_y = ((1, 0), (0, -1j)) = cirq.ZPowGate(exponent=-0.5, global_shift=0)
    rot_y = cirq.Circuit(ZPowGate(exponent=-0.5, global_shift=0).on(qubits[0])) 

    ko_unitary = cirq.unitary(rot_y + refl_p)

    w, v = np.linalg.eig(ko_unitary)
    ko_eigen_probs = np.array([abs(wf.conj().T @ v[:, 0])**2, abs(wf.conj().T @ v[:, 1])**2])
    assert np.isclose(np.sum(ko_eigen_probs), 1)
    print(np.abs(np.angle(w))/ 2)
    print("expected value ", wf.conj().T @ P1 @ wf)
    print("s2 ",  )


if __name__ == "__main__":
    main()