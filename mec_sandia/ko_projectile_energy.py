"""
Code for using the KO algorithm to measure the projectile kinetic energy.

The synthesizer is the time-evolution unitary
the encoder is a circuit that maps to a binary representation of the kinetic energy.
"""
from functools import cached_property
from typing import Sequence, Tuple
from attrs import frozen
import numpy as np
import cirq

from cirq_qubitization.cirq_algos.select_and_prepare import PrepareOracle, SelectOracle
import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_algos.mean_estimation import (
    CodeForRandomVariable,
    MeanEstimationOperator,
)

from mec_sandia.ft_pw_with_projectile import ToffoliCostBreakdown


@frozen
class TimeEvolutionSynthesizer(PrepareOracle):
    r"""Synthesizes the state that is the ouput of the time evolution"""

    block_encoding_costs: ToffoliCostBreakdown
    evolution_time: float

    def __pow__(self, power):
        return self

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        r"""selection bits is the number of qubits used for the nuclear wavepacket.
            selection length is the exponential of this.
        """
        num_qubits = 3 * self.block_encoding_costs.nn
        return cq.SelectionRegisters.build(q=(num_qubits, 2**num_qubits))

    @cached_property
    def junk_registers(self) -> cq.Registers:
        r"""Junk registers is everything else involved in qubitized time-evolution build"""
        return cq.Registers.build(
            garbage=self.block_encoding_costs.qc_total - 3 * self.block_encoding_costs.nn,
    ) 

    # def decompose_from_registers(self, *args, **kwargs) -> None:
    #     yield cirq.Circuit()

    def _t_complexity_(self) -> cq.TComplexity:
        r"""Number of queries is taken from the appendix of PhysRevA.99.040301"""
        lambda_by_time = np.abs(self.evolution_time) * self.block_encoding_costs.lambda_total
        eps_ph = self.block_encoding_costs.target_qpe_eps
        total_queries = 2 * (lambda_by_time + 1.04 * (lambda_by_time)**(1/3)) * np.log2(1/eps_ph)**(2/3)
        # standard 7-T per Toffoli
        # print(f"{self.block_encoding_costs.tofc_total=}")
        # print(f"{total_queries=}")
        return cq.TComplexity(t=7 * self.block_encoding_costs.tofc_total * total_queries, clifford=0)

@frozen
class ProjectileKineticEnergyEncoder(SelectOracle):
    r"""Encodes integer kinetic energy shifted to it's expected mean"""

    block_encoding_costs: ToffoliCostBreakdown

    def __pow__(self, power):
        return self

    @cached_property
    def control_registers(self) -> cq.Registers:
        return cq.Registers([])

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        num_qubits = 3 * self.block_encoding_costs.nn
        return cq.SelectionRegisters.build(q=(num_qubits, 2**num_qubits))

    @cached_property
    def target_registers(self) -> cq.Registers:
        nmean = self.get_nmean()
        nproj = self.block_encoding_costs.nn
        if nmean < nproj:
            nmean = nproj
        nf = 2 * nmean - 1
        return cq.Registers.build(t=nf)

    # def decompose_from_registers(self, *args, **kwargs) -> None:
    #     yield None

    def get_nmean(self) -> int:
        r"""Compute the num-bits needed to represent the k_mean value"""
        integer_kmean = np.ceil(self.block_encoding_costs.kmean / (self.block_encoding_costs.Omega**1/3) / (2 * np.pi))
        return np.floor(np.log2(integer_kmean)) + 1

    def _three_sums_helper(self, register_bitsize: int) -> int:
        return 3 * register_bitsize**2 - register_bitsize - 1

    def _t_complexity_(self) -> cq.TComplexity:
        nproj = self.block_encoding_costs.nn
        nmean = self.get_nmean()
        if nmean < nproj:
            nmean = nproj
        nf = 2 * nmean - 1

        step1 = self._three_sums_helper(nproj)
        step2 = 3 * (2 * nmean * nproj - nmean) + self._three_sums_helper(nmean)
        step3 = self._three_sums_helper(nf) 

        # standard 7-T per Toffoli
        return cq.TComplexity(t=7 * (step1 + step2 + step3), clifford=0)

def construct_mean_estimation_operator(block_encoding_costs: ToffoliCostBreakdown, evolution_time=1, arctan_bitsize=64):
    synthesizer = TimeEvolutionSynthesizer(block_encoding_costs=block_encoding_costs, 
                                           evolution_time=evolution_time)
    encoder = ProjectileKineticEnergyEncoder(block_encoding_costs=block_encoding_costs)
    code = CodeForRandomVariable(synthesizer=synthesizer, encoder=encoder)
    mean_op = MeanEstimationOperator(code, arctan_bitsize=arctan_bitsize)

    # print(cq.t_complexity(synthesizer))
    # print(cq.t_complexity(encoder))
    # print(cq.t_complexity(mean_op))
    return mean_op


def get_resource_state(m: int):
    """Returns a state vector representing the resource state on m qubits from Eq.17 of Ref-1.
    
    Returns a numpy array of size 2^{m} representing the state vector corresponding to the state
    $$
        \sqrt{\frac{2}{2^m + 1}} \sum_{n=0}^{2^{m}-1} \sin{\frac{\pi(n + 1)}{2^{m}+1}}\ket{n}
    $$
    
    Args:
        m: Number of qubits to prepare the resource state on.
    
    Ref:
        1) [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
            (https://arxiv.org/abs/1805.03662)
            Eq. 17
    """
    den = 1 + 2 ** m
    norm = np.sqrt(2 / den)
    return norm * np.sin(np.pi * (1 + np.arange(2**m)) / den)        
    
def phase_estimation_for_mean_estimation(mean_estimation_op: MeanEstimationOperator, mean_eps: float) -> cirq.OP_TREE:
    """phase estimation the mean estimiation operator from KO algorithm.
    
    The method yields an OPTREE to construct Heisenberg limited phase estimation circuit 
    for learning eigenphases of the KO operator with `m` bits of accuracy dictationed by
    the input precision. 
    
    Args:
        mean_estimation_op: KO algorithm unitary
        mean_eps: decisired precision
    """
    qpe_eps = mean_eps / 6
    m = int(np.ceil(np.log2(2 * np.pi / qpe_eps)))
    mean_op_regs = mean_estimation_op.registers.get_named_qubits()

    m_qubits = [cirq.q(f'm_{i}') for i in range(m)]
    # state_prep = cirq.StatePreparationChannel(get_resource_state(m), name='𝜒_m')

    # yield state_prep.on(*m_qubits)
    for i in range(m):
        for jj in range(2**i):
            yield mean_estimation_op.on_registers(**mean_op_regs, control=m_qubits[i])
    yield cirq.qft(*m_qubits, inverse=True)


if __name__ == "__main__":
    # Let's read in the Carbon example provided by Sandia
    from mec_sandia.vasp_utils import read_vasp
    from mec_sandia.config import VASP_DATA
    from mec_sandia.ft_pw_with_projectile import pw_qubitization_with_projectile_costs_from_v5
    import os
    
    ase_cell = read_vasp(os.path.join(VASP_DATA, "H_2eV_POSCAR"))
    # Next we can get some system paramters
    volume_ang = ase_cell.get_volume()
    
    # To compute rs parameter we need volume in Bohr
    from ase.units import Bohr
    volume_bohr = volume_ang / Bohr**3
    # and the number of valence electrons
    num_elec = np.sum(ase_cell.get_atomic_numbers())
    num_nuclei = len(np.where(ase_cell.get_atomic_numbers() == 1)[0])
    from mec_sandia.vasp_utils import compute_wigner_seitz_radius
    # Get the Wigner-Seitz radius
    rs = compute_wigner_seitz_radius(volume_bohr, num_elec)
   
    num_bits_momenta = 6 # Number of bits in each direction for momenta
    eps_total = 1e-3 # Total allowable error
    num_bits_nu = 8 # extra bits for nu 

    _, _, _, tofc_breakdown = \
    pw_qubitization_with_projectile_costs_from_v5(np=num_bits_momenta, 
                                                  nn=num_bits_momenta,
                                                  eta=num_elec, 
                                                  Omega=volume_bohr, 
                                                  eps=eps_total, 
                                                  nMc=num_bits_nu,
                                                  nbr=20, 
                                                  L=num_nuclei, 
                                                  zeta=2,
                                                  mpr=4000,
                                                  kmean=7000,
                                                  phase_estimation_costs=False,
                                                  return_subcosts=True)
    
    synthesizer = TimeEvolutionSynthesizer(block_encoding_costs=tofc_breakdown, evolution_time=1)
    encoder = ProjectileKineticEnergyEncoder(block_encoding_costs=tofc_breakdown)

    mean_op = construct_mean_estimation_operator(block_encoding_costs=tofc_breakdown, evolution_time=1)
    circuit = cirq.Circuit(phase_estimation_for_mean_estimation(mean_op, 1.0E-2))

    print(f"{cq.t_complexity(mean_op)=}")
    print(f"{cq.t_complexity(circuit[:-1])=}")