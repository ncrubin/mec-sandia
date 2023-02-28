from ase.units import Bohr, Hartree
import numpy as np
import os
import pytest

from density_matrix import (
    fermi_factor,
    find_chemical_potential,
    compute_electron_number,
    DensityMatrix,
)
from vasp_utils import read_kohn_sham_data, read_vasp, compute_wigner_seitz
from ueg import UEG, calc_beta_from_theta, calc_fermi_energy, calc_theta_from_beta


def test_chemical_potential():
    target_num_elec = 14
    system = UEG.build(target_num_elec, 1.0, 10)
    theta = 1.0
    beta = system.calc_beta_from_theta(theta)
    mu = find_chemical_potential(system.eigenvalues, beta, system.num_elec)
    test_nelec = compute_electron_number(mu, system.eigenvalues, beta)
    assert np.isclose(test_nelec, target_num_elec)


def test_sampling_grand_canonical():
    target_num_elec = 14
    system = UEG.build(target_num_elec, 1.0, 10)
    theta = 0.125
    beta = system.calc_beta_from_theta(theta)
    mu = find_chemical_potential(system.eigenvalues, beta, system.num_elec)
    occs = fermi_factor(system.eigenvalues, mu, beta)
    num_samples = 10000
    np.random.seed(7)
    dm = DensityMatrix.build_grand_canonical(occs, num_samples)
    nav, nav_err = dm.compute_electron_number()
    assert np.isclose(nav, target_num_elec, atol=0.02)
    fermi, fermi_err = dm.compute_occupations()
    assert np.allclose(fermi[::2], occs, atol=0.02)
    assert np.allclose(fermi[1::2], occs, atol=0.02)
    kinetic_energy, kinetic_energy_err = dm.contract_diagonal_one_body(
        system.eigenvalues
    )
    reference_energy = 2 * sum(system.eigenvalues * occs)
    assert np.isclose(kinetic_energy, reference_energy, atol=2*kinetic_energy_err)

def test_exact_grand_canonical():
    target_num_elec = 4
    system = UEG.build(target_num_elec, 1.0, 0.5)
    theta = 1.0
    beta = system.calc_beta_from_theta(theta)
    mu = find_chemical_potential(system.eigenvalues, beta, system.num_elec)
    occs = fermi_factor(system.eigenvalues, mu, beta)
    dm = DensityMatrix.build_grand_canonical_exact(system.eigenvalues, mu, beta)
    nav, nav_err = dm.compute_electron_number()
    assert np.isclose(nav, target_num_elec, atol=nav_err)
    fermi, fermi_err = dm.compute_occupations()
    assert np.allclose(fermi[::2], occs)
    assert np.allclose(fermi[1::2], occs)
    kinetic_energy, kinetic_energy_err = dm.contract_diagonal_one_body(
        system.eigenvalues
    )
    reference_energy = 2 * sum(system.eigenvalues * occs)
    assert np.isclose(kinetic_energy, reference_energy, atol=2*kinetic_energy_err)

def test_sampling_canonical():
    target_num_elec = 4
    system = UEG.build(target_num_elec, 1.0, 0.5)
    theta = 0.5
    beta = system.calc_beta_from_theta(theta)
    mu = find_chemical_potential(system.eigenvalues, beta, system.num_elec)
    occs = fermi_factor(system.eigenvalues, mu, beta)
    num_samples = 10000
    np.random.seed(7)
    dm = DensityMatrix.build_canonical(occs, num_samples, target_num_elec)
    nav, nav_err = dm.compute_electron_number()
    assert np.isclose(nav, target_num_elec)
    dm_exact = DensityMatrix.build_canonical_exact(system.eigenvalues, beta, target_num_elec)
    assert np.isclose(nav, target_num_elec)
    fermi, fermi_err = dm.compute_occupations()
    ref_occ, _ = dm_exact.compute_occupations()
    assert np.allclose(fermi, ref_occ, atol=0.02)
    kinetic_energy, kinetic_energy_err = dm.contract_diagonal_one_body(
        system.eigenvalues
    )
    ref_kin, _ = dm_exact.contract_diagonal_one_body(system.eigenvalues)
    assert np.isclose(kinetic_energy, ref_kin, atol=2*kinetic_energy_err)

_test_path = os.path.dirname(os.path.abspath(__file__))
_test_params_sandia = [
    (10, 0.17220781705), (1, 0.017220781705)
]
@pytest.mark.parametrize("input,expected", _test_params_sandia)
def test_compute_fermi_temperature(input, expected):
    temp = input # eV
    cell_vasp = read_vasp(f"{_test_path}/vasp_data/C_POSCAR")
    num_carbon = len(np.where(cell_vasp.get_atomic_numbers() == 6)[0])
    num_elec = 1 + num_carbon * 4
    rs = compute_wigner_seitz(cell_vasp.get_volume() / Bohr**3.0, num_elec)
    fermi_energy = calc_fermi_energy(rs)
    assert np.isclose(fermi_energy, 2.134010122410822)
    T_Ha = temp / Hartree
    beta = 1.0 / T_Ha
    theta = calc_theta_from_beta(beta, rs)
    assert np.isclose(theta, expected)

_test_path = os.path.dirname(os.path.abspath(__file__))
_test_params_sandia = [
    (10), (1)
]
@pytest.mark.parametrize("temp", _test_params_sandia)
def test_compute_fermi_temperature(temp):
    cell_vasp = read_vasp(f"{_test_path}/vasp_data/C_POSCAR")
    num_carbon = len(np.where(cell_vasp.get_atomic_numbers() == 6)[0])
    num_elec = 1 + num_carbon * 4
    eigs, occs = read_kohn_sham_data(f"{_test_path}/vasp_data/C_{temp}eV_EIGENVAL")
    num_samples = 10000
    # Test grand canonical ensemble
    target_num_elec = num_elec
    np.random.seed(7)
    dm = DensityMatrix.build_grand_canonical(occs, num_samples)
    nav, nav_err = dm.compute_electron_number()
    assert np.isclose(nav, target_num_elec, atol=2*nav_err)
    fermi, fermi_err = dm.compute_occupations()
    assert np.allclose(fermi[::2], occs, atol=0.02)
    assert np.allclose(fermi[1::2], occs, atol=0.02)
    kinetic_energy, kinetic_energy_err = dm.contract_diagonal_one_body(
        eigs 
    )
    reference_energy = 2 * sum(eigs * occs)
    assert np.isclose(kinetic_energy, reference_energy, atol=2*kinetic_energy_err)
    # Test canonical ensemble
    target_num_elec = num_elec
    dm = DensityMatrix.build_canonical(occs, num_samples, target_num_elec)
    nav, nav_err = dm.compute_electron_number()
    # Should be exactly right
    assert np.isclose(nav, target_num_elec)
    fermi, _ = dm.compute_occupations()
    # Should not agree very well.
    assert not np.allclose(fermi[::2], occs, atol=0.02)
    assert not np.allclose(fermi[1::2], occs, atol=0.02)