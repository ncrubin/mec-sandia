import numpy as np
from density_matrix import (
    fermi_factor,
    find_chemical_potential,
    compute_electron_number,
    DensityMatrix,
)

from ueg import UEG


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
    theta = 1.0
    beta = system.calc_beta_from_theta(theta)
    mu = find_chemical_potential(system.eigenvalues, beta, system.num_elec)
    occs = fermi_factor(system.eigenvalues, mu, beta)
    num_samples = 10000
    np.random.seed(7)
    dm = DensityMatrix.build_grand_canonical(occs, num_samples)
    nav, nav_err = dm.compute_electron_number()
    # A bit dodgy using atol = std_err as only expected to be true ~ 0.6 percent
    # of the time but for this seed it turns out to be ok.
    assert np.isclose(nav, target_num_elec, atol=nav_err)
    fermi, fermi_err = dm.compute_occupations()
    assert np.allclose(fermi, occs, atol=0.02)
    kinetic_energy, kinetic_energy_err = dm.contract_diagonal_one_body(
        system.eigenvalues
    )
    reference_energy = sum(system.eigenvalues * occs)
    assert np.isclose(kinetic_energy, reference_energy, atol=kinetic_energy_err)

def test_exact_grand_canonical():
    target_num_elec = 14
    system = UEG.build(target_num_elec, 1.0, 10)
    theta = 1.0
    beta = system.calc_beta_from_theta(theta)
    mu = find_chemical_potential(system.eigenvalues, beta, system.num_elec)
    occs = fermi_factor(system.eigenvalues, mu, beta)
    num_samples = 10000
    np.random.seed(7)
    dm = DensityMatrix.build_grand_canonical(occs, num_samples)
    nav, nav_err = dm.compute_electron_number()
    # A bit dodgy using atol = std_err as only expected to be true ~ 0.6 percent
    # of the time but for this seed it turns out to be ok.
    assert np.isclose(nav, target_num_elec, atol=nav_err)
    fermi, fermi_err = dm.compute_occupations()
    assert np.allclose(fermi, occs, atol=0.02)
    kinetic_energy, kinetic_energy_err = dm.contract_diagonal_one_body(
        system.eigenvalues
    )
    reference_energy = sum(system.eigenvalues * occs)
    assert np.isclose(kinetic_energy, reference_energy, atol=kinetic_energy_err)