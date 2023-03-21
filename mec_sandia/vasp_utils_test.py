import os

import numpy as np
import pytest
from ase.build import bulk
from ase.io import vasp
from ase.units import Bohr

from mec_sandia.vasp_utils import (
    compute_cell_length_from_density,
    compute_wigner_seitz_radius,
    read_kohn_sham_data,
)
from mec_sandia.vasp_utils import read_vasp as local_read_vasp

_test_path = os.path.dirname(os.path.abspath(__file__))

# sampling of systems from Tab. V1 of https://arxiv.org/abs/2105.12767
_test_params = [
    (("Li", "bcc", 284.94, 6, True), 2.25),  # AE
    (["Li", "bcc", 284.94, 2, False], 3.24),  # Valence
    (["K", "bcc", 961.67, 38, True], 1.82),  # AE
    (["FeO", "rocksalt", 539.84, 136, True], 0.98),  # AE
    (["AlAs", "zincblende", 1197.86, 32, False], 2.08),  # Valence
    (["C", "diamond", 307.04, 48, True], 1.15),  # AE
    (["Si", "diamond", 1080.43, 32, False], 2.01),  # Valence
]


@pytest.mark.parametrize("input,expected", _test_params)
def test_compute_rs(input, expected):
    atom, lattice_type, volume, valence, all_elec = input
    # All cubic cells.
    lattice_param = volume ** (1.0 / 3.0)
    cell = bulk(atom, lattice_type, a=lattice_param, cubic=True)
    if all_elec:
        assert sum(cell.get_atomic_numbers()) == valence
    assert np.isclose(volume, cell.get_volume())
    rs = compute_wigner_seitz_radius(cell.get_volume(), valence)
    assert np.isclose(rs, expected, atol=1e-2)
    vasp.write_vasp(f"{atom}.POSCAR", cell)
    cell_vasp = vasp.read_vasp(f"{atom}.POSCAR")
    rs = compute_wigner_seitz_radius(cell.get_volume(), valence)
    assert np.isclose(rs, expected, atol=1e-2)
    try:
        os.remove(f"{atom}.POSCAR")
    except FileNotFoundError:
        pass


_test_params_sandia = [
    ((f"{_test_path}/../vasp_data/C_POSCAR", True), 0.81),
    (
        (f"{_test_path}/../vasp_data/C_POSCAR", False),
        0.93,
    ),  # 1s Carbon electrons are pseudized
]


@pytest.mark.parametrize("input,expected", _test_params_sandia)
def test_compute_rs_sandia_carbon(input, expected):
    filename, all_elec = input
    cell_vasp = vasp.read_vasp(filename)
    num_elec = sum(cell_vasp.get_atomic_numbers())
    if not all_elec:
        num_carbon = len(np.where(cell_vasp.get_atomic_numbers() == 6)[0])
        # Not really clear if we should include the hydrogen atom here?
        num_elec = 1 + num_carbon * 4
    # Note converting to Bohr,
    rs = compute_wigner_seitz_radius(cell_vasp.get_volume() / Bohr**3.0, num_elec)
    assert np.isclose(rs, expected, atol=1e-2)


def test_compute_rs_sandia_deuterium():
    filename = f"{_test_path}/../vasp_data/D_POSCAR"
    cell_vasp = local_read_vasp(filename)
    num_elec = sum(cell_vasp.get_atomic_numbers())
    rs = compute_wigner_seitz_radius(cell_vasp.get_volume() / Bohr**3.0, num_elec)
    assert np.isclose(rs, 0.81, atol=1e-2)


def test_local_reader():
    filename = f"{_test_path}/../vasp_data/C_POSCAR"
    cell_vasp_local = local_read_vasp(filename)
    num_elec = sum(cell_vasp_local.get_atomic_numbers())
    rs = compute_wigner_seitz_radius(cell_vasp_local.get_volume() / Bohr**3.0, num_elec)
    assert np.isclose(rs, 0.81, atol=1e-2)
    cell_vasp = vasp.read_vasp(filename)
    num_elec = sum(cell_vasp.get_atomic_numbers())
    rs = compute_wigner_seitz_radius(cell_vasp.get_volume() / Bohr**3.0, num_elec)
    assert np.isclose(rs, 0.81, atol=1e-2)


def test_box_length_from_target_density():
    target_density = 10  # g / cm^3
    # Deuterium
    res = compute_cell_length_from_density(64, 2, 10)
    assert np.isclose(res, 5.24, atol=1e-2)
    # beryllium
    res = compute_cell_length_from_density(64, 9, 10)
    assert np.isclose(res, 8.65, atol=1e-2)
    # carbon
    res = compute_cell_length_from_density(64, 12, 10)
    assert np.isclose(res, 9.52, atol=1e-2)


def test_read_kohn_sham_data():
    cell_vasp = vasp.read_vasp(f"{_test_path}/../vasp_data/C_POSCAR")
    num_carbon = len(np.where(cell_vasp.get_atomic_numbers() == 6)[0])
    num_elec = (
        1 + num_carbon * 4
    )  # 1s orbitals are pseudized, only 4 electrons / carbon
    eigs, occs = read_kohn_sham_data(f"{_test_path}/../vasp_data/C_10eV_EIGENVAL")
    assert np.isclose(2 * sum(occs), num_elec)
    eigs, occs = read_kohn_sham_data(f"{_test_path}/../vasp_data/C_1eV_EIGENVAL")
    assert np.isclose(2 * sum(occs), num_elec)
    cell_vasp = local_read_vasp(f"{_test_path}/../vasp_data/D_POSCAR")
    num_elec = len(np.where(cell_vasp.get_atomic_numbers() == 1)[0])
    eigs, occs = read_kohn_sham_data(f"{_test_path}/../vasp_data/D_1eV_EIGENVAL")
    assert np.isclose(2 * sum(occs), num_elec)
