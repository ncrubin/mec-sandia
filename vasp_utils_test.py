import numpy as np
import os
import pytest

from ase.build import bulk
from ase.io import vasp
from ase.units import Bohr

from vasp_utils import compute_wigner_seitz

_test_path = os.path.dirname(os.path.abspath(__file__))

# sampling of systems from Tab. V1 of https://arxiv.org/abs/2105.12767
_test_params = [
    (("Li", "bcc", 284.94, 6, True), 2.25), # AE
    (["Li", "bcc", 284.94, 2, False], 3.24), # Valence
    (["K", "bcc", 961.67, 38, True], 1.82), # AE
    (["FeO", "rocksalt", 539.84, 136, True], 0.98), # AE
    (["AlAs", "zincblende", 1197.86, 32, False], 2.08), # Valence
    (["C", "diamond", 307.04, 48, True], 1.15), # AE
    (["Si", "diamond", 1080.43, 32, False], 2.01), # Valence
    ]
@pytest.mark.parametrize("input,expected", _test_params)
def test_compute_rs(input, expected):
    atom, lattice_type, volume, valence, all_elec = input
    # All cubic cells.
    lattice_param = volume ** (1.0/3.0)
    cell = bulk(atom, lattice_type, a=lattice_param, cubic=True)
    if all_elec:
        assert sum(cell.get_atomic_numbers()) == valence
    assert np.isclose(volume, cell.get_volume())
    rs = compute_wigner_seitz(cell.get_volume(), valence)
    assert np.isclose(rs, expected, atol=1e-2)
    vasp.write_vasp(f"{atom}.POSCAR", cell)
    cell_vasp = vasp.read_vasp(f"{atom}.POSCAR")
    rs = compute_wigner_seitz(cell.get_volume(), valence)
    assert np.isclose(rs, expected, atol=1e-2)
    try:
        os.remove(f"{atom}.POSCAR")
    except FileNotFoundError:
        pass

# @pytest.mark.parametrize("input,expected", [(f"{_test_path}/vasp_data/C_POSCAR", 1.0), (
_test_params_sandia = [
    ((f"{_test_path}/vasp_data/C_POSCAR", True), 0.81),
    ((f"{_test_path}/vasp_data/C_POSCAR", False), 0.93), # 1s Carbon electrons are pseudized
    ]
@pytest.mark.parametrize("input,expected", _test_params_sandia)
def test_compute_rs_sandia_carbon(input, expected):
    filename, all_elec = input 
    cell_vasp = vasp.read_vasp(filename)
    num_elec = sum(cell_vasp.get_atomic_numbers())
    if not all_elec:
        num_carbon = len(np.where(cell_vasp.get_atomic_numbers()==6)[0])
        # Not really clear if we should include the hydrogen atom here?
        num_elec = 1 + num_carbon * 4
    # Note converting to Borh, 
    rs = compute_wigner_seitz(cell_vasp.get_volume()/Bohr**3.0, num_elec)
    assert np.isclose(rs, expected, atol=1e-2)

# def test_compute_rs_sandia_deuterium():
#     filename = f"{_test_path}/vasp_data/D_POSCAR"
#     cell_vasp = vasp.read_vasp(filename)
#     num_elec = sum(cell_vasp.get_atomic_numbers())
#     rs = compute_wigner_seitz(cell_vasp, num_elec)
#     assert np.isclose(rs, 1.0, atol=1e-2)