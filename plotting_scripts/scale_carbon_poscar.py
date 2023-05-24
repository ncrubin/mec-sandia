"""
Remove longer dimension from the carbon simulation cell
"""
import copy
import os
import numpy
import numpy as np
import ase
from ase.io import read, write

from mec_sandia.config import VASP_DATA, REPO_DIRECTORY

from ase.units import Bohr

import os
import numpy as np
from mec_sandia.vasp_utils import read_vasp
from mec_sandia.config import VASP_DATA
from mec_sandia.ft_pw_with_projectile import pw_qubitization_with_projectile_costs_from_v5
import math
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False

def main():
    """
    x-axis of the carbon unit cell is double the length 
    """
    fname = os.path.join(VASP_DATA, "C_POSCAR")
    os.chdir(os.path.join(REPO_DIRECTORY, 'vasp_data'))
    ase_atom = read(fname)
    print(ase_atom)

    cut_len = 0.5
    positions = ase_atom.get_scaled_positions()
    mask = [pos[0] < cut_len for pos in positions]
    new_positions = numpy.asarray(
        [pos for pos, mval in zip(ase_atom.get_positions(), mask) if mval])
    new_positions_ncr = []
    for pos, mval in zip(ase_atom.get_positions(), mask):
        if mval: 
            new_positions_ncr.append(pos)
    assert np.allclose(new_positions_ncr, new_positions)

    new_numbers = [num for num, mval in zip(ase_atom.get_atomic_numbers(), mask) if mval]
    if not numpy.any(new_numbers == 1):
        new_numbers[0] = 1

    new_cell = copy.deepcopy(ase_atom.cell)
    new_cell.array[0, 0] /= 2
    new_atoms = ase.Atoms(numbers=new_numbers, positions=new_positions, cell=new_cell)
    print(new_atoms)

    write('C_POSCAR_cubic.vasp', new_atoms, format='vasp', direct=True)



if __name__ == "__main__":
    main()