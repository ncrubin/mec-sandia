import numpy as np

import ase

def compute_wigner_seitz(volume: float, num_valence_elec: int) -> float:
    """Extract wigner seitz radius from ase Atoms object.

    :param volume: Cell volume in Bohr^3.
    :param num_valence_elec: Number of electrons in cell.

    :returns rs: Wigner-Seitz radius
    """
    rs = ((3 * volume) / (4*np.pi*num_valence_elec)) ** (1.0/3.0)
    return rs

def read_vasp(poscar_file):
    """ASE complains about deuterium in poscar file. So handrole
    """
    with open(poscar_file, "r") as fid:
        info = fid.readline()
        lattice_constant = float(fid.readline())
        lattice_vectors = np.zeros((3,3))
        for i in range(3):
            lattice_vectors[i] = [float(x) for x in fid.readline().split()]
        basis_vectors = lattice_constant * lattice_vectors 
        atom_names = fid.readline().split()
        num_atoms = [int(x) for x in fid.readline().split()]
        assert len(num_atoms) == len(atom_names)
        coord_type = fid.readline().strip().lower()
        assert coord_type == "direct"
        tot_atoms = sum(num_atoms)
        coords = []
        D_indx = np.where(atom_names)
        for atom in range(tot_atoms): 
            coords.append([float(x) for x in fid.readline().split()])
        symbols = []
        for sym, num_at in zip(atom_names, num_atoms): 
            symbols += list(sym*num_at)
        symbols_no_D = ["H" if name == "D" else name for name in symbols ]
        D_indx = [indx for indx, name in enumerate(symbols) if name == "D"]
        atoms = ase.Atoms(symbols=symbols_no_D, cell=basis_vectors, pbc=True)
        atoms.set_scaled_positions(coords)
        masses = np.array(atoms.get_masses())
        masses[D_indx] *= 2.0
        atoms.set_masses(masses)
        symbols = np.array(atoms.get_chemical_symbols())
        symbols[D_indx] = "D"
        # This could be overridden but not yet.
        #atoms.set_chemical_symbols(symbols)
        return atoms