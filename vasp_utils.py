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