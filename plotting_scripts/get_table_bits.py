import numpy as np
from ase.units import Hartree

box_length = 15
ecut_ev = np.logspace(1, 5, 8)
mass_proj = 1836 * 4
prefactor = 1.0 / mass_proj
from mec_sandia.gaussians import (estimate_error_kinetic_energy, get_ngmax,
                                  kinetic_energy)

nbits = [2**n for n in range(2, 10)]
fac = 2*np.pi / box_length
ecut_ha = [0.5  * fac**2.0 * n**2.0 for n in nbits]
ndim = 3
v_proj = 4.0  # atomic units just taken from carbon
ke = 0.5 * mass_proj * v_proj**2.0  # classical ke
kproj = np.array([mass_proj * v_proj, 0, 0])
print(kproj)

def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    return f"${a} \\times 10^{{{b}}}$"


for isk, sigma_k in enumerate([1, 4, 6, 10]):
    for ec in ecut_ha:
        nmax = get_ngmax(ec, box_length)
        ke = prefactor*kinetic_energy(ec, box_length, sigma_k, ndim=ndim, kproj=kproj)
        ke_int = prefactor*(ndim*sigma_k**2 + np.dot(kproj, kproj)) /2
        # The factor of 1/2 is because of how we're using ecut, we only sum from
        # [-nmax/2, nmax/2] which means we should inegrate from [-kmax/2,
        # kmax/2]. (kmax = (2Ecut)^{1/2}) is a spherical cutoff.
        kcut = 0.5*(2*ec)**0.5
        ke_err_apprx = prefactor*estimate_error_kinetic_energy(kcut, sigma_k)
        nbits = np.ceil(np.log2(nmax))
        if np.abs(ke-ke_int) < 1e-3:
            print(f"{sigma_k} & {sci_notation(ec*Hartree, 1)} & {sci_notation(nmax**ndim, 1)} & {nbits} & {sci_notation(np.abs(ke-ke_int), 1)} \\\\")
            break