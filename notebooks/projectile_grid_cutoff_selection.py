"""
How fine a mesh does the projectile need?

We can answer this by looking at the Gaussian and comparing the expectation value
to the infinite mesh limit. 

For a Gaussian 

E[x^2] = sigma^2 
in 1D if centered around zero.

In 3D
E[r^2] = 3sigma^2 
for an isotropic Gaussian.

Assuming a square box we have.
<T> = (int p(x) x^2) * (2pi)^2/ L^2 / (2 * m)
    = E[x^2] * (2pi)^2/ L^2 / (2 * m)

where the expectation value is evaluated by summation 
or in the infinite k-limit it is just sigma^2 (for zero centered gaussian).
"""

import numpy as np
from mec_sandia.pw_basis import SquareBoxPlanewaves
import matplotlib.pyplot as plt
from pyscf.lib.numpy_helper import cartesian_prod

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']


def main():
    L = 15 # box length in bohr
    vol = L**3
    a = np.eye(3) * L
    m = 1836.

    # 1-D Problem
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # set up grid number of points along a single dimension
    # 2 * pi * x / L
    nx_grid = 2**np.arange(2, 6)  # quantum algorithm takes things as a power of 2
    ke_cutoffs_eV = (2 * np.pi)**2 * nx_grid**2 / L**2 * 27.11 # highest energy components in eV
    print(ke_cutoffs_eV)
    # variance for gaussian
    sigma_squared = [1, 2, 4, 6]
    for idx, ss in enumerate(sigma_squared):
        k2_expectation_diff = []
        for nx in nx_grid:
            # note the grid is indexed by |p>
            kgrid = np.arange(-nx/2, nx/2)
            norm_constant = 1 / (np.sqrt(2 * np.pi) * np.sqrt(ss))
            ygrid = norm_constant * np.exp(-0.5 * kgrid**2 / ss)

            # grid spacing is 1.
            discrete_rho_xsquared = np.sum(ygrid * kgrid**2) * (2 * np.pi)**2 / L**2 / (2 * m)  # extra (2pi)^2 / L^2 is for converting from int to wavenumber
            exact_rho_xsquared = ss * (2 * np.pi)**2 / L**2 / (2 * m)
            print(discrete_rho_xsquared, exact_rho_xsquared)
            k2_expectation_diff.append(np.abs(np.sqrt(discrete_rho_xsquared) - np.sqrt(exact_rho_xsquared)) * 1000) # times 1K for milliHartree
        print(k2_expectation_diff)
        ax.plot(ke_cutoffs_eV, k2_expectation_diff, color=colors[idx], label=r"$\sigma = {}$".format(ss), marker='o')
        
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$E_{cut}$ [eV]", fontsize=14)
    ax.set_ylabel(r"Projectile Kinetic Energy Error [mHa]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
    ax.set_title("One Dimensional Gaussian Kinetic Energy standard Error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("one_D_Gaussian_Variance_convergence.png", format='PNG', dpi=300)
    plt.show()

    sigma_squared = 2
    nx = 2**7
    kgrid = np.arange(-nx/2, nx/2)
    norm_constant = 1 / (np.sqrt(2 * np.pi) * np.sqrt(sigma_squared))
    ygrid = norm_constant * np.exp(-0.5 * kgrid**2 / sigma_squared)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(kgrid, ygrid)
    plt.savefig("gaussian.png", format='PNG', dpi=300)


   # 3-D Problem
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ke_cutoffs_eV = 3 * (2 * np.pi)**2 * nx_grid**2 / L**2 * 27.11
    print(ke_cutoffs_eV)
    sigma_squared = [1, 2, 4, 6]
    for idx, ss in enumerate(sigma_squared):
        k2_expectation_diff = []
        for nx in nx_grid:
            kgrid = np.arange(-nx/2, nx/2)
            nxyz_grid = cartesian_prod([kgrid, kgrid, kgrid])
            norm_constant = 1 / (np.sqrt(2 * np.pi)**3 * np.sqrt(ss)**3)
            ygrid = norm_constant * np.exp(-0.5 * np.sum(nxyz_grid**2, axis=-1) / ss)

            discrete_rho_xsquared = np.sum(ygrid * np.sum(nxyz_grid**2, axis=-1)) * (2 * np.pi)**2 / L**2 / (2 * m)
            exact_rho_xsquared = 3 * ss * (2 * np.pi)**2 / L**2 / (2 * m)
            k2_expectation_diff.append(np.abs(np.sqrt(discrete_rho_xsquared) - np.sqrt(exact_rho_xsquared)) * 1000) # times 1K for milliHartree
            print(np.sum(ygrid))
        print(k2_expectation_diff)
        ax.plot(ke_cutoffs_eV, k2_expectation_diff, color=colors[idx], label=r"$\sigma = {}$".format(ss), marker='o')
        
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$E_{cut}$ [eV]", fontsize=14)
    ax.set_ylabel(r"Projectile Kinetic Energy Error [mHa]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
    ax.set_title("Three Dimensional Gaussian Kinetic Energy standard Error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("three_D_Gaussian_Variance_convergence.png", format='PNG', dpi=300)
    plt.show()

    nx_grid = 2**np.arange(3, 8)
    sigma_squared = 4
    for nx_grid_val in nx_grid:
        sbp = SquareBoxPlanewaves(a, nx_grid_val)
        p_basis = sbp.get_p_basis()
        kp = sbp.get_kp_basis()
        ke_vals = np.sum(kp**2, axis=-1) / (2 * m)
        ke_val_test = sbp.get_ke_diagonal_values(mass=m)
        assert np.allclose(ke_vals, ke_val_test)
        print("Max Ke-proj x-direction ", ke_vals[0])
        print("Max Ke-e x-direction ", np.sum(kp[0]**2) / 2 * 27.11) 
        print("Total Planewaves ", sbp.N)

        sigma = np.sqrt(sigma_squared)
        norm_constant = 1 / ((2 * np.pi)**(3/2) * sigma**(3))
        prob_dist = norm_constant * np.exp(-np.sum(p_basis**2, axis=-1)/(2 * sigma_squared))
        print(np.sum(prob_dist))
        print(np.sum(prob_dist * ke_vals) / (2 * m))
        print()


if __name__ == "__main__":
    main()