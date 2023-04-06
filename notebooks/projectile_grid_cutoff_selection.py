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
    v_proj = 4.0 # atomic units just taken from carbon
    mass_proj = 1836
    ke = 0.5 * mass_proj * v_proj**2.0 # classical ke
    kproj = np.array([mass_proj*v_proj, 0, 0])
    kproj_x = mass_proj * v_proj
    p_proj = int(np.ceil(kproj_x * L / 2 / np.pi))


    fig, ax = plt.subplots(nrows=1, ncols=1)
    nx = 100
    ss = 6
    kgrid = np.arange(-nx/2, nx/2) * 2 * np.pi / L
    ygrid = np.exp(-0.5 * kgrid**2 / ss)
    norm_constant = np.sum(ygrid)
    ygrid /= norm_constant
    # ke = (kgrid)**2 / (2 * m)
    ke = (kgrid - kproj_x)**2 / (2 * m)
    print("<T_proj> = ", np.sum(ke * ygrid))
    ax.plot(kgrid, ygrid)
    ax2 = ax.twinx()
    ax2.plot(kgrid, ke, 'C1-')
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$k_{p}$ [$B^{-1}$]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("gaussian.png", format='PNG', dpi=300)
    # plt.show()
    # exit()


    # 1-D Problem
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # set up grid number of points along a single dimension
    # 2 * pi * x / L.  
    nx_grid = 2**np.arange(2, 9)  # quantum algorithm takes things as a power of 2
    ke_cutoffs_eV = (2 * np.pi)**2 * nx_grid**2 / L**2 * 27.11 / 2 # highest energy components in eV
    print(ke_cutoffs_eV)
    # variance for gaussian
    sigma_squared = [4, 16, 36, 100]
    for idx, ss in enumerate(sigma_squared):
        k2_expectation_diff = []
        for ix, nx in enumerate(nx_grid):
            # note the grid is indexed by |p>
            kgrid = np.arange(-nx/2, nx/2)
            true_norm_constant = 1 / (np.sqrt(2 * np.pi) * np.sqrt(ss))
            ygrid = np.exp(-0.5 * ((2 * np.pi)**2 / L**2) * kgrid**2 / ss)
            norm_constant = np.sum(ygrid)
            ygrid /= norm_constant
            
            # diff of true norm constant
            assert np.isclose(np.sum(ygrid), 1)


            # grid spacing is 1.
            discrete_rho_xsquared = np.sum(ygrid * (kgrid - p_proj)**2) * (2 * np.pi)**2 / L**2 / (2 * m)  # extra (2pi)^2 / L^2 is for converting from int to wavenumber
            exact_rho_xsquared = (ss + p_proj**2 * (2 * np.pi)**2 / L**2) / (2 * m)
            #print(discrete_rho_xsquared, exact_rho_xsquared)
            k2_expectation_diff.append(np.abs(np.sqrt(discrete_rho_xsquared) - np.sqrt(exact_rho_xsquared))) # times 1K for milliHartree
            print("ecut = ", ke_cutoffs_eV[ix])
            print("nmax = ", nx)
            print("min k = ", min(*kgrid))
            print("ke exact = ", exact_rho_xsquared)
            print("ke sum= ", discrete_rho_xsquared)
            print("norm exact = ", true_norm_constant)
            print("norm sum= ", norm_constant)
            print("sigma = ", ss**0.5)
            print(k2_expectation_diff)
        ax.plot(ke_cutoffs_eV, k2_expectation_diff, color=colors[idx], label=r"$\sigma^{{2}} = {}$".format(ss), marker='o')

    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$E_{cut}$ [eV]", fontsize=14)
    ax.set_ylabel(r"Projectile Kinetic Energy Error [Ha]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
    ax.set_title("One Dimensional Gaussian Kinetic Energy standard Error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.ylim([1e-11, 1e-2])
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("one_D_Gaussian_Variance_convergence.png", format='PNG', dpi=300)
    #plt.show()

    print("PLOT 2")
    # 1-D Fionn's grid
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # set up grid number of points along a single dimension
    # 2 * pi * x / L.  
    nx_grid = 2**np.arange(2, 6)  # quantum algorithm takes things as a power of 2
    ke_cutoffs = np.array([2000, 4000, 8000, 12_000]) / 27.11 # eV
    ke_cutoffs = np.array([2000, 4000, 8000, 12_000]) / 27.11 # eV
    ke_cutoffs_eV = np.array(ke_cutoffs) * 27.11 # highest energy components in eV
    
    ke_cutoffs = np.array([36.749322474956635, 367.49322474956637, 1837.4661237478317, 3674.9322474956634, 18374.661237478318])
    nx_grid = np.asarray(np.ceil(np.sqrt(2 * ke_cutoffs * L**2 / (2 * np.pi)**2)), dtype=int)
    print(ke_cutoffs_eV)
    # variance for gaussian
    sigma_squared = [16, 36, 100]
    for idx, ss in enumerate(sigma_squared):
        k2_expectation_diff = []
        for nx in nx_grid:
            # note the grid is indexed by |p>
            kgrid = np.arange(-nx/2, nx/2)
            # norm_constant = 1 / (np.sqrt(2 * np.pi) * np.sqrt(ss))
            ygrid = np.exp(-0.5 * ((2 * np.pi)**2 / L**2) * kgrid**2 / ss)
            norm_constant = np.sum(ygrid)
            ygrid /= norm_constant

            # grid spacing is 1.
            factor = 2 * np.pi / L
            discrete_rho_xsquared = np.sum(ygrid * (kgrid * factor - p_proj * factor)**2) / (2 * m)  # extra (2pi)^2 / L^2 is for converting from int to wavenumber
            exact_rho_xsquared = (ss + p_proj**2) * (2 * np.pi)**2 / L**2 / (2 * m)
            print(discrete_rho_xsquared, exact_rho_xsquared)
            k2_expectation_diff.append(np.abs(np.sqrt(discrete_rho_xsquared) - np.sqrt(exact_rho_xsquared)))
        print(k2_expectation_diff)
        ax.plot(ke_cutoffs, k2_expectation_diff, color=colors[idx], label=r"$\sigma^{{2}} = {}$".format(ss), marker='o')
        
    # ax.tick_params(which='both', labelsize=14, direction='in')
    # ax.set_xlabel("$E_{cut}$ [eV]", fontsize=14)
    # ax.set_ylabel(r"Projectile Kinetic Energy Error [Ha]", fontsize=14)
    # ax.tick_params(which='both', labelsize=14, direction='in')
    # ax.legend(loc='lower left', fontsize=10, ncol=1, frameon=False)
    # ax.set_title("One Dimensional Gaussian Kinetic Energy standard Error")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    # plt.savefig("one_D_Gaussian_Fionns_cutoffs_Variance_convergence.png", format='PNG', dpi=300)
    #plt.show()


if __name__ == "__main__":
    main()