"""
Calculate  The GHL norm and system

See note for set up

set box-length and grid points. These define the real space grid

ivals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
jvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
kvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)

r_{p} = p * Omega^{1/3} / N^{1/3}

N^{1/3} = points_per_dim
Omega^{1/3} = L


"""
import itertools
import math
import numpy as np

from pyscf.lib.numpy_helper import cartesian_prod

class RealSpaceGrid:

    def __init__(self, box_length: float, points_per_dim: int):
        """
        :param box_length: L length of box in atomic units
        :param points_per_dim: N^1/3
        :param num_elec: number of electrons
        """
        self.L = box_length
        self.points_per_dim = points_per_dim

        # if np.isclose(self.points_per_dim % 2, 0):
        #     raise ValueError("number of points in a direction must be odd")

        self.omega = np.linalg.det(box_length * np.eye(3))
        self.kfac = 2 * math.pi / self.L
        self.miller_vals = None
        self.norb = points_per_dim**3

    def get_miller(self):
        if self.miller_vals is not None:
            return self.miller_vals
        if np.isclose(self.points_per_dim % 2, 1):
            reciprocal_max_dim = (self.points_per_dim - 1) / 2
            # build all possible sets of miller indices
            ivals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
            jvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
            kvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
            ijk_vals = cartesian_prod([ivals, jvals, kvals])
            self.miller_vals = ijk_vals
        else:
            reciprocal_max_dim = self.points_per_dim / 2
            ivals = np.arange(-reciprocal_max_dim, reciprocal_max_dim)
            jvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim)
            kvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim)
            ijk_vals = cartesian_prod([ivals, jvals, kvals])
            self.miller_vals = ijk_vals
        return ijk_vals
    
    def get_real_space_grid(self):
        if self.miller_vals is None:
            self.miller_vals = self.get_miller()
        return self.L * self.miller_vals / self.points_per_dim
    
    def get_momentum_space_grid(self):
        if self.miller_vals is None:
            self.miller_vals = self.get_miller()
        return 2 * np.pi * self.miller_vals / self.L

    def get_eris(self):
        """
        Return Chemist ordered ERIs (1'1|2'2)
        The factor of 1/2 is ALREADY included in this
        """
        ijk_vals = self.get_miller()
        norb = len(ijk_vals)
        eris = np.zeros(4 * (norb,))
        distance_mat = np.zeros((norb, norb))
        for l, m in itertools.product(range(len(ijk_vals)), repeat=2):
            distance_mat[l, m] = np.linalg.norm(ijk_vals[l] - ijk_vals[m])
        distance_mat = np.reciprocal(distance_mat, 
                                     out=np.zeros_like(distance_mat), 
                                     where=~np.isclose(distance_mat, 
                                                       np.zeros_like(distance_mat)
                                                       )
                                    )
        for l, m in itertools.product(range(norb), repeat=2):
            eris[l, l, m, m] = distance_mat[l, m]

        eris *= self.points_per_dim / (2 * self.L)
        return eris

    def fourier_transform_matrix(self):
        """position space to momentum space
        |k> = 1/sqrt(N) exp(-i k.r) |r> 
        r - > k is (1/sqrt(N)) exp[-i k_{nu} r_{p}] = (1 / sqrt(N)) exp[-i 2pi nu (nu . p) / N^{1/3}]
        """
        real_space_grid = self.get_real_space_grid()
        momentum_space_grid = self.get_momentum_space_grid()
        w_exponent = -1j * np.einsum('ix,jx->ij', momentum_space_grid, real_space_grid)
        return np.exp(w_exponent) / np.sqrt(self.norb)
        
    def get_kspace_h1(self,):
        """
        Generate diagonal kinetic energy term and then compute the inverse 
        fourier transform 
        """
        knu = self.get_momentum_space_grid()
        return 0.5 * np.einsum('ix,ix->i', knu, knu)

    def get_rspace_h1(self,):
        # recall |r> = U_ft^{dag}|k>
        diag_k_space_h1 = np.diag(self.get_kspace_h1())
        u_ft = self.fourier_transform_matrix()
        return u_ft.conj().T @ diag_k_space_h1 @ u_ft

if __name__ == "__main__":
    rsg = RealSpaceGrid(5, 4)
    u = rsg.fourier_transform_matrix()
    assert np.allclose(u.conj().T @ u, np.eye(4**3))
