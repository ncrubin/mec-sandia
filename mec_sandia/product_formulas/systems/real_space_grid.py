"""
Calculate  The GHL norm and system
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
        if np.isclose(self.points_per_dim % 2, 0):
            raise ValueError("number of points in a direction must be odd")
        self.omega = np.linalg.det(box_length * np.eye(3))
        self.kfac = 2 * math.pi / self.L
        self.miller_vals = None

    def get_miller(self):
        if self.miller_vals is not None:
            return self.ijk_vals
        reciprocal_max_dim = (self.points_per_dim - 1) / 2
        # build all possible sets of miller indices
        ivals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
        jvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
        kvals = np.arange(-reciprocal_max_dim, reciprocal_max_dim+1)
        ijk_vals = cartesian_prod([ivals, jvals, kvals])
        self.miller_vals = ijk_vals
        return ijk_vals

    def get_eris(self):
        """
        Return Chemist ordered ERIs (1'1|2'2)
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

    def fourier_transform_matrix(self, miller_vals):
        dx = self.L / self.points_per_dim
        real_space_grid = miller_vals * dx
        print(real_space_grid)
        
    def get_h1(self,):
        """
        Generate diagonal kinetic energy term and then compute the inverse 
        fourier transform 
        """
        ijk_vals = self.get_miller()
        print(ijk_vals * self.L / self.points_per_dim)
        exit()
        gvals = 2 * np.pi * ijk_vals / self.L
        print(gvals)
        exit()
        # compute g^2
        g2vals = np.einsum('ix,ix->i', gvals, gvals)
        self.fourier_transform_matrix(ijk_vals)

if __name__ == "__main__":
    # rsg_inst = RealSpaceGrid(5, 15)
    # rsg_inst.get_h1()

    import matplotlib.pyplot as plt
    L = 5
    N = 9
    miller = np.arange(N)

    U_ft = np.zeros((N, N), dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            U_ft[k, n] = np.exp(-1j * 2 * np.pi * n * k / N)
    U_ft *= 1./np.sqrt(N)
    assert np.allclose(U_ft.conj().T @ U_ft, np.eye(N))

    xk0 = np.zeros(N)
    xk0[0] = 1.
    xk0_x = U_ft.conj().T @ xk0
    print(xk0_x)
    assert np.allclose(xk0_x, 1./np.sqrt(N))

    xk1 = np.zeros(N)
    xk1[1] = 1.
    xk1_x = U_ft.conj().T @ xk1
    print(xk1_x)
    # arange n-index , k=1
    true_vec = np.exp(1j * 2 * np.pi * np.arange(N) / N) / np.sqrt(N)
    assert np.allclose(xk1_x, true_vec)

    