import itertools
import numpy as np
from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid

def test_eris():
    rsg_inst = RealSpaceGrid(5, 3)
    eris = rsg_inst.get_eris()
    real_space_grid = rsg_inst.get_real_space_grid()
    for l, m in itertools.product(range(rsg_inst.norb), repeat=2):
        if l == m:
            assert np.isclose(eris[l, l, m, m], 0)
        else:
            delta = np.linalg.norm(real_space_grid[l] - real_space_grid[m])
            assert np.isclose(0.5 /delta, eris[l, l, m, m])

def test_fourier_transform():
    rsg_inst = RealSpaceGrid(5, 3)
    real_space_grid = rsg_inst.get_real_space_grid()
    momentum_space_grid = rsg_inst.get_momentum_space_grid()
    miller = rsg_inst.get_miller()
    w_exponent = -1j * np.einsum('ix,jx->ij', momentum_space_grid, real_space_grid)
    for nu, p in itertools.product(range(rsg_inst.norb), repeat=2):
        knu_dot_rp = momentum_space_grid[nu].dot(real_space_grid[p])
        test_knu_dot_rp = miller[nu].dot(miller[p]) * 2 * np.pi / rsg_inst.points_per_dim
        assert np.isclose(knu_dot_rp, test_knu_dot_rp)

    fourier_transform_matrx = np.exp(w_exponent) / np.sqrt(rsg_inst.norb)
    assert np.allclose(fourier_transform_matrx.conj().T @ fourier_transform_matrx, np.eye(rsg_inst.norb))

def test_kspace_h1():
    """
    return 0.5 * k_nu^2 
    """
    rsg_inst = RealSpaceGrid(5, 3)
    test_kspace_h1 = rsg_inst.get_kspace_h1()
    momentum_space_grid = rsg_inst.get_momentum_space_grid()
    miller = rsg_inst.get_miller()
    for ii in range(rsg_inst.norb):
        assert np.isclose(test_kspace_h1[ii], 2 * np.pi**2 * miller[ii].dot(miller[ii]) / rsg_inst.L**2)

def test_rspace_h1():
    """
    return U_ft^ (k^2/2) |k><k| U_ft
    """
    rsg_inst = RealSpaceGrid(5, 3)
    test_rspace_h1 = rsg_inst.get_rspace_h1()
    assert np.allclose(test_rspace_h1.imag, 0)
    assert np.allclose(test_rspace_h1.conj().T, test_rspace_h1)
    wt, _ = np.linalg.eigh(test_rspace_h1)
    w = sorted(rsg_inst.get_kspace_h1())
    assert np.allclose(wt, w)

if __name__ == "__main__":
    # test_fourier_transform() 
    test_rspace_h1()
