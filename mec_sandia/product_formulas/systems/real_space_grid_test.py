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

def test_even_number_points():
    rsg = RealSpaceGrid(5, 4)
    u = rsg.fourier_transform_matrix()
    assert np.allclose(u.conj().T @ u, np.eye(4**3))

def test_diagonal_coulomb_matrix():
    rsg = RealSpaceGrid(5, 3)
    eris = 2 * rsg.get_eris()
    of_eris = eris.transpose((0, 2, 3, 1))
    dc_mat = 2 * rsg.get_diagonal_coulomb_matrix()
    for l, m in itertools.product(range(rsg.norb), repeat=2):
        assert np.isclose(eris[l, l, m, m], dc_mat[l, m])

    from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb
    tb_dc = DiagonalCoulomb(np.einsum('ijlk', -0.5 * of_eris))
    tb_dc_v2 = DiagonalCoulomb(0.5 * dc_mat)
    assert np.allclose(tb_dc._tensor[1], 0)
    assert np.allclose(tb_dc._tensor[2], 0.5 * dc_mat)
    assert np.allclose(tb_dc._tensor[2], tb_dc_v2._tensor[2])

def test_real_space_h1():
    rsg = RealSpaceGrid(5, 11)
    diag_k_space_h1 = np.diag(rsg.get_kspace_h1())
    import time
    start_time = time.time()
    u_ft = rsg.fourier_transform_matrix()
    test_rh1 = np.einsum('ix,x,xj->ij', u_ft.conj().T, rsg.get_kspace_h1(), u_ft, optimize=True)
    end_time = time.time()
    print("einsum_time ", end_time - start_time)

    assert np.allclose(test_rh1, u_ft.conj().T @ diag_k_space_h1 @ u_ft)

    start_time = time.time()
    rh1 = rsg.get_rspace_h1() 
    end_time = time.time()
    print("matmul_time ", end_time - start_time)
    assert np.allclose(test_rh1, rh1)
    print("PASSED TEST")
