from matplotlib.pyplot import box
import numpy as np
from mec_sandia.gaussians import (
    discrete_gaussian_wavepacket,
    estimate_cutoff,
    estimate_error_kinetic_energy,
    kinetic_energy,
    estimate_kinetic_energy,
)
import pytest


def test_discrete_gaussian_wavepacket():
    sigma = 4
    box_length = 15
    ecut = 2000
    gaussian, kmesh, norm = discrete_gaussian_wavepacket(
        ecut, box_length, sigma, ndim=1
    )
    norm_exact = (sigma * box_length) / (np.sqrt(2 * np.pi))
    assert gaussian.shape == (151,)
    assert kmesh.shape == (151, 1)
    assert np.allclose(norm_exact, norm)
    gaussian, kmesh, norm = discrete_gaussian_wavepacket(
        ecut, box_length, sigma, ndim=2
    )
    assert gaussian.shape == (151**2,)
    assert kmesh.shape == (151**2, 2)
    norm_exact = (sigma**2.0 * box_length**2.0) / (2 * np.pi)
    assert np.allclose(norm_exact, norm)
    gaussian, kmesh, norm = discrete_gaussian_wavepacket(
        ecut, box_length, sigma, ndim=3
    )
    assert gaussian.shape == (151**3,)
    assert kmesh.shape == (151**3, 3)
    norm_exact = (sigma**3.0 * box_length**3.0) / ((2 * np.pi) ** (3 / 2.0))
    assert np.allclose(norm_exact, norm)
    sigma = 1
    box_length = 15
    ecut = 200
    gaussian, kmesh, norm = discrete_gaussian_wavepacket(
        ecut, box_length, sigma, ndim=4
    )
    assert gaussian.shape == (48**4,)
    assert kmesh.shape == (48**4, 4)
    norm_exact = (sigma**4.0 * box_length**4.0) / ((2 * np.pi) ** (2.0))
    assert np.allclose(norm_exact, norm)


def test_kinetic_energy():
    sigma = 4
    box_length = 15
    ecut = 2000
    ke = kinetic_energy(ecut, box_length, sigma, ndim=1)
    assert np.allclose(ke, sigma**2.0 / 2)
    sigma = 4
    box_length = 15
    ecut = 2000
    ke = kinetic_energy(ecut, box_length, sigma, ndim=2)
    assert np.allclose(ke, 2 * sigma**2.0 / 2)
    sigma = 4
    box_length = 15
    ecut = 2000
    ke = kinetic_energy(ecut, box_length, sigma, ndim=3)
    assert np.allclose(ke, 3 * sigma**2.0 / 2)
    sigma = 4
    box_length = 15
    ecut = 2000
    kproj = 2 * np.pi / box_length * np.random.random(3)
    ke = kinetic_energy(ecut, box_length, sigma, ndim=3, kproj=kproj)
    assert np.allclose(ke, (3 * sigma**2.0 + np.dot(kproj, kproj)) / 2)


_prec = 1e-8
@pytest.mark.parametrize(
    "input,expected", [((_prec, 10), _prec), ((_prec, 4), _prec), ((_prec, 100), _prec)]
)
def test_estimate_cutoff(input, expected):
    prec, sigma = input
    kopt = estimate_cutoff(prec, sigma)
    assert np.isclose(estimate_error_kinetic_energy(kopt, sigma), expected)
    box_length = 15
    kopt = estimate_cutoff(prec, sigma)
    ecut = 0.5 * kopt**2.0
    # 4 ecut for the sum apparently
    ke_sum = kinetic_energy(4*ecut, box_length, sigma, ndim=1)
    assert abs(ke_sum - 0.5*sigma**2.0) < expected 
    assert estimate_kinetic_energy(kopt, sigma) - 0.5*sigma**2.0 < expected