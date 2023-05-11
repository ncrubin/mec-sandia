import numpy as np

from mec_sandia.stopping_power import compute_stopping_power


def test_stopping_power():
    v_proj = 10.0  # atomic units just taken from carbon
    mass_proj = 1836
    time_vals = np.linspace(0, 40, 20)
    np.random.seed(7)
    kproj_vals = np.array(
        [np.array([mass_proj * v_proj - 1e-3 * t, 0, 0]) for t in time_vals]
    )
    box_length = 15
    ecut = 2000
    sigma_k = 10.0
    stopping_deriv = 0.17957
    stopping_data = compute_stopping_power(
        ecut,
        box_length,
        sigma_k,
        time_vals,
        kproj_vals,
        stopping_deriv,
        mass_proj,
        num_samples=100,
    )
    assert np.isclose(stopping_data.stopping, -0.006962978108035145)
    assert len(stopping_data.kinetic) == 20
