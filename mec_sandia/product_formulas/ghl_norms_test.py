import numpy as np
from mec_sandia.product_formulas.ghl_norms import compute_tau_norm, compute_nu_eta_norm
from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid

def test_tau_norm():
    ll_vals = 2**(np.arange(1, 5)) + 1
    for ll in range(3, 22, 2): # [25]: #ll_vals:
        rsg_inst = RealSpaceGrid(5, ll)
        tau_norm_val = np.max(rsg_inst.get_kspace_h1())
        test_tau_norm = compute_tau_norm(rsg_inst)
        rspace_h1 = rsg_inst.get_rspace_h1()
        tau_ksummed = np.sum(np.abs(rspace_h1), axis=1)
        print(ll, test_tau_norm / tau_norm_val)
    # assert np.isclose(tau_norm_val, test_tau_norm)


def test_nu_norm_0():
    rsg = RealSpaceGrid(5, 3)
    eris = rsg.get_eris()
    eta_test = np.einsum('llmm->lm', eris)
    eta = np.zeros((rsg.norb, rsg.norb))
    for l in range(rsg.norb):
        for m in range(rsg.norb):
            eta[l, m] = eris[l, l, m, m]
    assert np.allclose(eta_test, eta)

    row_sorted_eta = -np.sort(-np.abs(eta), axis=1)
    for ii in range(row_sorted_eta.shape[0]):
        assert np.allclose(np.sort(row_sorted_eta[ii])[::-1], row_sorted_eta[ii])

if __name__ == "__main__":
    # test_tau_norm()
    test_nu_norm_0()

