"""
Compute the norms defined by GHL for real-space grid Hamiltonians
"""
import numpy as np
from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid


def compute_tau_norm(rsg: RealSpaceGrid) -> float:
    rspace_h1 = rsg.get_rspace_h1().real
    tau_ksummed = np.sum(np.abs(rspace_h1), axis=1)
    return np.max(tau_ksummed)

def compute_nu_eta_norm(rsg: RealSpaceGrid, eta: int) -> float:
    eris = rsg.get_eris()
    nu_mat = np.einsum('llmm->lm', eris)
    return np.max(np.sum(-np.sort(-np.abs(nu_mat), axis=1)[:, :eta], axis=1))

if __name__ == "__main__":
    # first test scaling with N
    for ppd in [2, 3, 4, 5, 6]:
        rsg = RealSpaceGrid(5, ppd)
        test_tau_norm = compute_tau_norm(rsg)
        rspace_h1 = rsg.get_rspace_h1()
        tau_norm_val = np.max(rsg.get_kspace_h1())
        print(ppd, ppd**3, test_tau_norm / tau_norm_val)