import itertools

import numpy as np
from pyscf import fci, gto, scf

from mec_sandia.ueg import UEG, UEGTMP


def test_build():
    ueg_inst = UEG.build(14, 1.0, 10)
    assert len(ueg_inst.eigenvalues) == 389
    assert np.isclose(sum(ueg_inst.eigenvalues[:7]), 6 * ueg_inst.eigenvalues[1])


def test_eris():
    # ueg = UEGTMP((7, 7), 1.0, 1)
    # assert ueg.nbasis == 19
    # eris_4 = ueg.eri_4()
    # assert eris_4.shape == (19,) * 4
    # for i, j, k, l in itertools.product(range(ueg.nbasis), repeat=4):
    #     # momentum conservation
    #     q1 = ueg.basis[k] - ueg.basis[i]
    #     q2 = ueg.basis[j] - ueg.basis[l]
    #     if np.allclose(q1, q2) and np.dot(q1, q1) > 0:
    #         assert np.isclose(
    #             eris_4[i, k, j, l],
    #             1.0 / ueg.vol * ueg.vq(ueg.kfac * (ueg.basis[k] - ueg.basis[i])),
    #         )
    #     else:
    #         assert eris_4[i, k, j, l] == 0

    # h1e = np.diag(ueg.sp_eigv)
    # assert h1e.shape == (ueg.nbasis,) * 2
    # # planewaves are MOs for UEG so DM = 2 * I in the oo block
    # e1b = 2 * np.sum(h1e[:7, :7])
    # e2b = 2 * np.einsum("ppqq->", eris_4[:7, :7, :7, :7])
    # e2b -= np.einsum("pqqp->", eris_4[:7, :7, :7, :7])
    # etot = e1b + e2b
    # assert np.isclose(etot, 13.60355734)  # HF energy from HANDE
    # Other reference point
    ueg_nel4 = UEGTMP(nelec=(2, 2), rs=2.0, ecut=1)
    h1e = np.diag(ueg_nel4.sp_eigv)
    eris = ueg_nel4.eri_8()
    mol = gto.M()
    mol.nelectron = ueg_nel4.nelec
    nbasis = h1e.shape[0]
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: np.eye(nbasis)
    mf._eri = eris
    mf.kernel()
    cisolver = fci.FCI(mf)
    assert np.isclose(cisolver.kernel()[0], 1.285524498812)