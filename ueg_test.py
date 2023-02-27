import numpy as np

from ueg import UEG

def test_build():
    ueg_inst = UEG.build(14, 1.0, 10)
    assert len(ueg_inst.eigenvalues) == 389
    assert np.isclose(sum(ueg_inst.eigenvalues[:7]), 6*ueg_inst.eigenvalues[1])
