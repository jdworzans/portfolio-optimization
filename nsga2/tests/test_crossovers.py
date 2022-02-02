import numpy as np

from nsga2.crossovers import SBX


def test_sbx_icdf():
    assert SBX.sbx_icdf(0, 4) == 0
    assert SBX.sbx_icdf(0.5, 4) == 1

def test_sbx():
    sbx = SBX()
    sbx.cross(np.zeros(3), np.ones(3))
