import pytest

from nsga2.model import NSGA2

def test_model_work():    
    nsga = NSGA2(18, 100, 10)
    nsga.simulate(False)
