from functools import partial

import numpy as np
import pytest
from nsga2.model import NSGA2
from nsga2.objectives import (get_objective, portfolios_neg_returns,
                              portfolios_vars)


def test_model_work():
    R = np.random.random(18)
    SIGMA = np.random.random((18, 18))
    SIGMA = SIGMA @ SIGMA.T
    objective = get_objective(
        partial(portfolios_neg_returns, R=R),
        partial(portfolios_vars, SIGMA=SIGMA),
    )
    nsga = NSGA2(18, 100, 10)
    nsga.simulate(objective, False)
