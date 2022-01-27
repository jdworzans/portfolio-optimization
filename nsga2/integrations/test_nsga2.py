from functools import partial

import numpy as np
from nsga2.model import NSGA2
from nsga2.objectives import (get_objective, portfolios_neg_returns,
                              portfolios_semivariances, portfolios_vars)


def test_model_work():
    R = np.random.random(18)
    SIGMA = np.random.random((18, 18))
    SIGMA = SIGMA @ SIGMA.T
    returns = np.random.randn(1000, 18)
    objective = get_objective(
        partial(portfolios_neg_returns, R=R),
        partial(portfolios_vars, SIGMA=SIGMA),
        partial(portfolios_semivariances, returns=returns),
    )
    nsga = NSGA2(18, 100, 10)
    nsga.simulate(objective, False)
