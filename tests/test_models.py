import numpy as np
from src.models.bayesian_model import BayesianReturnModel

def test_bayesian_model():
    model = BayesianReturnModel()
    ret = np.random.normal(0.01, 0.02, 100)

    mean, var = model.fit(ret)

    assert mean != 0
    assert var > 0
