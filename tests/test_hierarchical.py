import numpy as np
from src.models.hierarchical_bayesian import HierarchicalBayesianModel

def test_hierarchical_model_runs():
    data = {
        "A": np.random.normal(0.01, 0.02, 100),
        "B": np.random.normal(0.02, 0.03, 120),
    }

    model = HierarchicalBayesianModel()
    trace = model.fit(data)

    means = model.get_posterior_means()
    assert len(means) == 2
