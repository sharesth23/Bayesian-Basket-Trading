from src.models.bayesian_model import BayesianReturnModel
from src.models.hierarchical_bayesian import HierarchicalBayesianModel
from src.models.portfolio_optimizer import PortfolioOptimizer

class BasketStrategy:
    def __init__(self, model_type="hierarchical"):
        self.model_type = model_type
        self.optimizer = PortfolioOptimizer()

        if model_type == "hierarchical":
            self.model = HierarchicalBayesianModel()
        else:
            self.model = BayesianReturnModel()

    def run(self, asset_returns):
        if self.model_type == "hierarchical":
            self.model.fit(asset_returns)
            posterior_means = self.model.get_posterior_means()
        else:
            posterior_means = {}
            for asset, ret in asset_returns.items():
                mean, _ = self.model.fit(ret)
                posterior_means[asset] = mean

        weights = self.optimizer.allocate(
            list(posterior_means.values())
        )

        return {
            "posterior_means": posterior_means,
            "weights": dict(zip(posterior_means.keys(), weights))
        }
