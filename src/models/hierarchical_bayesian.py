import pymc as py 
import numpy as np 

class HierarchicalBayesianModel:
    def __init__(self):
        self.model = None
        self.trace = None
        self.assest_names = None

    def fit( self, asset_returns : dict):
        self .asset_names = list(asset_returns.keys())

        y = np.concatenate(list(asset_returns.values())
        group_idx = np.concatenate([np.full(len(asset_returns[a]), i)
            for i, a in enumerate(self.asset_names)
        ])

        with pm.Model() as model:
            global_mean = pm.Normal("global_mean", mu=0, sigma=1)
            tau = pm.HalfNormal("tau", sigma=1)

            asset_means = pm.Normal(
                "asset_means",
                mu=global_mean,
                sigma=tau,
                shape=len(self.asset_names)
            )
            sigma = pm.HalfNormal("sigma", sigma=1)

            pm.Normal(
                "returns",
                mu=asset_means[group_idx],
                sigma=sigma,
                observed=y
            )

            trace = pm.sample(
                1000,
                tune=1000,
                chains=2,
                target_accept=0.9,
                progressbar=False
            )

        self.model = model
        self.trace = trace
        return trace


    def get_posterior_means(self):
        means = self.trace.posterior["asset_means"].mean(
            dim=("chain", "draw")
        ).values

        return dict(zip(self.asset_names, means))

