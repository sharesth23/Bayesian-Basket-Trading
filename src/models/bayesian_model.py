import numpy as np

class BayesianReturnModel:
   

    def __init__(self, prior_mean=0.0, prior_var=1.0):
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def fit(self, returns):
     
        n = len(returns)
        sample_mean = np.mean(returns)
        sample_var = np.var(returns)

        
        post_var = 1 / (n / sample_var + 1 / self.prior_var)

      
        post_mean = post_var * (self.prior_mean / self.prior_var + n * sample_mean / sample_var)

        self.posterior_mean = post_mean
        self.posterior_var = post_var

        return post_mean, post_var

    def predict(self):
        return self.posterior_mean
