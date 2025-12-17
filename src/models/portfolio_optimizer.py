import numpy as np 

class Portfolio:

    def  allocte(self,expected_returns):
        er = np.array(expected_returns)
        er = np.maximum(er, 0)
        if er.sum() ==0 :
            return np.ones_like(er)/len(er)


        weights = er/ er.sum()
        return weights
        
