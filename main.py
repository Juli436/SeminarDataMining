import matplotlib.pyplot as plt
import numpy as np
import bayesian_estimator

import data_generation as dg
import frequentistEstimator as fe



if __name__ == '__main__':
    x, y, betas = dg.get_data(10, 2, 0.1,0, 1,  10)
    #x = np.linspace(0,10, 100)
    print(betas)
    #plt.scatter(x[1], y)
    #plt.show()
    betas_est = fe.ml_estimator(x,y)
    print(betas_est)
    mean, var = bayesian_estimator.BayesianEstimator().bayesian_estimate_with_gaussian_prior_default(x, y, 2)
    print(mean)
    print(var)





