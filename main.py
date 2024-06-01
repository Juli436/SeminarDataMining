import matplotlib.pyplot as plt
import numpy as np

import data_generation as dg
import frequentistEstimator as fe



if __name__ == '__main__':
    x, y, betas = dg.get_data(10000, 3, 0.1,0, 1,  10)
    #x = np.linspace(0,10, 100)
    print(betas)
    plt.scatter(x[1], y)
    plt.show()
    betas_est = fe.ml_estimator(x,y)
    print(betas_est)




