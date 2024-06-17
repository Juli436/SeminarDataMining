import matplotlib.pyplot as plt
import numpy as np
import bayesian_estimator

import data_generation as dg
import frequentistEstimator as fe
import parse_data as data
from scipy.stats import multivariate_normal


def plotGaussian(mean, var, low_x, low_y, high_x, high_y, step, n, true_mean, ml_mean):
    a= np.mgrid[low_x:high_x:step, low_y:high_y:step]
    a[0] = np.flip(a[0])
    val = multivariate_normal(mean, var)
    temp = np.dstack(a)
    z = val.pdf(temp)

    plt.xlabel('beta_1')
    plt.ylabel('beta_0')
    if n != 1:
        plt.suptitle("Posterior for "+str(n)+" samples",  fontsize=11)
    else:
        plt.suptitle("Posterior for " + str(n) + " sample", fontsize=11)
    if n >=2:
        plt.title(f"Mean: ({mean[1]:.5f}, {mean[0]:.5f}), True Parameters: ({true_mean[1]:.5f}, {true_mean[0]:.5f})\n"
              f"ML Estimate: ({ml_mean[1]:.5f}, {ml_mean[0]:.5f})", fontsize=9)
    else:
        plt.title(f"Mean: ({mean[1]:.5f}, {mean[0]:.5f}), True Parameters: ({true_mean[1]:.5f}, {true_mean[0]:.5f})", fontsize=9)
    #plt.text(1.2, 0,  "abc")


    plt.imshow(z, extent=(low_x, high_x, low_y, high_y))
    cb = plt.colorbar()
    #cb.set_ticks([0, np.amax(z)], labels=['low','high'])
    trial = 2
    plt.savefig(str(trial) + "_trial_" + str(n) + '_samples.png')

    plt.show()


def get_cov(n, sig):
    cov = np.ndarray((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                cov[i][j] = sig[i] ** 2
            else:
                cov[i][j] = 0
    return cov


#assumes all beta are independent gaussians with a certain variance
def plot_post(n, x, y, sig_noise, sig_betas, mean_betas, params, true_betas):
    if n == 0:
        cov = get_cov(params, sig_betas)
        plotGaussian(mean_betas, cov, -1, -1, 1, 1, 0.01, n, true_betas, [0,0])
        return
    red_x = [x[0][0:n],x[1][0:n]]
    red_y = y[0:n]
    if n>=2:
        betas_ml = fe.ml_estimator(red_x, red_y)
    else:
        betas_ml = np.zeros(params)

    cov = get_cov(params, sig_betas)
    mean, var = bayesian_estimator.bayesian_estimate_with_gaussian_prior(red_x,red_y, sig_noise,mean_betas, cov)

    plotGaussian(mean, var, -1, -1, 1,1 , 0.01, n, true_betas, betas_ml)


def print_plots_example():
    x, y, betas = dg.get_data(10, 2, 0.3, 0, 0.5, 1)
    print(x)
    print(y)
    print(betas)
    # x = np.linspace(0,10, 100)

    # plt.scatter(x[1], y)
    # plt.show()

    betas_est = fe.ml_estimator(x, y)
    print(betas_est)

    mean, var = bayesian_estimator.bayesian_estimate_with_gaussian_prior(x,y, 0.3, [0,0], [[0.5**2, 0],[0, 0.5**2]])
    # plotGaussian(mean, var, -1, -1, 1,1, 0.01, 10)
    print(mean)
    print(var)
    for i in range(0, 11):
        plot_post(i, x, y, 0.3, [0.5, 0.5], [0.5, 0.5], 2, betas)


print_plots_example()