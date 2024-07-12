import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg

import bayesian_estimator

import data_generation as dg
import frequentistEstimator as fe

from scipy.stats import multivariate_normal


def plotGaussian(mean, var, low_x, low_y, high_x, high_y, step, n, ml_mean, true_mean=None):
    a = np.mgrid[low_x:high_x:step, low_y:high_y:step]
    a[0] = np.flip(a[0])
    val = multivariate_normal(mean, var)
    temp = np.dstack(a)
    z = val.pdf(temp)

    plt.xlabel('beta_1')
    plt.ylabel('beta_0')
    if n != 1:
        plt.suptitle("Posterior for " + str(n) + " samples", fontsize=11)
    else:
        plt.suptitle("Posterior for " + str(n) + " sample", fontsize=11)
    if n >= 2:
        if true_mean is not None:
            plt.title(
                f"Mean: ({mean[1]:.5f}, {mean[0]:.5f}), True Parameters: ({true_mean[1]:.5f}, {true_mean[0]:.5f})\n"
                f"ML Estimate: ({ml_mean[1]:.5f}, {ml_mean[0]:.5f})", fontsize=9)
        else:
            plt.title(
                f"Mean: ({mean[1]:.5f}, {mean[0]:.5f}), ML Estimate: ({ml_mean[1]:.5f}, {ml_mean[0]:.5f})", fontsize=9)
    else:
        if true_mean is not None:
            plt.title(
                f"Mean: ({mean[1]:.5f}, {mean[0]:.5f}), True Parameters: ({true_mean[1]:.5f}, {true_mean[0]:.5f})",
                fontsize=9)
        else:
            plt.title(
                f"Mean: ({mean[1]:.5f}, {mean[0]:.5f})",
                fontsize=9)
    # plt.text(1.2, 0,  "abc")

    plot = plt.imshow(z, extent=(low_x, high_x, low_y, high_y))
    cb = plt.colorbar()
    # cb.set_ticks([0, np.amax(z)], labels=['low','high'])
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


# assumes all beta are independent gaussians with a certain variance
def plot_post(n, x, y, sig_noise, sig_betas, mean_betas, params, true_betas=None):
    if n == 0:
        cov = get_cov(params, sig_betas)
        plotGaussian(mean_betas, cov, -2, -2, 2, 2, 0.01, n, [0, 0], true_betas)
        return
    red_x = [x[0][0:n], x[1][0:n]]
    red_y = y[0:n]
    if n >= 2:
        betas_ml = fe.ml_estimator(red_x, red_y)
    else:
        betas_ml = np.zeros(params)

    cov = get_cov(params, sig_betas)

    mean, var = bayesian_estimator.bayesian_estimate_with_gaussian_prior(red_x, red_y, sig_noise, mean_betas, cov)

    plotGaussian(mean, var, -2, -2, 2, 2, 0.01, n, betas_ml, true_betas)
    return mean, var


def print_plots_example():
    x, y, betas = dg.get_data(10, 2, 1, 0, 1, 30, fixed_betas=[0.212, 1.345])
    print(x)
    print(y)
    print(betas)
    # x = np.linspace(0,10, 100)

    # plt.scatter(x[1], y)
    # plt.show()

    betas_est = fe.ml_estimator(x, y)
    print(betas_est)

    mean, var = bayesian_estimator.bayesian_estimate_with_gaussian_prior(x, y, 0.3, [0, 0],
                                                                         [[0.5 ** 2, 0], [0, 0.5 ** 2]])
    # plotGaussian(mean, var, -1, -1, 1,1, 0.01, 10)
    print(mean)
    print(var)
    for i in range(0, 11):
        plot_post(i, x, y, 0.3, [1, 1], [0, 0], 2, betas)


def simulateUpdate(prev_num_samples, prev_x, prev_y, new_x, new_y, sig_betas, mean_betas, sig_noise):
    num_samples = prev_num_samples + 1
    x = np.ones((2, num_samples))
    y = np.zeros(num_samples)
    for i in range(0, prev_num_samples):
        x[1][i] = prev_x[1][i]
        y[i] = prev_y[i]
    x[1][prev_num_samples] = new_x
    y[prev_num_samples] = new_y


    try:
        mean, var = plot_post(num_samples, x, y, sig_noise, sig_betas, mean_betas, 2)
    except numpy.linalg.LinAlgError:
        print("Singular Matrix\n")
        return x, y, None, None
    return x, y, mean, var


def run_simulation(mean_prior=None, sig_prior=None, sig_noise=0.3):
    if mean_prior is None:
        mean_prior = np.asarray([0,0])
    if sig_prior is None:
        sig_prior = np.asarray([1,1])
    x_ar = [[0],[0]]
    y_ar = [0]
    plot_post(0, None, None, None, sig_prior, mean_prior, 2)
    mean = 0
    var = 0

    x_prep = [25.2, 20.1, 21.3, 19]
    y_prep = [32.5, 26.4, 29.1,24.2]
    ind_prep = 0

    for i in range(0, 1000):
        x = input("Please type value for x or type \"predict\": ")
        if x == "predict":
            x_pred = input("Please input x value to get prediction for: ")
            mean_pred, var_pred = bayesian_estimator.predict([1, float(x_pred)], mean, var)
            y_plot = np.linspace(0, 35, 1000)
            p_plot = bayesian_estimator.gaussian_var(mean_pred, y_plot, var_pred)
            plt.plot(y_plot, p_plot)
            plt.ylabel("p(y)")
            plt.xlabel("y")
            plt.show()
            continue
        if x == "exit":
            break

        if x == "a":
            x = x_prep[ind_prep]
            y = y_prep[ind_prep]
            ind_prep = (ind_prep+1)%(len(x_prep))

        else:
            y = input("Please type value for y: ")
            if y == "continue":
                continue
        x = float(x)
        y = float(y)

        x_ar, y_ar, mean, var = simulateUpdate(i, x_ar, y_ar, x, y,  sig_prior, mean_prior,sig_noise)






def get_plots_grid():
    fig, axs = plt.subplots(2, 2)
    x, y, betas = dg.get_data(10, 2, 0.3, 0, 0.5, 1)
    print(x)
    print(y)
    print(betas)
    for i in range(0, 11):
        plot_post(i, x, y, 0.3, [0.5, 0.5], [0.5, 0.5], 2, betas)


#print_plots_example()
run_simulation()
