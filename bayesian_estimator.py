import numpy as np


def gaussian_sig( mean, x, sig):
    norm = 1/(sig*np.sqrt(2*np.pi))
    exp = np.exp(-0.5*((x-mean)/sig)**2)
    return norm * exp

def gaussian_var( mean, x, var):
    norm = 1/(np.sqrt(2*np.pi*var))
    exp = np.exp(-0.5*((x-mean)/np.sqrt(var))**2)
    return norm * exp


#assuming gaussian prior over betas
def prior(beta, mean, sig):
    return gaussian_sig(mean, beta, sig)

def likelihood( x, y, beta, sig_er):
    y_hat = np.dot(beta, x)
    er = y -y_hat
    p_er = gaussian_sig(0, er, sig_er)
    return p_er

def var_posterior( cov_betas, sig_noise, x):
    inv_cov_betas = np.linalg.inv(cov_betas)

    temp = (1/float(sig_noise))**2 * np.dot(x, np.transpose(x)) #other way around bc weird np matrix stuff
    res = np.linalg.inv(inv_cov_betas + temp)

    return res

def mean_posterior( var_post, cov_betas, mean_betas, sig_noise, x, y):
    temp = np.dot(np.linalg.inv(cov_betas), mean_betas)
    temp2 = np.dot(np.dot((1/float(sig_noise))**2, x), y)
    return np.dot(var_post, temp + temp2)


def bayesian_estimate_with_gaussian_prior(x, y, sig_noise, mean_betas, cov_betas):
    var_post = var_posterior(cov_betas, sig_noise, x)
    mean_post = mean_posterior(var_post, cov_betas, mean_betas, sig_noise, x, y)
    return mean_post, var_post

def bayesian_estimate_with_gaussian_prior_default( x, y, n):
    cov_betas = np.ndarray((n, n))
    for i in range(0,n):
        for j in range(0,n):
            cov_betas[i][j] = 1 if i == j else 0
    mean_betas = np.zeros(n)
    sig_noise = 0.1
    return bayesian_estimate_with_gaussian_prior(x, y, sig_noise, mean_betas, cov_betas)

def predict(x, mean_estimate, variance_estimate, sig_noise = 0):
    mean = np.dot(x, mean_estimate)
    var = np.dot(np.dot(x, variance_estimate)+(sig_noise**2), np.transpose(x))
    return mean, var











