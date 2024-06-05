import numpy as np

class BayesianEstimator:
    def __init__(self):
        self.name = "abc"


    def gaussian(self, mean, x, sig):
        norm = 1/(sig*np.sqrt(2*np.pi))
        exp = np.exp(-0.5*((x-mean)/sig)**2)
        return norm * exp


    #assuming gaussian prior over betas
    def prior(self, beta, mean, sig):
        return self.gaussian(mean, beta, sig)

    def likelihood(self, x, y, beta, sig_er):
        y_hat = np.dot(beta, x)
        er = y -y_hat
        p_er = self.gaussian(0, er, sig_er)
        return p_er

    def var_posterior(self, cov_betas, sig_noise, x):
        inv_cov_betas = np.linalg.inv(cov_betas)

        temp = (1/float(sig_noise))**2 * np.dot(x, np.transpose(x)) #other way around bc weird np matrix stuff
        res = np.linalg.inv(inv_cov_betas + temp)

        return res

    def mean_posterior(self, var_posterior, cov_betas, mean_betas, sig_noise, x, y):
        temp = np.dot(np.linalg.inv(cov_betas), mean_betas)
        temp2 =  np.dot(np.dot((1/float(sig_noise))**2, x), y)
        return np.dot(var_posterior, temp + temp2)


    def bayesian_estimate_with_gaussian_prior(self, x, y, sig_noise, mean_betas, cov_betas):
        var_posterior = self.var_posterior( cov_betas, sig_noise, x)
        mean_posterior =  self.mean_posterior( var_posterior, cov_betas, mean_betas, sig_noise, x, y)
        return mean_posterior, var_posterior

    def bayesian_estimate_with_gaussian_prior_default(self, x, y, n):
        cov_betas = np.ndarray((n, n))
        for i in range(0,n):
            for j in range(0,n):
                cov_betas[i][j] = 1 if i == j else 0
        mean_betas = np.zeros(n)
        sig_noise = 0.1
        return self.bayesian_estimate_with_gaussian_prior(x, y, sig_noise, mean_betas, cov_betas)






       # betas = np.


