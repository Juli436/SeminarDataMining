import numpy as np


# draws n datapoints using a linear combination of the parameters, which are generated from a normal
# distribution. They are sampled from 0 to upper_boundary
def get_data(n, params,  sig, mean_beta, sig_beta, upper_boundary, fixed_betas=None):
    if fixed_betas is None:
        betas = np.random.normal(loc=mean_beta, scale=sig_beta, size=params)
    else:
        betas = fixed_betas
    x = np.random.uniform(high=upper_boundary, size=(params, n))
    x[0] = np.ones(n)  # to model constant y-intercept
    y = np.dot(betas, x)
    noise = np.random.normal(size=n, scale=sig)
    return x, y + noise, betas
