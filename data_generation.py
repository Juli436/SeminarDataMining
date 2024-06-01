import numpy as np


# draws n datapoints using a linear combination of the parameters, which are generated from a normal
# distribution. They are sampled from 0 to upper_boundary
def get_data(n, params,  sig, mean_beta, sig_beta, upper_boundary):
    betas = np.random.normal(loc=mean_beta, scale=sig_beta, size=params)
    x = np.random.uniform(high=upper_boundary, size=(params, n))
    print(betas)
    print(x)
    y = np.dot(betas, x)
    print(y)
    noise = np.random.normal(size=n, scale=sig)
    return x, y + noise, betas
