import numpy as np


# draws datapoints from sinus curve from 0 to max_val, using a uniform distribution to select the
# x - values and with a normally distributed noise with variance sig. Returns n samples
def get_data_from_sin(n, sig, max_val):
    x = np.random.uniform(high=max_val, size=n)
    y = np.sin(x)
    noise = np.random.normal(size=n, scale=sig)
    return x, y + noise
