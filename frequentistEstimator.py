import numpy as np

def ml_estimator(x, y):
    xt_x_inv = np.linalg.inv(np.dot( x, np.transpose(x)))  # has to be other way around bc of weird np array to matrix shapes
    temp = np.dot(xt_x_inv, x)
    betas = np.dot(temp, y)
    return betas
