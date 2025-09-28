import numpy as np

def rmse(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat)**2)))

def nrmse(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    rng = (np.max(y) - np.min(y)) + 1e-12
    return float(np.sqrt(np.mean((y - yhat)**2))/rng)
