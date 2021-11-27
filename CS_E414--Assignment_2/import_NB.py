import numpy as np

# Training data - 3 randomly-generated Gaussian-distributed clouds of points in 2d space
def import_NB():

    np.random.seed(0)
    N = 1000
    y = np.random.randint(low=0, high=2+1, size = N)
    classes = list(set(y))
    means = np.array([[-1, 1, -1], [-1, 1, 1],])
    covariances = np.random.random_sample((2, 3)) + 1
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])

    return X, y