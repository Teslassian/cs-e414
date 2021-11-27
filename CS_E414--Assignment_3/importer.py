import numpy as np

def kmeans_data():
    # Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
    np.random.seed(0)
    # Number of points
    N = 1000
    # Labels for each cluster
    y = np.random.randint(low=0, high=2, size = N)
    # Mean of each cluster
    means = np.array([[-2, 2], [-2, 2],])
    # Covariance (in X and Y direction) of each cluster
    covariances = np.random.random_sample((2, 2)) + 1
    # Dimensions of each point
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    
    return X, y

def gmm_data():
    # Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
    np.random.seed(0)
    # Number of points
    N = 1000
    # Labels for each cluster
    y = np.random.randint(low=0, high=2, size = N)
    # Mean of each cluster
    means = np.array([[-2, 2], [-2, 2],])
    # Covariance (in X and Y direction) of each cluster
    covariances = np.random.random_sample((2, 2)) + 1
    # Dimensions of each point
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    
    return X, y
