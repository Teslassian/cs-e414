import numpy as np

def sigmoid(x):
    '''Computes the sigmoid function'''
    s = 1/(1+np.exp(-x))
    
    return s

def hessian(N, d, X, w, lambda_):
    '''Computes the Hessian matrix'''
    H = 0
    I = np.eye(d)
    for n in range(N):
        H += sigmoid(w.T @ X[:,n,None])  *  (1-sigmoid(w.T @ X[:,n,None]))  *  (X[:,n,None] @ X[:,n,None].T)  +  1/lambda_*I
 
    return H
    
def gradient_vector(N, X, y, w, lambda_):
    '''Computes the gradient of the negative log-likelihood'''
    l = -((y-sigmoid(w.T @ X)) @ X.T).T + 1/lambda_*w
      
    return l

def fit(X, y):
    '''Data'''
    ones = np.ones((1,X.shape[1]))
    X = np.vstack([ones, X])                  # Prepend an all-ones feature to X
    epsilon = 1e-10                           # Convergence metric
    max_iter = 500

    '''Metrics'''
    d = X.shape[0]                           # Number of dimensions
    N = X.shape[1]                           # Number of samples
    C = list(set(y))                         # Classes
    lambda_ = 1                              # Regularization parameter

    '''Matrix initialization'''
    y = y[np.newaxis,:]
    w = np.random.randn(d,1)

    '''Newton-Raphson'''
    i = 0
    e = 2*epsilon
    while ((e > epsilon) and (i < max_iter)):
        w_prev = w
        H = hessian(N, d, X, w, lambda_)
        l = gradient_vector(N, X, y, w, lambda_)
        w = w - np.linalg.inv(H) @ l
        e = np.linalg.norm(w - w_prev)/np.linalg.norm(w_prev)
        i += 1

    return w

def predict(X, w):
    ones = np.ones((1,X.shape[1]))
    X = np.vstack([ones, X])                  # Prepend an all-ones feature to X
    N = X.shape[1]  # do this more elegantly with self.__N
    PCXW = np.squeeze(sigmoid(w.T @ X))
    y_pred = np.zeros(N)
    for n in range(N):
        if (PCXW[n] >= 0.5):
            y_pred[n] = 1
            
    return y_pred.astype(int)
