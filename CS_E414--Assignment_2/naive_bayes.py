import numpy as np

def predict(X, y):

    '''Metrics'''
    d = X.shape[0]                           # Number of dimensions
    N = X.shape[1]                           # Number of samples
    C = list(set(y))                         # Classes
    K = len(C)                               # Number of classes
    Nj = []                                  # Number of samples per class
    for c in C:
        Nj.append(len(y[y==c]))
        
    '''Maximum Likelihood Estimations'''
    PCj = []   # Prior Probabilities
    Uj = []    # Class Means
    u_Sj = []  # Variances
    Sj = []    # Diagonal Covariance Matrix - check how to do this if it's shared
    for c, i in zip(C, range(K)):
        PCj.append(Nj[i]/N)
        Uj.append(1/Nj[i] * np.sum(X[:,y==c], axis=1)[:,np.newaxis])
    #     u_Sj.append(1/Nj[i] * np.sum((X[:,y==c]-Uj[i]) * (X[:,y==c]-Uj[i]), axis=1))
        Sj.append(np.diag(np.diagonal(1/Nj[i] * (X[:,y==c]-Uj[i]) @ (X[:,y==c]-Uj[i]).T)))
        
    '''Class-conditional distributions'''
    PXCj = []    
    for i in range(K):
        temp = 1/np.sqrt(np.linalg.det(2*np.pi*Sj[i]))  *  np.exp(-0.5 * (X-Uj[i]).T @ np.linalg.inv(Sj[i]) @ (X-Uj[i]))
        PXCj.append(np.diagonal(temp))
        
    '''Posterior Probabilities'''
    PCjX = []
    for i in range(K):
        PCjX.append(PCj[i] * PXCj[i])
        
    '''Maximum Posterior Selection'''
    y_pred = np.zeros(N)
    for n in range(N):
        temp = np.zeros(K)
        for i in range(K):
            temp[i] = PCjX[i][n]
        y_pred[n] = np.argmax(temp)
    y_pred = y_pred.astype(int)
    
    return y_pred