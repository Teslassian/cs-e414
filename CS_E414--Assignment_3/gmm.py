import numpy as np
from kmeans import *
from pprint import pprint
from time import sleep

def gmm(X, K, max_iter):
    D = X.shape[0]
    N = X.shape[1]
            
#     # Initialize with Kmeans
#     u, labels = kmeans(X, K=K, max_iter=10)
    
    # Initialize Randomly
    labels = np.random.randint(0, K, N)

            
    C = list(set(labels))
        
    Nj = []
    Pj = []
    Uj = []
    Sj = []
    for c, k in zip(C, range(K)):
        Nj.append(np.sum(labels==c))
        Pj.append(Nj[k]/N)
        Uj.append(1/Nj[k] * np.sum(X[:,labels==c], axis=1)[:,np.newaxis])
#         Sj.append(1/Nj[k] * (X[:,labels==c] - Uj[k]) @ (X[:,labels==c] - Uj[k]).T) # causes noninvertible matrices
        Sj.append(np.diag(np.ones(D))) # Initialize to prevent noninvertible matrices
    
    for i in range(max_iter):
        print('i')
        print(i)
        
        # Expectation Step
        PnjGnj = []
        PnjGnj_sum = 0
        Ynj = []
        for c, k in zip(C, range(K)):
            print('k')
            print(k)
            
            PnGn = np.zeros((N))
            for n in range(N):
                PnGn[n] = Pj[k] * 1/np.sqrt(np.linalg.det(2*np.pi*Sj[k])) * np.exp(-0.5*(X[:,n,None]-Uj[k]).T @ np.linalg.inv(Sj[k]) @ (X[:,n,None]-Uj[k]))
            PnjGnj.append(PnGn)
            PnjGnj_sum += PnjGnj[k]
        for c, k in zip(C, range(K)):
            Ynj.append(PnjGnj[k] / PnjGnj_sum)

        # Maximization Step
        Nj = []
        Pj = []
        Uj = []
        Sj = []
        for c, k in zip(C, range(K)):
            Nj.append(np.sum(Ynj[k]))
            Pj.append(Nj[k]/N)
            Uj.append((1/Nj[k] * X @ Ynj[k][:,np.newaxis]))
            Sj.append(1/Nj[k] * (Ynj[k] * (X - Uj[k])) @ (X - Uj[k]).T)

        # Log-Likelihood
        PnjGnj = []
        PnjGnj_sum = 0
        for c, k in zip(C, range(K)):
            PnGn = np.zeros((N))
            for n in range(N):
                PnGn[n] = Pj[k] * 1/np.sqrt(np.linalg.det(2*np.pi*Sj[k])) * np.exp(-0.5*(X[:,n,None]-Uj[k]).T @ np.linalg.inv(Sj[k]) @ (X[:,n,None]-Uj[k]))
            PnjGnj.append(PnGn)
            PnjGnj_sum += PnjGnj[k]
        LL = np.sum(np.log(PnjGnj_sum))

    Probabilities = Ynj[0]
    for k in range(K-1):
        Probabilities = np.stack((Probabilities, Ynj[k+1]), axis=0)
    labels = np.argmax(Probabilities, axis=0)
    
    return Probabilities, Nj, Uj, Sj