import numpy as np

def kmeans(X, K, max_iter):
    N = X.shape[1]
    l = np.random.randint(0,K,(N,))
    C = list(set(l))
    
    for i in range(max_iter):
        u = []
        for c in C:
            u.append((np.sum(X[:,l==c], axis=1) / np.sum([l==c]))[:,np.newaxis])
        Xk = np.zeros((N,K))
        for k in range(K):
            Xk[:,k] = np.linalg.norm((X - u[k]), axis=0)
        l = np.argmin(Xk, axis=1)

    return u, l




# Short implementation go brrr hahah
# import numpy as np

# def kmeans(X, max_iter):
#     N = X.shape[1]
#     l = np.random.randint(0,2,(N,))
#     C = list(set(l))
#     for i in range(max_iter):
#         u = []
#         Xk = np.zeros((N,len(C)))
#         for c in C:
#             u.append((np.sum(X[:,l==c], axis=1) / np.sum([l==1]))[:,np.newaxis])
#             Xk[:,c] = np.linalg.norm((X - u[c]), axis=0)
#         l = np.argmin(Xk, axis=1)
#     return u, l








































