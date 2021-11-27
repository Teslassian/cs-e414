'''
Computes responsibilities. Assumes one-dimensional data and a k component mixture model.

@param p: mixture coeffecients.
@type p: 1-dimensional real valued list of length k.

@param u: class means.
@type u: 1-dimensional real valued list of length k.

@param s: class standard deviations.
@type s: 1-dimensional real valued list of length k.

@param x: vector of scalar observations
@type x: 1-dimensional real valued list of length n.

@param c: class label
@type c: an integer in the range [0, k-1]

@return: the calculated responsibility of each observation associated with class c
@rtype: 1-dimensional real valued list of size n
'''
def estimate_gamma(p,u,s,x,c):
    import math
    from math import exp
    from math import sqrt

    g = []
    for i in range(len(x)):
        gamma = (1.0) / (s[c] * sqrt(math.pi * 2))
        gamma = p[c] *  gamma * math.exp((-1.0 / (2 * (s[c] ** 2))) * ((x[i] - u[c]) ** 2))
        g.append(gamma)

    sum_vals = []
    for i in range(len(x)):
        sum_val = 0
        for j in range(len(u)):
            gamma = (1.0) / (s[j] * sqrt(math.pi * 2))
            gamma = p[j] *  gamma * math.exp((-1.0 / (2 * (s[j] ** 2))) * ((x[i] - u[j]) ** 2))
            sum_val += gamma
        sum_vals.append(sum_val)

    for i in range(len(x)):
        g[i] = g[i] / sum_vals[i]
        g[i] = round(g[i], 4)

    return g