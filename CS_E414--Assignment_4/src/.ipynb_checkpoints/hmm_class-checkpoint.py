'''
Module implementing Hidden Markov model parameter estimation.

To avoid repeated warnings of the form "Warning: divide by zero encountered in log", 
it is recommended that you use the command "np.seterr(divide="ignore")" before 
invoking methods in this module.  This warning arises from the code using the 
fact that python sets log 0 to "-inf", to keep the code simple.

Initial version created on Mar 28, 2012

@author: kroon, herbst
'''

from warnings import warn
import numpy as np
from gaussian import Gaussian
np.seterr(divide="ignore")

class HMM(object):
    '''
    Class for representing and using hidden Markov models.
    Currently, this class only supports left-to-right topologies and Gaussian
    emission densities.

    The HMM is defined for n_states emitting states (i.e. states with 
    observational pdf's attached), and an initial and final non-emitting state (with no 
    pdf's attached). The emitting states always use indices 0 to (n_states-1) in the code.
    Indices -1 and n_states are used for the non-emitting states (-1 for the initial and
    n_state for the terminal non-emitting state). Note that the number of emitting states
    may change due to unused states being removed from the model during model inference.

    To use this class, first initialize the class, then either use load() to initialize the
    transition table and emission densities, or fit() to initialize these by fitting to
    provided data.  Once the model has been fitted, one can use viterbi() for inferring
    hidden state sequences, forward() to compute the likelihood of signals, score() to
    calculate likelihoods for observation-state pairs, and sample()
    to generate samples from the model.
        
    Attributes:
    -----------
    data : (d,n_obs) ndarray 
        An array of the trainining data, consisting of several different
        sequences.  Thus: Each observation has d features, and there are a total of n_obs
        observation.   An alternative view of this data is in the attribute signals.

    diagcov: boolean
        Indicates whether the Gaussians emission densities returned by training
        should have diagonal covariance matrices or not.
        diagcov = True, estimates diagonal covariance matrix
        diagcov = False, estimates full covariance matrix

    dists: (n_states,) list
        A list of Gaussian objects defining the emitting pdf's, one object for each 
        emitting state.

    maxiters: int
        Maximum number of iterations used in Viterbi re-estimation.
        A warning is issued if 'maxiters' is exceeded. 

    rtol: float
        Error tolerance for Viterbi re-estimation.
        Threshold of estimated relative error in log-likelihood (LL).

    signals : ((d, n_obs_i),) list
        List of the different observation sequences used to train the HMM. 
        'd' is the dimension of each observation.
        'n_obs_i' is the number of observations in the i-th sequence.
        An alternative view of thise data is in the attribute data.
            
    trans : (n_states+1,n_states+1) ndarray
        The left-to-right transition probability table.  The rightmost column contains probability
        of transitioning to final state, and the last row the initial state's
        transition probabilities.   Note that all the rows need to add to 1. 
    
    Methods:
    --------
    fit():
        Fit an HMM model to provided data using Viterbi re-estimation (i.e. the EM algorithm).

    forward():
        Calculate the log-likelihood of the provided observation.

    load():
        Initialize an HMM model with a provided transition matrix and emission densities
    
    sample():
        Generate samples from the HMM
    
    viterbi():
        Calculate the optimal state sequence for the given observation 
        sequence and given HMM model.
    
    Example (execute the class to run the example as a doctest)
    -----------------------------------------------------------
    >>> import numpy as np
    >>> from gaussian import Gaussian
    >>> signal1 = np.array([[ 1. ,  1.1,  0.9, 1.0, 0.0,  0.2,  0.1,  0.3,  3.4,  3.6,  3.5]])
    >>> signal2 = np.array([[0.8, 1.2, 0.4, 0.2, 0.15, 2.8, 3.6]])
    >>> data = np.hstack([signal1, signal2])
    >>> lengths = [11, 7]
    >>> hmm = HMM()
    >>> hmm.fit(data,lengths, 3)
    >>> trans, dists = hmm.trans, hmm.dists
    >>> means = [d.get_mean() for d in dists]
    >>> covs = [d.get_cov() for d in dists]
    >>> covs = np.array(covs).flatten()
    >>> means = np.array(means).flatten()
    >>> print(trans)
    [[0.66666667 0.33333333 0.         0.        ]
     [0.         0.71428571 0.28571429 0.        ]
     [0.         0.         0.6        0.4       ]
     [1.         0.         0.         0.        ]]
    >>> print(covs)
    [0.01666667 0.01459184 0.0896    ]
    >>> print(means)
    [1.         0.19285714 3.38      ]
    >>> signal = np.array([[ 0.9515792,   0.9832767,   1.04633007,  1.01464327,  0.98207072,        	1.01116689, 0.31622856,  0.20819263,  3.57707616]])
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)  
    [0 0 0 0 0 0 1 1 2]
    >>> print(ll)
    2.9053411643345126
    >>> hmm.load(trans, dists)
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 0 0 0 0 0 1 1 2]
    >>> print(ll)
    2.9053411643345126
    >>> print(hmm.score(signal, vals))
    2.905341164334513
    >>> print(hmm.forward(signal))
    2.905342356042732
    >>> signal = np.array([[ 0.9515792,   0.832767,   3.57707616]])
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 1 2]
    >>> print(ll)
    -14.975826945102282
    >>> samples, states = hmm.sample()
    '''

    def __init__(self, diagcov=True, maxiters=20, rtol=1e-4): 
        '''
        Create an instance of the HMM class, with n_states hidden emitting states.
        
        Parameters
        ----------
        diagcov: boolean
            Indicates whether the Gaussians emission densities returned by training
            should have diagonal covariance matrices or not.
            diagcov = True, estimates diagonal covariance matrix
            diagcov = False, estimates full covariance matrix

        maxiters: int
            Maximum number of iterations used in Viterbi re-estimation
            Default: maxiters=20

        rtol: float
            Error tolerance for Viterbi re-estimation
            Default: rtol = 1e-4
        '''
        
        self.diagcov = diagcov
        self.maxiters = maxiters
        self.rtol = rtol
        
    def fit(self, data, lengths, n_states):
        '''
        Fit a left-to-right HMM model to the training data provided in `data`.
        The training data consists of l different observation sequences, 
        each sequence of length n_obs_i specified in `lengths`. 
        The fitting uses Viterbi re-estimation (an EM algorithm).

        Parameters
        ----------
        data : (d,n_obs) ndarray 
            An array of the training data, consisting of several different
            sequences. 
            Note: Each observation has d features, and there are a total of n_obs
            observation. 

        lengths: (l,) int ndarray 
            Specifies the length of each separate observation sequence in `data`
            There are l difference training sequences.

        n_states : int
            The number of hidden emitting states to use initially. 
        '''
        
        # Split the data into separate signals and pass to class
        self.data = data
        newstarts = np.cumsum(lengths)[:-1]
        self.signals = np.hsplit(data, newstarts)
        self.trans = HMM._ltrtrans(n_states)
        self.trans, self.dists, newLL, iters = self._em(self.trans, self._ltrinit())

    def load(self, trans, dists):
        '''
        Initialize an HMM model using the provided data.

        Parameters
        ----------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
        
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting state.
    
        '''

        self.trans, self.dists = trans, dists

    def _n_states(self):
        '''
        Get the number of emitting states used by the model.

        Return
        ------
        n_states : int
        The number of hidden emitting states to use initially. 
        '''

        return self.trans.shape[0]-1

    def _n_obs(self):
        '''
        Get the total number of observations in all signals in the data associated with the model.

        Return
        ------
        n_obs: int 
            The total number of observations in all the sequences combined.
        '''

        return self.data.shape[1]

    @staticmethod
    def _ltrtrans(n_states):
        '''
        Intialize the transition matrix (self.trans) with n_states emitting states (and an initial and 
        final non-emitting state) enforcing a left-to-right topology.  This means 
        broadly: no transitions from higher-numbered to lower-numbered states are 
        permitted, while all other transitions are permitted. 
        All legal transitions from a given state should be equally likely.

        The following exceptions apply:
        -The initial state may not transition to the final state
        -The final state may not transition (all transition probabilities from 
         this state should be 0)
    
        Parameter
        ---------
        n_states : int
            Number of emitting states for the transition matrix

        Return
        ------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table initialized as described below.
        '''

        trans = np.zeros((n_states + 1, n_states + 1))
        trans[-1, :] = 1. / n_states
        for row in range(n_states):
            prob = 1./(n_states + 1 - row)
            for col in range(row, n_states+1):
                trans[row, col] = prob
        return trans

    def _ltrinit(self):
        '''
        Initial allocation of the observations to states in a left-to-right manner.
        It uses the observation data that is already available to the class.
    
        Note: Each signal consists of a number of observations. Each observation is 
        allocated to one of the n_states emitting states in a left-to-right manner
        by splitting the observations of each signal into approximately equally-sized 
        chunks of increasing state number, with the number of chunks determined by the 
        number of emitting states.
        If 'n' is the number of observations in signal, the allocation for signal is specified by:
        np.floor(np.linspace(0, n_states, n, endpoint=False))
    
        Returns
        ------
        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.
        '''

        states = np.zeros((self._n_obs(), self._n_states()))
        i = 0
        for s in self.signals:
            vals = np.floor(np.linspace(0, self._n_states(), num=s.shape[1], endpoint=False))
            for v in vals:
                states[i][int(v)] = 1
                i += 1
        return np.array(states,dtype = bool)

    def viterbi(self, signal):
        '''
        See documentation for _viterbi()
        '''
        return HMM._viterbi(signal, self.trans, self.dists)

    @staticmethod
    def _viterbi(signal, trans, dists):
        '''
        Apply the Viterbi algorithm to the observations provided in 'signal'.
        Note: `signal` is a SINGLE observation sequence.
    
        Returns the maximum likelihood hidden state sequence as well as the
        log-likelihood of that sequence.

        Note that this function may behave strangely if the provided sequence
        is impossible under the model - e.g. if the transition model requires
        more observations than provided in the signal.
    
        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        seq : (n,) ndarray
            The optimal state sequence for the signal (excluding non-emitting states)

        ll : float
            The log-likelihood associated with the sequence
        '''
        
        # In this function, you may want to take log 0 and obtain -inf.
        # To avoid warnings about this, you can use np.seterr.
        
        np.seterr(divide='ignore')
        
        N = len(dists)
        T = signal.shape[1]
        
        delta = np.zeros((N,T))
        b = np.zeros((N,T))
        for t in range(T):
            for j in range(N):
                if (t==0):
                    delta[j,t] = dists[j].f(signal[:,t]) * trans[-1,j]
                    b[j,t] = -1
                else:
                    delta[j,t] = dists[j].f(signal[:,t]) * np.max(trans[0:-1,j] * delta[:,t-1])
                    b[j,t] = np.argmax(trans[0:-1,j] * delta[:,t-1])
        b_N = np.argmax(trans[0:-1,j] * delta[:,t])
        
        delta_T = np.max(trans[0:-1,-1] * delta[:,-1])
        ll = np.log(delta_T)
        
        seq = np.zeros((T)).astype(int)
        for t in range(T-1,0,-1):
            if (t==T-1): seq[t] = b_N
            else: seq[t] = b[seq[t+1],t+1]
        
        return seq, ll
        
    def score(self, signal, seq):
        '''
        See documentation for _score()
        '''
        return HMM._score(signal, seq, self.trans, self.dists)
    @staticmethod
    def _score(signal, seq, trans, dists):
        '''
        Calculate the likelihood of an observation sequence and hidden state correspondence.
        Note: signal is a SINGLE observation sequence, and seq is the corresponding series of
        emitting states being scored.
    
        Returns the log-likelihood of the observation-states correspondence.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        seq : (n,) ndarray
            The state sequence provided for the signal (excluding non-emitting states)

        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        ll : float
            The log-likelihood associated with the observation and state sequence under the model.
        '''
    
        T = len(seq)
        seq = np.insert(seq, len(seq), np.max(seq)+1)
        seq = np.insert(seq, len(seq), -1)
        
        p = 1
        for i in range(-1,T):
            p_xs = 1 if (seq[i]==-1) else dists[seq[i]].f(signal[:,i])
            a = trans[seq[i], seq[i+1]]
            p *= a*p_xs
        
        return np.sum(np.log(p))
  
    def forward(self, signal):
        '''
        See documentation for _forward()
        '''
        return HMM._forward(signal, self.trans, self.dists)

    @staticmethod
    def _forward(signal, trans, dists):
        '''
        Apply the forward algorithm to the observations provided in 'signal' to
        calculate its likelihood.
        Note: `signal` is a SINGLE observation sequence.
    
        Returns the log-likelihood of the observation.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        ll : float
            The log-likelihood associated with the observation under the model.
        '''
        
#         print('signal')
#         print(signal)
#         print('\ntrans')
#         print(trans)
#         print('\ndists')
#         print(dists[0].f(signal[:,0]))
        
#         N = len(dists)
#         print('\nN')
#         print(N)
#         T = signal.shape[1]
#         print('\nT')
#         print(T)
#         alpha = np.zeros((N,T))
#         for i in range(T):
#             for j in range(N):
#                 if (i==0):
#                     alpha[j,i] = dists[j].f(signal[:,i]) * trans[-1,j]
#                 else:
#                     alpha[j,i] = dists[j].f(signal[:,i]) * np.sum(trans[0:-1,j] * alpha[:,i-1])
#         alpha_T = np.sum(trans[0:-1,-1] * alpha[:,-1])
#         log_alpha_T = np.log(alpha_T)

#         print('\nlog_alpha_T')
#         print(log_alpha_T)
        
#         return log_alpha_T
    
#         def alpha(t, j, signal, trans, dists):
#             if (t==0):
#                 return np.log(dists[j].f(signal[:,0])) + np.log(trans[-1,j])
#             sum_ = np.log(0)
#             for i in range(len(dists)):
#                 sum_ = np.logaddexp(sum_, np.log(trans[i,j]) + alpha(t-1, i, signal, trans, dists))
#             return np.log(dists[j].f(signal[:,t])) + sum_
        
#         total = np.log(0)
#         for j in range(len(dists)):
#             total = np.logaddexp(total, np.log(trans[j,-1]) + alpha(signal.shape[1]-1, j, signal, trans, dists))
#         return total
    
        log_alpha_t = np.log(0)
        for i in range(len(dists)):
            log_alpha_t = np.logaddexp(log_alpha_t,   np.log(dists[i].f(signal[:,i])))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#         trans = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        
#         print(trans)
#         print(alpha)
#         print(np.round(np.log(alpha),2))

        #Recursive Implementation
#         def p(t): return dists[0].f(signal[:,t])
        
#         def rec(t):
#             print(t)
#             print(p(t))
#             if (t==0): return p(t)
#             else: return p(t) * rec(t-1)
            
#         print(rec(T-1))
        

#         #LogAddExp Implementation
#         def p(t,j): return dists[j].f(signal[:,t])
        
#         def alpha(logSum, j):
# #             print(f'j: {j}')
#             if (j==0):
#                 return np.log(p(0,j) * trans[-1,j])
#             else:
#                 return alpha(np.logaddexp())
#             return (p(3,j) * np.logaddexp(logSum, np.log(alpha(logSum,3,j-1))))
                                
        
#         print(alpha(np.log(0),0))
        
#         return log_alpha_T

    def _calcstates(self, trans, dists):
        '''
        Calculate state sequences on the 'signals' maximizing the likelihood for 
        the given HMM parameters.
        
        Calculate the state sequences for each of the given 'signals', maximizing the 
        likelihood of the given parameters of a HMM model. This allocates each of the
        observations, in all the equences, to one of the states. 
    
        Use the state allocation to calculate an updated transition matrix.   
    
        IMPORTANT: As part of this updated transition matrix calculation, emitting states which 
        are not used in the new state allocation are removed.
    
        In what follows, n_states is the number of emitting states described in trans, 
        while n_states' is the new number of emitting states.
        
        Note: signals consists of ALL the training sequences and is available
        through the class.
        
        Parameters
        ----------        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.
    
        Return
        ------    
        states : bool (n_obs,n_states') ndarray
            The updated state allocations of each observation in all signals
        trans : (n_states'+ 1,n_states'+1) ndarray
            Updated transition matrix 
        ll : float
            Log-likelihood of all the data
        '''
        
        # The core of this function involves applying the _viterbi function to each signal stored in the model.
        # Remember to remove emitting states not used in the new state allocation.
        
        signals = self.signals
        S = len(signals)
        n_obs_arr = np.zeros(S)
        for s in range(S): n_obs_arr[s] = signals[s].shape[1]
        n_obs_cum = np.cumsum(n_obs_arr)
        n_obs_cum = np.insert(n_obs_cum, 0, 0).astype(int)
        n_obs_all = np.sum(n_obs_arr).astype(int)

        seq_vect = np.array([])
        seq_list = []
        ll_list = []
        for (s, signal) in zip(range(len(signals)), signals):
            seq, ll = self._viterbi(signal, trans, dists)
        #             seq = np.where(seq==1, 2, seq)   # testing code
            seq_vect = np.concatenate((seq_vect, seq)).astype(int)
            seq_list.append(seq)
            ll_list.append(ll)

        n_states = np.unique(seq_vect).size
        states_new = np.zeros((n_obs_all, n_states)) - 99
        trans_new = np.zeros((n_states+1,n_states+1))

        for (s, signal, seq) in zip(range(S), signals, seq_list):
            u_states = list(set(seq))
            u_states.sort()

            for j, j_state in zip(range(n_states), u_states):
                states_new[n_obs_cum[s]:n_obs_cum[s+1],j] = seq==j_state

            seq = np.insert(seq, len(seq), np.max(seq)+1)
            seq = np.insert(seq, len(seq), -1)

            u_states = list(set(seq))
            u_states.sort()

            for i, i_state in zip(range(-1,n_states), u_states[:-1]):
                for j, j_state in zip(range(n_states+1), u_states[1:]):
                    if (i==-1): trans_new[i,j] += int(seq[0] == j_state)
                    else: trans_new[i,j] += np.sum(seq[np.where(seq==i_state)[0]+1] == j_state)

        trans_new /= np.sum(trans_new, axis=1, keepdims=True)
        ll_all = sum(ll_list)
        
        return states_new, trans_new, ll_all
        
        
    def _updatecovs(self, states):
        '''
        Update estimates of the means and covariance matrices for each HMM state
    
        Estimate the covariance matrices for each of the n_states emitting HMM states for 
        the given allocation of the observations in self.data to states. 
        If self.diagcov is true, diagonal covariance matrices are returned.

        Parameters
        ----------
        states : bool (n_obs,n_states) ndarray
            Current state allocations for self.data in model
        
                    states : (n_obs, n_states) ndarray
                        Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
                        'states[:,j]' specifies the allocation of all the observations to state j.
        
        Return
        ------
        covs: (n_states, d, d) ndarray
            The updated covariance matrices for each state

        means: (n_states, d) ndarray
            The updated means for each state
        '''

        # In this method, if a class has no observations, assign it a mean of zero
        # In this method, estimate a full covariance matrix and discard the non-diagonal elements
        # if a diagonal covariance matrix is required.
        # In this method, if a zero covariance matrix is obtained, assign an identity covariance matrix

        diagcov = self.diagcov
        data = self.data
        d = data.shape[0]
        n_states = states.shape[1]
        covs = np.zeros((n_states, d, d))
        means = np.zeros((n_states, d))
        
        for j in range(n_states):
            x = data[:,states[:,j]==1]
            N = x.shape[1]
            if (x.any()): means[j,:] = np.mean(x)
            covs[j,:,:] = 1/N * (x - means[j,:,None]) @ (x-means[j,:,None]).T
            if diagcov: covs[j,:,:] = np.diag(np.diag(covs[j,:,:]))
            if not covs[j,:,:].any(): covs[j,:,:] = np.eye(d)

        return covs, means
        
    def _em(self, trans, states):
        '''
        Perform parameter estimation for a hidden Markov model (HMM).
    
        Perform parameter estimation for an HMM using multi-dimensional Gaussian 
        states.  The training observation sequences and signals are available 
        to the class, and states designates the initial allocation of emitting states to the
        signal time steps. The HMM parameters are estimated using Viterbi 
        re-estimation. 
        
        Note: It is possible that some states are never allocated any 
        observations.  Those states are then removed from the states table, effectively reducing
        the number of emitting states. In what follows, n_states is the original 
        number of emitting states, while n_states' is the final number of 
        emitting states, after those states to which no observations were assigned,
        have been removed.
    
        Parameters
        ----------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
        
        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.
        
        Return
        ------
        trans : (n_states'+1,n_states'+1) ndarray
            Updated transition probability table

        dists : (n_states',) list
            Gaussian object of each component.

        newLL : float
            Log-likelihood of parameters at convergence.

        iters: int
            The number of iterations needed for convergence
        '''
        
        covs, means = self._updatecovs(states) # Initialize the covariances and means using the initial state allocation                
        dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
        oldstates, trans, oldLL = self._calcstates(trans, dists)
        converged = False
        iters = 0
        
        while (converged==False) and iters <  self.maxiters:
            # Perform one iteration of the EM algorithm and test for convergence here
            covs, means = self._updatecovs(oldstates)
            dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
            oldstates, trans, newLL = self._calcstates(trans, dists)
            if np.abs((newLL - oldLL) / oldLL) < self.rtol: converged = True
            oldLL = newLL
            iters += 1
            
        if iters >= self.maxiters:
            warn("Maximum number of iterations reached - HMM parameters may not have converged")
                 
        return trans, dists, newLL, iters
        
    def sample(self):
        '''
        Draw samples from the HMM using the present model parameters. The sequence
        terminates when the final non-emitting state is entered. For the
        left-to-right topology used, this should happen after a finite number of 
        samples is generated, modeling a finite observation sequence. 
        
        Returns
        -------
        samples: (n,) ndarray
            The samples generated by the model
        states: (n,) ndarray
            The state allocation of each sample. Only the emitting states are 
            recorded. The states are numbered from 0 to n_states-1.

        Sample usage
        ------------
        Example below commented out, since results are random and thus not suitable for doctesting.
        However, the example is based on the model fit in the doctests for the class.
        #>>> samples, states = hmm.samples()
        #>>> print(samples)
        #[ 0.9515792   0.9832767   1.04633007  1.01464327  0.98207072  1.01116689
        #  0.31622856  0.20819263  3.57707616]           
        #>>> print(states)   #These will differ for each call
        #[1 1 1 1 1 1 2 2 3]
        '''
        
        #######################################################################
        import scipy.interpolate as interpolate
        def draw_discrete_sample(discr_prob):
            '''
            Draw a single discrete sample from a probability distribution.
            
            Parameters
            ----------
            discr_prob: (n,) ndarray
                The probability distribution.
                Note: sum(discr_prob) = 1
                
            Returns
            -------
            sample: int
                The discrete sample.
                Note: sample takes on the values in the set {0,1,n-1}, where
                n is the the number of discrete probabilities.
            '''

            if not np.sum(discr_prob) == 1:
                raise ValueError('The sum of the discrete probabilities should add to 1')
            x = np.cumsum(discr_prob)
            x = np.hstack((0.,x))
            y = np.array(range(len(x)))
            fn = interpolate.interp1d(x,y)           
            r = np.random.rand(1)
            return np.array(np.floor(fn(r)),dtype=int)[0]
        #######################################################################

        
        N = self.trans.shape[0] - 1
        state = 0
        states = []
        samples = []
        while state < 3:
            states.append(state)
            samples.append((self.dists)[state].sample())
            state = draw_discrete_sample(self.trans[state,:])
        return samples, states

if __name__ == "__main__":
    import doctest
    doctest.testmod() 
