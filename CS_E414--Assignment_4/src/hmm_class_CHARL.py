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
from numpy.lib.shape_base import dsplit
from gaussian import Gaussian

np.seterr(divide="ignore")
np.seterr(under="ignore")

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
        The training data consists of l different observaion sequences, 
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
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting state.

        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
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
        #trans[-1, :] = 1. / n_states GIVEN CODE BY LECTURER
        trans[-1, 0:-1] = 1. / n_states
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
        # print(states.shape[0])
        # print(states[0:100,:])
        # print(states[100:200,:])
        # print(states[200:300,:])
        # print(states[300:400,:])
        # print(states[400:500,:])
        # print(states[500:600,:])
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
        #pass
        # In this function, you may want to take log 0 and obtain -inf.
        # To avoid warnings about this, you can use np.seterr.
        n_obs = signal.shape[1]
        n_states = trans.shape[0] -1 # num emitting states
        delta = np.zeros((n_states, n_obs))
        back = np.zeros((n_states, n_obs))
        for n in range(n_states):
            delta[n, 0] = np.log(dists[n].f(signal[:,0]) * trans[-1, n])
            back[n, 0] = -1

        for t in range(1, n_obs):
            for n in range(n_states):
                p = np.log( dists[n].f(signal[:,t]))
                col = np.zeros((n_states))
                for n2 in range(n_states):
                    col[n2] =  trans[n2, n] * np.exp( delta[n2, t-1])
                #if t > n_obs- 170 : print(col) Compare signature set 1 to 2. Set 2 has way more zeros
                index_max = np.argmax(col)
                max = col[index_max]
                delta[n, t] = p + np.log(max)
                back[n ,t] = index_max
                #if max == 0.0:
                #    delta[n, t] = p
                #    back[n ,t] = back[n ,t -1] 
                #else:
                #    delta[n, t] = p + np.log(max)
                #    back[n ,t] = index_max

        col = np.zeros((n_states))
        for n in range(n_states):
            col[n] =  trans[n, -1] * np.exp( delta[n, n_obs-1])
        index_max = np.argmax(col)
        ll = np.log(col[index_max])
        seq = np.zeros((n_obs))
        seq[n_obs -1] = index_max

        seq = seq.astype(int)
        for t in range(n_obs -2, -1, -1): # t = n_obs - 2 down to and including 0
            seq[t] = back[ seq[t +1], t + 1]

        #print(seq)
        #print(back[4,-200:]) #-190
        #print(back[:,-1])
        #print(dists[0].get_mean())
        #print(np.log(dists[0].f(signal[:,n_obs-1])))  #Signature set 2: -100
        #print(np.log(dists[4].f(signal[:,n_obs-1])))  #Signature set 2: -3 , which is expected
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
        #pass
        n_obs = signal.shape[1]
        n_states = trans.shape[0] -1 # num emitting states
        ll = 0
        prev_state = -1 # last row is initial null state
        for n in range(n_obs):
            state = seq[n]
            a = trans[prev_state, state]
            p = dists[state].likelihood(signal[:, n])
            ll = ll + np.log(a) + np.log(p)
            prev_state = state
        ll = ll + np.log(trans[prev_state, -1])
        return ll



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
        #pass
        
        n_obs = signal.shape[1]
        n_states = trans.shape[0] -1 # num emitting states
        fwd = np.zeros((n_states, n_obs))
        #================= METHOD 1: Rescale ====================================
        # fwd_sum = 0
        # for n in range(n_states):
        #     fwd[n, 0] = dists[n].f(signal[:,0]) * trans[n_states, n]
        #     fwd_sum = fwd_sum + fwd[n, 0]
        # fwd[:, 0] = fwd[:, 0] / fwd_sum
        # fwd_sum = np.log(fwd_sum)

        # for t in range(1, n_obs):
        #     t_sum = 0
        #     for n in range(n_states):
        #         p = dists[n].f(signal[:,t])
        #         sum = 0
        #         for n2 in range(n_states):
        #             sum = sum + trans[n2, n] * fwd[n2, t-1]
        #         fwd[n ,t] = p * sum
        #         t_sum = t_sum + fwd[n ,t]
        #     fwd[: ,t] = fwd[: ,t] / t_sum
        #     fwd_sum = fwd_sum + np.log(t_sum)
        
        # fwd_N_T = 0
        # for n in range(n_states):
        #     fwd_N_T = fwd_N_T + trans[n, n_states] * fwd[n, n_obs-1]
           
        # ll = np.log(fwd_N_T) + fwd_sum
        # return ll
        #================= METHOD 2: Log-sum exponent ============================
        for n in range(n_states):
            fwd[n, 0] = np.log(dists[n].f(signal[:,0]) * trans[-1, n])

        for t in range(1, n_obs):
            for n in range(n_states):
                p = np.log( dists[n].f(signal[:,t]))
                col = np.zeros((n_states))
                for n2 in range(n_states):
                    col[n2] =  trans[n2, n] * np.exp( fwd[n2, t-1])
                index_max = np.argmax(col)
                val = col[index_max]

                if (val != 0.0): # better way of doing this?
                    col = col/val
                    fwd[n ,t] = p + np.log(val) + np.log( np.sum(col) )
                else:
                    fwd[n ,t] = p


        col = np.zeros((n_states))
        for n in range(n_states):
            col[n] =  trans[n, n_states] * np.exp( fwd[n, n_obs-1])
        index_max = np.argmax(col)
        val = col[index_max]
        col = col/val
        ll =  np.log(val) + np.log( np.sum(col) )
        return ll

    def _calcstates(self, trans, dists):
        '''
        Calculate state sequences on the 'signals' maximizing the likelihood for 
        the given HMM parameters.
        
        Calculate the state sequences for each of the given 'signals', maximizing the 
        likelihood of the given parameters of a HMM model. This allocates each of the
        observations, in all the sequences, to one of the states. 
    
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

        #pass # The core of this function involves applying the _viterbi function to each signal stored in the model.
        # Remember to remove emitting states not used in the new state allocation.
        n_states  = trans.shape[0] -1
        n_obs = self._n_obs()
        states = np.zeros((n_obs, n_states)).astype(bool) # check for empty columns (state) at end i.e drop columns
        new_trans = np.zeros((n_states +1, n_states + 1))

        ll_sum = 0
        obs_count = 0
        for i in range(len(self.signals)):
            signal = self.signals[i]
            seq, ll = self._viterbi(signal, trans, dists)
            ll_sum = ll_sum + ll
            prev_state = -1 # state -1
            for j in range(len(seq)):
                states[obs_count, seq[j]] = True
                obs_count = obs_count + 1
                new_trans[prev_state, seq[j]] = new_trans[prev_state, seq[j]] + 1
                prev_state = seq[j]
            new_trans[prev_state, -1] = new_trans[prev_state, -1] + 1 # transistion to final non-emitting null state


        #check for empty states
        for s in range(n_states -1, -1, -1):
            #print(s)
            s_obs_indices = np.ravel(np.where(states[:,s]))
            if len(s_obs_indices) == 0: # no observation indices present for state s
                states = np.delete(states, s, 1)
                new_trans = np.delete(new_trans, s, 1) #delete column
                new_trans = np.delete(new_trans, s, 0) # delete row
        #print("==")
        n_states = new_trans.shape[0] - 1
        for s in range(n_states + 1):
            new_trans[s, :] = new_trans[s, :] / np.sum(new_trans[s, :]) #normalise row

        return states, new_trans, ll_sum



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
        
        Return
        ------
        covs: (n_states, d, d) ndarray
            The updated covariance matrices for each state

        means: (n_states, d) ndarray
            The updated means for each state
        '''
        #pass
        # In this method, if a class has no observations, assign it a mean of zero
        # In this method, estimate a full covariance matrix and discard the non-diagonal elements
        # if a diagonal covariance matrix is required.
        # In this method, if a zero covariance matrix is obtained, assign an identity covariance matrix
        n_obs, n_states  = states.shape
        d = self.data.shape[0]
        means = np.zeros((n_states, d))
        covs = np.zeros((n_states, d, d))
        for s in range(n_states):
            s_obs_indices = np.ravel(np.where(states[:,s]))
            if len(s_obs_indices) == 0:
                covs[s,:,:] = np.diag(np.ones((d)))
            else:
                s_obs = self.data[:,s_obs_indices]
                covs[s,:,:] = np.cov(s_obs, bias=True)
                means[s, :] = np.mean(s_obs, axis=1)
                if self.diagcov:
                    covs[s,:,:] = np.diag(np.diag(covs[s,:,:]))
        #print(means)
        return covs, means

               
    def _em(self, trans, states):
        '''
        Perform parameter estimation for a hidden Markov model (HMM).
    
        Perform parameter estimation for an HMM using multi-dimensional Gaussian 
        states.  The training observation sequences, signals,  are available 
        to the class, and states designates the initial allocation of emitting states to the
        signal time steps.   The HMM parameters are estimated using Viterbi 
        re-estimation. 
        
        Note: It is possible that some states are never allocated any 
        observations.  Those states are then removed from the states table, effectively redusing
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
        #print(covs) no valueable information
        #print("============") 
        #print(trans)
        dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
        oldstates, trans, oldLL = self._calcstates(trans, dists)
        #print(trans)
        converged = False
        iters = 0
        while not converged and iters <  self.maxiters:
            #pass # Perform one iteration of the EM algorithm and test for convergence here
            covs, means = self._updatecovs(oldstates)              
            dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
            oldstates, trans, newLL = self._calcstates(trans, dists)
            
            #print(oldLL)
            #print(newLL)
            if np.abs((newLL - oldLL) / oldLL) < self.rtol:
                converged = True
            oldLL = newLL
            iters += 1
        if iters >= self.maxiters:
            warn("Maximum number of iterations reached - HMM parameters may not have converged")

        covs, means = self._updatecovs(oldstates)              
        dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
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
        d = self.data.shape[0]
        samples = np.zeros((d,0))
        states = []
        state = -1
        n_states = self._n_states()
        while state != (n_states):
            r = np.random.rand()
            row = np.cumsum(self.trans[state, :])
            found_next_state = False
            s = 0
            while not found_next_state:
                if r < row[s]:
                    state = s
                    found_next_state = True
                s = s + 1
            if state < n_states:
                states.append(int(state))
                sample = self.dists[state].sample()
                samples = np.c_[samples, sample]

        return samples, states
            

        
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
        
        #pass # Using the function defined above, draw samples from the HMM 

if __name__ == "__main__":
    import doctest
    doctest.testmod() 
