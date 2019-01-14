import numpy as np
from scipy import linalg as LA
import time
from Parallel import writeOutput

# This container class stores the results of the forward-backward algorithm.
# It contains summary statistics (NOT entire sequence) on the inferred hidden-state sequence.
# seqLength     : Underlying suequence length.
# transitions   : ndarray. transitions[i,j] is the inferred number of transitions i->j
# emissions     : ndarray. emissions[i,j] is the inferred number of emissions i->j
# gamma0        : the posterior distribution of states at the beginning of the sequence
# logLikelihood : the log-l of the hidden & observed sequence.
class HiddenSeqSummary(object):
    
    def __init__(self, seqLength, transitions, emissions, gamma0, logLikelihood):
        
        self.length = seqLength
        
        # verify input arrays have valid dimensions
        nStates, nEmissions = emissions.shape
        assert transitions.shape == (nStates, nStates)
        assert gamma0.shape == (nStates, )
        
        # log( P(O|theta) ), where theta are the parameters used for the hidden-state inference
        # (ie, theta are the parameters used for the Baum-Welch expectation step)
        self.logL = logLikelihood
        
        self.gamma0      = gamma0
        self.emissions   = emissions
        self.transitions = transitions 
        
        # the following 4 arrays allow efficient calculation of Q (the target function in the BW maximization step):
        # TODO use or remove...
        # IncFrom[i] is the proportion of transitions i->j for some j>i
        self.incFrom = np.array([np.sum(self.transitions[i,(i+1):]) for i in xrange(nStates)])
        # DecFrom[i] is the proportion of transitions i->j for some j<i
        self.decFrom = np.array([np.sum(self.transitions[i,0:i]) for i in xrange(nStates)])
        #   IncTo[j] is the proportion of transitions i->j for some i<j
        self.incTo   = np.array([np.sum(self.transitions[0:j,j]) for j in xrange(nStates)])
        #   DecTo[j] is the proportion of transitions i->j for some i>j
        self.decTo   = np.array([np.sum(self.transitions[(j+1):,j]) for j in xrange(nStates)])
            
    # Combine two classes to one (ie calculate the combined statistics on both sequences)
    def __add__(self, other):
        
        assert self.transitions.shape == other.transitions.shape
        assert self.emissions.shape   == other.emissions.shape
        
        length          = self.length      + other.length
        transitions     = self.transitions + other.transitions
        emissions       = self.emissions   + other.emissions
        gammaa0         = self.gamma0      + other.gamma0
        logL            = self.logL        + other.logL
        
        return HiddenSeqSummary(length, transitions, emissions, gammaa0, logL)

# This class implements the expectation step of the BW algorithm (ie the forward-backward algorithm)
class BaumWelchExpectation(object):
    # TODO this could be a function rather than a class
    # theta : HMM instance
    # obs   : an observed emissions sequence (ObservedSequence class)
    def __init__(self, theta, obs):
        
        self._obs = obs
        # TODO reference theta instead? the entire hmm structure is duplicated here... depends on how I implement theta()
        self._nStates = theta.nStates
        self._nEmissions = theta.nEmissions
        self._initialDist = theta.initialDist
        self._transitionMat = theta.transitionMat
        self._emissionMat = theta.emissionMat
        
        self._run   = False
        
    # expectation step of BW    
    def inferHiddenStates(self):
        
        assert self._run == False
        
        start = time.time()
        self._runForward()
        writeOutput('BWE FORWARD : %f seconds'%(time.time()-start), filename ='DBG')
        
        start = time.time()
        self._runBackwards()
        writeOutput('BWE BACKWARD : %f seconds'%(time.time()-start), filename ='DBG')
        
        self._run = True
        
        return self._res
    
    # forward run of BW: compute alpha_t for all t's in self._obs.positions.
    # all alphas are scaled such that || alpha_t ||_1 = 1.
    # scaling factors are not saved.
    def _runForward(self):
        
        # pre-processing: compute auxilary matrices such that:
        # alpha_(t+d) = C*mats[j,d-1]*alpha_t, assuming
        #       1. O_(t+1) = ... = O_(t+d-1) = 0;  and
        #       2. O_(t+d) = j.
        # (Where C is a scaling constant; scalingFactors[j,d-1] = log(C))
        start = time.time()
        mats, scalingFactors = self._calcForwardMats()
        writeOutput('BWE FORWARD PREPROCESSING : %f seconds'%(time.time()-start), filename ='DBG')
            
        # initilize ndarray for forward variables
        # forward variables are calculated & stored for the positions in self._obs.positions
        self._alpha = np.empty( (self._obs.nPositions, self._nStates), dtype=np.float64 )
        
        # initilize alpha_0 (alpha_0 (i) = pi_i * b_i(O_0) )
        self._alpha[0,:]  = self._initialDist * self._emissionMat[:,self._obs.posTypes[0]]
        
        # calculate P(O|theta) = sum( alpha )  (re-correcting for scaling)
        s                   = np.sum(self._alpha[0,:])
        self._logLikelihood = np.log(s)
        
        # scale alpha_0 (all alpha vectors are scaled to sum to 1.0)
        self._alpha[0,:] /= s
        
        # calculate alpha_1, alpha_2, ..., recursively
        for ind in xrange(1, self._obs.nPositions):
            d = self._obs.positions[ind] - self._obs.positions[ind-1]
            j = self._obs.posTypes[ind]
            np.dot(mats[j, d-1, :, :], self._alpha[ind-1,:], out=self._alpha[ind,:])
            
            # scale alpha_ind and update self._logLikelihood
            s                    = np.sum(self._alpha[ind,:])
            self._alpha[ind,:]  /= s
            self._logLikelihood += (np.log(s) + scalingFactors[j, d-1])
            
    
    # backward run of BW: compute beta_t for all t's in self._obs.positions.
    # all betas are scaled such that || beta_t ||_1 = 1.
    # beta vectors & scaling factors are not saved;
    # TODO describe what's save
    def _runBackwards(self):
        
        # pre-processing: compute auxilary matrices
        # beta_t = C*mats[j,d-1]*beta_(t+d), assuming
        #       1. O_(t+1) = ... = O_(t+d-1) = 0;  and
        #       2. O_(t+d) = j.
        # (Where C is an arbitrary scaling constant)
        start = time.time()
        mats = self._calcBackwardMats()
        writeOutput('BWE BACKWARD PREPROCESSING : %f seconds'%(time.time()-start), filename ='DBG')
        
        # initialize beta_(L-1) (beta_(L-1) (i) = 1.0 )
        beta = np.ones( self._nStates, dtype=np.float64 )
        
        # allocate ndarray for temporary variables
        tmpVec = np.empty(  self._nStates,                       dtype=np.float64 )
        tmpMat = np.empty( (self._nStates, self._nStates), dtype=np.float64 )
        
        # initialize window histogram;
        # windowH[j,d-1] coressponds to windows [O_t, ..., O_t+d] where:
        #    - We don't assume anything about O_t
        #    - O_(t+1) = ... = O_(t+d-1) = 0.
        #    - O_(t+d) = j
        windowH = np.zeros( (self._nEmissions, self._obs.maxDistance, self._nStates, self._nStates) , dtype=np.float64 )
        
        # move backwards window by window 
        for ind in xrange(self._obs.nPositions - 2, -1, -1):
            
            # process window [pos_ind, pos_(ind+1)]; notice 'beta' variable corresponds to beta_(ind+1)
            d = self._obs.positions[ind+1] - self._obs.positions[ind]
            j = self._obs.posTypes[ind+1]
            
            np.outer(self._alpha[ind,:], beta, out=tmpMat)
            tmpMat *= mats[j, d-1, :, :]
            # normalize tmpMat to a matrix of probabilities
            # tmpMat (i,j) = Pr(state_ind = i, state_(ind+1)=j | entire observed sequence)
            tmpMat             /= np.sum(tmpMat)
            windowH[j,d-1,:,:] += tmpMat
            
            # update beta from beta_(ind+1) to beta_ind
            np.dot(mats[j, d-1, :, :], beta, out=tmpVec)
            beta, tmpVec = tmpVec, beta
            
            # normalize beta (scale to sum to 1.0)
            beta /= np.sum(beta)
        
        # initialize result arrays
        start = time.time()
        resTransitions = np.zeros( (self._nStates, self._nStates   ), dtype=np.float64 )
        resEmissions   = np.zeros( (self._nStates, self._nEmissions), dtype=np.float64 )
        
        # Handle emmision from fisrt bp separately... gamma[0] = alpha[0]*beta[0]
        gammaZero  = beta * self._alpha[0,:]
        gammaZero /= np.sum(gammaZero)
        resEmissions[:, self._obs.posTypes[0]] += gammaZero
                
        # The following section of the code
        # (Specifically: windowH[0, d-1, :, :] /= mats[0, d-1, :, :])
        # assumes that the matrices mats[j,d-1,:,:] don't have zero entries.
        assert np.count_nonzero(mats) == (self._nEmissions * self._obs.maxDistance * self._nStates * self._nStates)
        
        # parse the window histogram from largest to smallest
        for d in xrange(self._obs.maxDistance, 0, -1):
            
            # NOTE we could, instead, break windows in the middle (& handle emissions only & size 1)
            # The latter approach might be more stable numerically. 
            
            # we now parse all windows windowH[:, d]
            
            # first, handle emissions from the last pb in the window
            for outputType in xrange(self._nEmissions):
                
                # consider window windowH[outputType, d]
                # to get the posterior state probabilities in the last bp, sum over columns
                # (i.e. compute gamma at postion pos_(ind+1), simultaneously for all windows of this type)
                np.sum(windowH[outputType, d-1, :, :], axis=0, out=tmpVec)
                
                # add this to the appropriate emission vec
                resEmissions[:, outputType] += tmpVec
            
            # now, handle transitions:
            # aggregate all windows of type [*,d] to single bin (type only mattered for emissions)
            for outputType in xrange(1,self._nEmissions):
                windowH[0, d-1, :, :] += windowH[outputType, d-1, :, :]
            
            # window-size d is now aggregated in windowH[0, d].
                        
            if d > 1:
                # we'll break it in the bp before last, and attain
                #  - one window of size (d-1), ending in output 0
                #  - and one window of size 1 (don't care what output type in its end as we're done that emission)
                
                # How many windows are aggregated in this bin? (as all matrices added to the window sum to 1.0)
                nWindows = np.sum(windowH[0, d-1, :, :])
                windowH[0, d-1, :, :] /= LA.norm(windowH[0, d-1, :, :], ord=1)
                windowH[0, d-1, :, :] /= mats[0, d-1, :, :]
                
                # sub window of size (d-1):
                np.dot(windowH[0, d-1, :, :], mats[0, 0, :, :].T, out=tmpMat)
                tmpMat *= mats[0, d-2, :, :]
                tmpMat *= (nWindows/np.sum(tmpMat))
                windowH[0, d-2, :, :] += tmpMat      
                
                # sub window of size 1:
                np.dot(mats[0, d-2, :, :].T, windowH[0, d-1, :, :], out=tmpMat)
                tmpMat *= mats[0, 0, :, :]
                tmpMat *= (nWindows/np.sum(tmpMat))
                resTransitions += tmpMat
            
            else:
                # d=1
                resTransitions += windowH[0, d-1, :, :]
        
        self._res = HiddenSeqSummary(self._obs.length, resTransitions, resEmissions, gammaZero, self._logLikelihood)
        
        writeOutput('BWE POSTPROCESSING : %f seconds'%(time.time()-start), filename ='DBG')

    
    # calc forward auxiliary mats
    def _calcForwardMats(self):
        # TODO am I missing a high precision float by choosing float64?
        mats   = np.empty( (self._nEmissions, self._obs.maxDistance, self._nStates, self._nStates) , dtype=np.float64 )
        scales = np.empty( (self._nEmissions, self._obs.maxDistance) , dtype=np.float64 )
        
        # initilize mats for d = 1 (i.e. advancing one base pair)
        for outputType in xrange(self._nEmissions):
            for i in xrange(self._nStates):
                for j in xrange(self._nStates):
                    # recurrence formula: alpha_(t+1)(j) = sum_i {alpha_t(i) * a_ij * b_j(O_t)}
                    mats[outputType, 0, i, j] = self._transitionMat[j,i] * self._emissionMat[i,outputType]
            
            # normalize matrix to have L1 norm 1 (scaling)
            s                          = LA.norm(mats[outputType, 0, :, :],ord=1)
            mats[outputType, 0, :, :] /= s
            scales[outputType, 0]      = np.log(s)
        
        # for blocks larger than 1, compute matrices recursively
        # if the last output is 0, notice mats[0,d-1] = mats[0,0] ^ d (scaled to L1 norm of 1)
        self._powerArray(mats[0, :, :, :], scales[0,:])
        # otherwise, mats[type,d-1] = mats[type,0] * mats[0,d-2]
        for outputType in xrange(1,self._nEmissions):
            for d in xrange(2, self._obs.maxDistance + 1):
                np.dot(mats[outputType, 0, :, :], mats[0, d-2, :, :], out=mats[outputType, d-1, :, :])
        
                # normalize matrix to have L1 norm 1 (scaling)
                s                            = LA.norm(mats[outputType, d-1, :, :],ord=1)
                mats[outputType, d-1, :, :] /= s
                scales[outputType, d-1]      = scales[outputType, 0] + scales[0, d-2] + np.log(s)
        
        self._mats = mats
        self._matStatus = 'forward'
        return self._mats, scales
    
    # transpose self._mats
    def _calcBackwardMats(self):
        assert self._matStatus == 'forward'
        for outputType in xrange(self._nEmissions):
            for d in xrange(1, self._obs.maxDistance + 1):
                self._mats[outputType, d-1, :, :] = self._mats[outputType, d-1, :, :].T.copy()
                self._mats[outputType, d-1, :, :] /= LA.norm(self._mats[outputType, d-1, :, :],ord=1)
        
        self._matStatus = 'backward'
        return self._mats
        
    # calculate M^2, ..., M^n for a square m*m matrix M.
    # all matrices are scaled to have L1 norm of one.
    # A: a (n,m,m) ndarray, such that A[0,:,:]=M.
    # The result is stored such that A[i,:,:] = scaled(M^(i+1)) (i.e. M, M^2, ..., M^n)
    def _powerArray(self, A, scales):
        
        assert A.ndim == 3
        n = A.shape[0]
        assert A.shape[1] == A.shape[2]
        assert scales.shape == (n,)
        
        k = 1
        while k < n:
            for i in xrange(1, k+1):
                if (k+i) <= n:
                    
                    # calculate A^(k+i) = A^k * A^i
                    np.dot(A[k-1,:,:], A[i-1,:,:], out=A[k+i-1,:,:])
                    
                    # rescale A^(k+i)
                    s             = LA.norm(A[k+i-1,:,:],ord=1)
                    A[k+i-1,:,:] /= s
                    scales[k+i-1] = np.log(s) + scales[k-1] + scales[i-1]
                    
            k *= 2 
