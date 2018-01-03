import math
import numpy as np
from SiteTypes import siteTypes

class EmissionProbs(object):
    
    def __init__(self, theta):
        
        self._theta = theta
        
        # logHomP[i] = log[ P(emitting a hom. observation | state i) ]
        logHomP = []
        S = 0.0
        for state in xrange(theta.nStates):
            # e^res is the probability that no mutations occured between times 0 and T_state
            if state > 0:
                S -= 2*theta.segments.delta[state-1]*theta.uVals[state-1]
            
            # Now, let t = log[ P(no mutations between time T_state and the time of coalescence | state) ]
            t1 = -math.expm1(-theta.segments.delta[state]*theta.lmbVals[state])
            t2 = -math.expm1(-2*theta.segments.delta[state]*theta.uVals[state])
            t  = math.log1p(t2/t1 - t2)
            t -= math.log1p(2*theta.uVals[state]/theta.lmbVals[state])
            # sanity check; given that the coalescence time is between T_state and T_(state+1)
            assert t <= 0
            assert t >= (-2*theta.segments.delta[state]*theta.uVals[state])
            
            assert (S + t) <= 0.0
            logHomP.append(S + t)
        
        self._logHomP = np.array(logHomP)
        self._logHetP = np.log(-np.expm1(self._logHomP))
            
    # probability of emitting a hom. observation, given a specific state.    
    def homProb(self, state):
        return math.exp(self._logHomP[state])
    
    # probability of emitting a het. observation, given a specific state.    
    def hetProb(self, state):
        return math.exp(self._logHetP[state])
    
    # return a matrix with the emission probabilities
    def emissionMat(self):
        res = np.zeros( (self._theta.nStates, 2) )
        
        for i in xrange(self._theta.nStates):
            res[i,siteTypes.hom] = self.homProb(i)
            res[i,siteTypes.het] = self.hetProb(i)
        
        return res

    # Calculate the empirical log-likelihood of a specific hidden-state sequence.
    def logLikelihood(self, seq):
        res  = np.dot(seq.emissions[:,siteTypes.hom], self._logHomP)
        res += np.dot(seq.emissions[:,siteTypes.het], self._logHetP)
        
        return res

