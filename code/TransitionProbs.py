import numpy as np
import math
import scipy.integrate as integrate
from Parallel import writeOutput

# TODO replace all asserts with assertTrue etc

# transitionProbs: This class calculates the transition probabilities of the model.
class TransitionProbs(object):
    
    # theta: a Theta instance
    def __init__(self, theta):
        
        # debug 
        #writeOutput('trnsProb init', 'DBG')
        #theta.printVals('DBG')
        
        self._theta = theta
        
        # Notations:
        # The transition probability p_ij from state i to state j could be decomposed as follows:
        # - If i<j, ('increasing' transition), log(p_ij) = IncFrom_i + IncTo_j.
        # - If i>j, ('decreasing' transition), log(p_ij) = DecFrom_i + DecTo_j.
        # The 'diagonal' transitions p_ii are calculated indirectly by completion to one ( p_ii = 1 - sum_{j!=i}p_ij ).
        
        # Calculate the coefficiants IncFrom, IncTo, DecFrom, DecTo (and other temp. variables)
        self._logLDict = dict()
        self._calcS()
        self._calcIncFrom()
        self._calcDecFrom()
        self._calcIncTo()
        self._calcDecTo()
        self._calcDiagonal()
        self._calcStationary()
    
    # Return the transition probability from state origin to state target.
    def transitionProb(self, origin, target):
        
        if origin < target:
            res = self._IncFrom[origin] + self._IncTo[target]
        elif origin > target:
            res = self._DecFrom[origin] + self._DecTo[target]
        else:
            res = self._diagonal[origin]
        
        assert res <= 0.0
        return math.exp(res)
    
    # return a matrix with the transition probabilities
    def transitionMat(self):
        
        res = np.zeros( (self._theta.nStates, self._theta.nStates) )
        
        for i in xrange(self._theta.nStates):
            for j in xrange(self._theta.nStates):
                res[i,j] = self.transitionProb(i, j)
        
        return res
    
    # return a vector with the stationary distribution
    def stationaryProb(self):
        
        return np.copy(self._stationaryProb)
    
    # Calculate the empirical log-likelihood of a specific hidden sequence (excluding emissions)
    # seq is a HiddenSeqSummary instance
    def logLikelihood(self, seq):
        
        res = 0.0
        
        for i in xrange(self._theta.nStates):
            res += seq.transitions[i,i]*self._diagonal[i]
            res += seq.gamma0[i]*math.log(self._stationaryProb[i])
        
        for i in xrange(self._theta.nStates - 1):
            res += seq.incFrom[i] * self._IncFrom[i]
            res += seq.decTo[i]   * self._DecTo[i]
        
        for i in xrange(1, self._theta.nStates):
            res += seq.incTo[i]   * self._IncTo[i]
            res += seq.decFrom[i] * self._DecFrom[i]
        
        return res            
        
    
    # S_j = sum_(k=0,..,j-1) (1-e^-2*lmb_k*delta_k)*lStar(T_(k+1),T_j)/(2*lmb_k)
    def _calcS(self):

        self._S = dict()        
        S = 0.0
        for j in xrange(self._theta.nStates-1):
            
            # Update S
            if j > 0:
                S *= self._lStar(j-1,j)
                S += (-math.expm1(-2.0*self._theta.lmbVals[j-1]*self._theta.segments.delta[j-1])
                      /(2.0*self._theta.lmbVals[j-1])
                     )
            self._S[j] = S
    
    # calculate the coefficients DecFrom (see constructor comments regarding notations)
    def _calcDecFrom(self):
        
        self._DecFrom = dict()
        for i in xrange(1,self._theta.nStates):                                   
            t1 = (self._theta.lmbVals[i]
                  /-math.expm1(-self._theta.lmbVals[i]*self._theta.segments.delta[i])
                 )
            t2 = self._g(self._theta.lmbVals[i],
                         self._theta.r,
                         self._theta.segments.boundaries[i],
                         self._theta.segments.boundaries[i+1])
            
            self._DecFrom[i] = math.log(t1) + math.log(t2) + math.log(0.5)
            
    # calculate the coefficients DecTo (see constructor comments regarding notations)
    def _calcDecTo(self):
        
        self._DecTo = dict()
          
        for j in xrange(self._theta.nStates-1):
            
            res = self._S[j] - 0.5/self._theta.lmbVals[j]
            res *= -math.expm1(-2.0
                               *self._theta.lmbVals[j]
                               *self._theta.segments.delta[j])
            res += self._theta.segments.delta[j] 
            
            self._DecTo[j] = math.log(res)
    
    # calculate the coefficients IncTo (see constructor comments regarding notations)
    def _calcIncTo(self):
        self._IncTo = dict()
        for j in xrange(1,self._theta.nStates):
            t = -math.expm1(-self._theta.lmbVals[j]*self._theta.segments.delta[j])
            self._IncTo[j] = self._logL(0,j) + math.log(t)
    
    # calculate the coefficients DecTo (see constructor comments regarding notations)
    def _calcIncFrom(self):
        
        self._IncFrom = dict()
        
        for i in xrange(self._theta.nStates-1):
            
            t1 = (self._theta.lmbVals[i]
                  /-math.expm1(-self._theta.lmbVals[i]*self._theta.segments.delta[i])
                 )
            
            t2  = self._S[i]
            t2 *= self._g(2.0*self._theta.lmbVals[i],
                          self._theta.r,
                          self._theta.segments.boundaries[i],
                          self._theta.segments.boundaries[i+1])
            t2 += self._gStar(2*self._theta.lmbVals[i],
                              self._theta.r,
                              self._theta.segments.boundaries[i],
                              self._theta.segments.boundaries[i+1])
            
            self._IncFrom[i] = math.log(t1) + math.log(t2) - self._logL(0,i)
    
    # calculate log(p_ii) by completion to 1 (that is 1 - sum_(j != i)p_ij )
    def _calcDiagonal(self):
        
        self._diagonal = dict()
        
        # marginalInc_i = sum_(j>i)e^IncTo[j]
        # marginalDec_i = sum_(j<i)e^DecTo[j]
        marginalInc, marginalDec = dict(), dict()
        marginalDec[0] = 0.0
        for i in xrange(1,self._theta.nStates):
            marginalDec[i] = marginalDec[i-1] + math.exp(self._DecTo[i-1])
        marginalInc[self._theta.nStates - 1] = 0.0
        for i in xrange(self._theta.nStates - 2,-1,-1):
            marginalInc[i] = marginalInc[i+1] + math.exp(self._IncTo[i+1])
            
        # calculate diagonal by completion to one
        self._diagonal[0] = math.log(1.0 - math.exp(self._IncFrom[0])*marginalInc[0])
        for i in xrange(1, self._theta.nStates - 1):
            res = (1.0
                   -math.exp(self._IncFrom[i])*marginalInc[i]
                   -math.exp(self._DecFrom[i])*marginalDec[i]
                  )
            self._diagonal[i] = math.log(res)
        i = self._theta.nStates - 1
        self._diagonal[i] = math.log(1.0 - math.exp(self._DecFrom[i])*marginalDec[i])
        
        # sanity check: assert log(p_ij) <= 0 for all i > j
        M = self._DecTo[0]
        for i in xrange(1, self._theta.nStates):
            M = max(M, self._DecTo[i-1])
            assert (self._DecFrom[i] + M) <= 0.0
        
        # sanity check: assert log(p_ij) <= 0 for all i < j
        M = self._IncTo[self._theta.nStates - 1]
        for i in xrange(self._theta.nStates - 2,-1,-1):
            M = max(M, self._IncTo[i+1])
            assert (self._IncFrom[i] + M) <= 0.0
        
        # sanity check: assert log(p_ii) <= 0
        for i in xrange(self._theta.nStates):
            assert self._diagonal[i] <= 0.0
    
    # calculates stationary distribution
    def _calcStationary(self):
        
        self._stationaryProb = np.zeros( self._theta.nStates )
        
        for i in xrange(self._theta.nStates):
            
            p = self._L(0, i) * -math.expm1(-self._theta.lmbVals[i]*self._theta.segments.delta[i])
            self._stationaryProb[i] = p
        
        s = np.sum(self._stationaryProb)
        # room for numerical error
        assert abs(s - 1.0) < 10**-10
        self._stationaryProb /= s
        

    # Numerical approximation for g 
    def _g(self, lmb, r, t1, t2):
        
        assert lmb > 0
        assert r   > 0
        assert t1  >= 0
        assert t2  > t1

        res =  integrate.quad(lambda x: math.exp(-lmb*(x-t1))*-math.expm1(-2*r*x)/x,
                              t1,
                              t2,
                              #epsrel = 1.49e-7,
                              #epsabs = 0,
                              limit  = 150)
        
        # TODO Make sure that integration error is under control
        # assert res[1]/res[0] < .001
        return res[0]
    
    # Numerical approximation for g* 
    def _gStar(self, lmb, r, t1, t2):
        
        assert lmb > 0
        assert r   > 0
        assert t1  >= 0
        assert t2  > t1

        res =  integrate.quad(lambda x: math.expm1(-lmb*(x-t1))*math.expm1(-2*r*x)/(lmb*x),
                              t1,
                              t2,
                              #epsrel = 1.49e-7,
                              #epsabs = 0,
                              limit  = 150)
        
        # TODO Make sure that integration error is under control
        # assert res[1]/res[0] < .001
        return res[0]
    
    # return L*(T_i, T_j)
    def _lStar(self, i, j):
        return math.exp(2*self._logL(i, j))
    
    # return L(T_i, T_j)
    def _L(self, i, j):
        return math.exp(self._logL(i, j))
    
    # return log(L(T_i, T_j))
    def _logL(self, i, j):
        assert i <= j
        
        if (i,j) not in self._logLDict:
            if i == j:
                res = 0.0
            elif j == (i+1):
                res = -self._theta.lmbVals[i]*self._theta.segments.delta[i]
            else:
                res = self._logL(i,j-1) + self._logL(j-1,j)
            self._logLDict[i,j] = res
        return self._logLDict[i,j]
