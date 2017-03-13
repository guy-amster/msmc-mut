import numpy as np
import math
from Containers import Model, Theta

####### APPROXIMATION DOES NOT WORK.

# TODO replace all asserts with assertTrue etc

# TODO consider removing the class and make this file a module with one public function, returning a Matrix 

# transitionProbs: This class calculates the transition probabilities of the model.
class TransitionProbs(object):
    
    def __init__(self, model, theta):
        
        # TODO save only something
        self._model = model
        # TODO save only lambda
        self._theta = theta
        
        # Notations:
        # The transition probability p_ij from state i to state j could be decomposed as follows:
        # - If i<j, ('increasing' transition), log(p_ij) = IncFrom_i + IncTo_j.
        # - If i>j, ('decreasing' transition), log(p_ij) = DecFrom_i + DecTo_j.
        # The transitions p_ii are calculated indirectly by completion to one ( p_ii = 1 - sum_{j!=i}p_ij ).
        
        # Calculate the coefficiants IncFrom, IncTo, DecFrom, DecTo (and other temp. variables)
        self._logLDict = dict()
        self._calcS()
        self._calcIncFrom()
        self._calcDecFrom()
        self._calcIncTo()
        self._calcDecTo()
        self._calcDiagonal()
    
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
    
    # Calculate the empirical log-likelihood of an observed sequence.
    # observed is a Containers.SequenceSummary class
    def logLikelihood(self, observed):
        
        res = 0.0
        
        # likelihood for loops
        for i in xrange(self._model.nStates):
            res += observed.loops[i]*self._diagonal[i]
        
        for i in xrange(self._model.nStates - 1):
            res += observed.incFrom[i]*self._IncFrom[i]
            res += observed.decTo[i]*self._DecTo[i]
        
        for i in xrange(1, self._model.nStates):
            res += observed.incTo[i]*self._IncTo[i]
            res += observed.decFrom[i]*self._DecFrom[i]
        
        return res            
        
    
    # S_j = sum_(k=0,..,j-1) (1-e^-2*lmb_k*delta_k)*lStar(T_(k+1),T_j)/(2*lmb_k)
    def _calcS(self):

        self._S = dict()        
        S = 0.0
        for j in xrange(self._model.nStates-1):
            
            # Update S
            if j > 0:
                S *= self._lStar(j-1,j)
                S += (-math.expm1(-2.0*self._theta.lambdaV[j-1]*self._model.segments.delta[j-1])
                      /(2.0*self._theta.lambdaV[j-1])
                     )
            self._S[j] = S
    
    # calculate the coefficients DecFrom (see constructor comments regarding notations)
    def _calcDecFrom(self):
        
        self._DecFrom = dict()
        for i in xrange(1,self._model.nStates):                                   
            self._DecFrom[i] = math.log(self._theta.r)
            
    # calculate the coefficients DecTo (see constructor comments regarding notations)
    def _calcDecTo(self):
        
        self._DecTo = dict()
          
        for j in xrange(self._model.nStates-1):
            
            res = self._S[j] - 0.5/self._theta.lambdaV[j]
            res *= -math.expm1(-2.0
                               *self._theta.lambdaV[j]
                               *self._model.segments.delta[j])
            res += self._model.segments.delta[j] 
            
            self._DecTo[j] = math.log(res)
    
    # calculate the coefficients IncTo (see constructor comments regarding notations)
    def _calcIncTo(self):
        self._IncTo = dict()
        for j in xrange(1,self._model.nStates):
            t = -math.expm1(-self._theta.lambdaV[j]*self._model.segments.delta[j])
            self._IncTo[j] = self._logL(0,j) + math.log(t) + math.log(self._theta.r)
    
    # calculate the coefficients DecTo (see constructor comments regarding notations)
    def _calcIncFrom(self):
        
        self._IncFrom = dict()
        
        for i in xrange(self._model.nStates-1):
            
            t1 = -math.expm1(-self._theta.lambdaV[i]*self._model.segments.delta[i])
                 
            t2 = self._S[i] - 0.5/self._theta.lambdaV[i]
            t2 *= -math.expm1(-2.0
                               *self._theta.lambdaV[i]
                               *self._model.segments.delta[i])
            t2 += self._model.segments.delta[i] 
            
            self._IncFrom[i] =  math.log(t2) - self._logL(0,i) - math.log(t1)
    
    # calculate log(p_ii) by completion to 1 (that is 1 - sum_(j != i)p_ij )
    def _calcDiagonal(self):
        
        self._diagonal = dict()
        
        # marginalInc_i = sum_(j>i)e^IncTo[j]
        # marginalDec_i = sum_(j<i)e^DecTo[j]
        marginalInc, marginalDec = dict(), dict()
        marginalDec[0] = 0.0
        for i in xrange(1,self._model.nStates):
            marginalDec[i] = marginalDec[i-1] + math.exp(self._DecTo[i-1])
        marginalInc[self._model.nStates - 1] = 0.0
        for i in xrange(self._model.nStates - 2,-1,-1):
            marginalInc[i] = marginalInc[i+1] + math.exp(self._IncTo[i+1])
            
        # calculate diagonal by completion to one
        self._diagonal[0] = math.log(1.0 - math.exp(self._IncFrom[0])*marginalInc[0])
        for i in xrange(1, self._model.nStates - 1):
            res = (1.0
                   -math.exp(self._IncFrom[i])*marginalInc[i]
                   -math.exp(self._DecFrom[i])*marginalDec[i]
                  )
            print res
            self._diagonal[i] = math.log(res)
        i = self._model.nStates - 1
        self._diagonal[i] = math.log(1.0 - math.exp(self._DecFrom[i])*marginalDec[i])
        
        # sanity check: assert log(p_ij) <= 0 for all i > j
        M = self._DecTo[0]
        for i in xrange(1, self._model.nStates):
            M = max(M, self._DecTo[i-1])
            assert (self._DecFrom[i] + M) <= 0.0
        
        # sanity check: assert log(p_ij) <= 0 for all i < j
        M = self._IncTo[self._model.nStates - 1]
        for i in xrange(self._model.nStates - 2,-1,-1):
            M = max(M, self._IncTo[i+1])
            assert (self._IncFrom[i] + M) <= 0.0
        
        # sanity check: assert log(p_ii) <= 0
        for i in xrange(self._model.nStates):
            assert self._diagonal[i] <= 0.0

  
    
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
                res = -self._theta.lambdaV[i]*self._model.segments.delta[i]
            else:
                res = self._logL(i,j-1) + self._logL(j-1,j)
            self._logLDict[i,j] = res
        return self._logLDict[i,j]
