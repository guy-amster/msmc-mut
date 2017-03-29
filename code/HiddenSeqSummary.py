import random
import math
import numpy as np
from scipy.optimize import minimize
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
from Containers import Theta, HmmTheta
from multiprocessing import Pool

# TODO DOC
# defined at the module level to allow calling from pool map
def _maxQSingleStartPoint(inpDict):

    hiddenSeqSum = inpDict['hiddenSeqSum']
    x0           = inpDict['x0']
    if x0 is None:
        x0 = Theta.random(hiddenSeqSum._model).toUnconstrainedVec()
        
    
    defVals = Theta(hiddenSeqSum._model).toUnconstrainedVec()
    
            
    # TODO DOC
    QNull = hiddenSeqSum.Q(Theta(hiddenSeqSum._model))*2
    logTen = math.log(10)
    def fun(x):
        for i in xrange(hiddenSeqSum._model.nFreeParams):
            z = abs(x[i] - defVals[i])
            if z >= logTen:
                return -QNull
        return -hiddenSeqSum.Q(Theta.fromUnconstrainedVec(hiddenSeqSum._model, x))
    
    consts = [{'type': 'ineq', 'fun': lambda x:  math.log(10) + (x[i] - defVals[i])} for i in range(len(defVals))] \
            +[{'type': 'ineq', 'fun': lambda x:  math.log(10) - (x[i] - defVals[i])} for i in range(len(defVals))]

    op = minimize(fun,
                  x0,
                  #constraints=tuple(consts),
                  tol=1e-7,
                  #options={'disp': True, 'maxiter': 1000000}
                  options={'maxiter': 1000000}
                  )

    return Theta.fromUnconstrainedVec(hiddenSeqSum._model, op.x)

# Summary statistics (NOT entire sequence) on hidden-state sequence
# seqLength     : Underlying suequence length.
# transitions   : ndarray. transitions[i,j] is the inferred number of transitions i->j
# emissions     : ndarray. emissions[i,j] is the inferred number of emissions i->j
# logLikelihood : the log-l of the hidden & observed sequence.
# gamma0        : the posterior distribution of states at the beginning of the sequence
class HiddenSeqSummary(object):
    
    def __init__(self, model, seqLength, transitions, emissions, gamma0, logLikelihood):
        
        self._model = model
        
        self.length = seqLength
        
        assert transitions.shape == (model.nStates, model.nStates)
        assert emissions.shape   == (model.nStates, model.nEmissions)
        assert gamma0.shape      == (model.nStates, )
        
        # log( P(O|theta) ), where theta are the parameters used for the hidden-state inference
        # (ie, theta are the parameters used for the Baum-Welch expectation step)
        self.logL = logLikelihood
        
        self.gamma0      = gamma0
        self.emissions   = emissions
        self.transitions = transitions 
        
        # IncFrom[i] is the proportion of transitions i->j for some j>i
        self.incFrom = np.array([np.sum(self.transitions[i,(i+1):]) for i in xrange(model.nStates)])
        
        # DecFrom[i] is the proportion of transitions i->j for some j<i
        self.decFrom = np.array([np.sum(self.transitions[i,0:i]) for i in xrange(model.nStates)])
        
        #   IncTo[j] is the proportion of transitions i->j for some i<j
        self.incTo   = np.array([np.sum(self.transitions[0:j,j]) for j in xrange(model.nStates)])
        
        #   DecTo[j] is the proportion of transitions i->j for some i>j
        self.decTo   = np.array([np.sum(self.transitions[(j+1):,j]) for j in xrange(model.nStates)])
            
    # Combine two classes to one
    def __add__(self, other):
        
        assert self.transitions.shape == other.transitions.shape
        assert self.emissions.shape   == other.emissions.shape
        
        length          = self.length      + other.length
        transitions     = self.transitions + other.transitions
        emissions       = self.emissions   + other.emissions
        gammaa0         = self.gamma0      + other.gamma0
        logL            = self.logL        + other.logL
        
        return HiddenSeqSummary(self._model, length, transitions, emissions, gammaa0, logL)
    
    # calculate Q(theta* | theta) (ie evaluation for EM maximization step)
    # where theta are the parameters that were used for the creation of this HiddenSeqSummary class
    # Q = E( log( P(hidden-state sequence Z, observations O | theta* ) ) ), where
    #     the expactation is over the posterior distribution of Z conditioned on theta (ie ~ P(Z | O, theta) )
    # This is all just standard EM stuff.
    # TODO doc is this Q per-bp? what have I done?
    def Q(self, thetaStar):
        
        # standard HMM 
        if self._model.modelType == 'basic':
            initialDist, transitionMat = thetaStar.chainDist  ()
            emissionMat                = thetaStar.emissionMat()
            res  = np.sum(np.log(transitionMat) * self.transitions)
            res += np.sum(np.log(emissionMat  ) * self.emissions  )
            res += np.sum(np.log(initialDist  ) * self.gamma0     )
            return res
        
        # our model
        # TODO benchmark to see if this is even faster than the general case above...
        # if not: remove likelihoods from these classes; remove incfrom etc; remove this
        else:
            res  = EmissionProbs  (self._model, thetaStar).logLikelihood(self)
            res += TransitionProbs(self._model, thetaStar).logLikelihood(self)
        
            return res
    
    # calculate theta* that maximizes Q (ie EM maximization step).
    # returns theta* and the attained maximum value.
    def maximizeQ(self, nProcesses = 1, nStartPoints = 120, initTheta = None):
        
        # we might as well use all available resources
        nStartPoints = max(nStartPoints, nProcesses)
        
        # in the case of a standard HMM, it's not necessary to evaluate Q, as there's a closed form global maximum
        if self._model.modelType == 'basic':
            trans = np.empty( (self._model.nStates, self._model.nStates) )
            for i in xrange(self._model.nStates):
                trans[i,:] = self.transitions[i, :] / np.sum(self.transitions[i, :])
        
            emiss = np.empty( (self._model.nStates, self._model.nEmissions) )
            for i in xrange(self._model.nStates):
                emiss[i,:] = self.emissions[i, :] / np.sum(self.emissions[i, :])
            
            maxTheta = HmmTheta(self._model, trans, self.gamma0, emiss)
            maxFound = self.Q(maxTheta)
        
        # in our case (the model constrains the matrices & initial distribution), we need to numerically find a (hopefully global) maximum
        else:
            inputs = [{'hiddenSeqSum': self, 'x0':None} for _ in xrange(nStartPoints)]
            if initTheta is not None:
                inputs.append({'hiddenSeqSum': self, 'x0':initTheta.toUnconstrainedVec()})

            if nProcesses > 1:
                p   = Pool(nProcesses)
                res = p.map(_maxQSingleStartPoint, inputs)       
                p.close()
            else:
                res = []
                for inp in inputs:
                    res.append(_maxQSingleStartPoint(inp))
                
            
            maxFound = -np.inf
            for theta in res:
                if self.Q(theta) > maxFound:
                    maxFound = self.Q(theta)
                    maxTheta = theta
        
        return maxTheta, maxFound
        
   