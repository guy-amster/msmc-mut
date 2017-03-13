import random
import math
import numpy as np
from scipy.optimize import basinhopping, minimize
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
from Containers import Theta, HmmTheta


# Summary statistics (NOT entire sequence) on hidden-state sequence
# seqLength     : Underlying suequence length.
# transitions   : ndarray. transitions[i,j] is the proportion, or absolute number, of transitions i->j
# emissions     : ndarray. emissions[i,j] is the proportion, or absolute number, of emissions i->j (where 0 stands for hom. and 1 for het)
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
        
        # number of observed sequences (e.g. different chromosomes) being summarized
        self.nSequences = 1
        
        #TODO how to add these?
        self.gamma0 = gamma0
        
        # emissions[i,j] is the  proportion of emissions i->j (i a state, j an observed output type)
        # i.e., the entire emissions matrix sums to 1
        self.emissions   = emissions   #TODO REMOVE/ float(np.sum(emissions))
        
        # transitions[i,j] is the proportion of transitions i->j
        # i.e., the entire transitions matrix sums to 1
        self.transitions = transitions #TODO REMOVE / float(np.sum(transitions))
        
        # IncFrom[i] is the proportion of transitions i->j for some j>i
        self.incFrom = np.array([np.sum(self.transitions[i,(i+1):]) for i in xrange(model.nStates)])
        
        # DecFrom[i] is the proportion of transitions i->j for some j<i
        self.decFrom = np.array([np.sum(self.transitions[i,0:i]) for i in xrange(model.nStates)])
        
        #   IncTo[j] is the proportion of transitions i->j for some i<j
        self.incTo   = np.array([np.sum(self.transitions[0:j,j]) for j in xrange(model.nStates)])
        
        #   DecTo[j] is the proportion of transitions i->j for some i>j
        self.decTo   = np.array([np.sum(self.transitions[(j+1):,j]) for j in xrange(model.nStates)])
            
    # Weighted average of 2 classes, weighted by sequence length.
    # (i.e., the transition and emission matrices in the sum is the proportion in the 2 sequences combined)
    def __add__(self, other):
        
        assert self.transitions.shape == other.transitions.shape
        assert self.emissions.shape   == other.emissions.shape
        
        l1, l2          = float(self.length), float(other.length)
        w1              = l1/(l1+l2)
        w2              = 1.0 - w1
        transitions     = (w1 * self.transitions) + (w2 * other.transitions)
        emissions       = (w1 * self.emissions  ) + (w2 * other.emissions  )
        logL            = self.logL + other.logL
        
        res             = HiddenSeqSummary(self._model, self.length + other.length, transitions, emissions, todogamma, logL)
        res.nSequences  = self.nSequences + other.nSequences 
        return res
    
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
            res  = (self.length - 1)  * np.sum(np.log(transitionMat) * self.transitions)
            res += self.length        * np.sum(np.log(emissionMat  ) * self.emissions  )
            res +=                      np.sum(np.log(initialDist  ) * self.gamma0     )
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
    def maximizeQ(self):
        
        # in the case of a standard HMM, it's not necessary to evaluate Q, as there's a closed form global maximum
        if self._model.modelType == 'basic':
            trans = np.empty( (self._model.nStates, self._model.nStates) )
            for i in xrange(self._model.nStates):
                trans[i,:] = self.transitions[i, :] / np.sum(self.transitions[i, :])
        
            emiss = np.empty( (self._model.nStates, self._model.nEmissions) )
            for i in xrange(self._model.nStates):
                emiss[i,:] = self.emissions[i, :] / np.sum(self.emissions[i, :])
            
            thetaRes = HmmTheta(self._model, trans, self.gamma0, emiss)

        
        # in our case (the model constrains the matrices & initial distribution), we need to numerically find a (hopefully global) maximum
        else:
            maxFound = -np.inf
            for _ in xrange(10):
                defVals = Theta(self._model).toUnconstrainedVec()
                x0 = Theta.random(self._model).toUnconstrainedVec()
                
                # TODO REMOVE OR DOC
                # fun = lambda x: -self.Q(Theta.fromUnconstrainedVec(self._model, x))
                
                # TODO REMOVE OR DOC
                QNull = self.Q(Theta(self._model))*2
                logTen = math.log(10)
                def fun(x):
                    for i in xrange(self._model.nFreeParams):
                        z = abs(x[i] - defVals[i])
                        if z >= logTen:
                            return -QNull
                    return -self.Q(Theta.fromUnconstrainedVec(self._model, x))
                
                consts = [{'type': 'ineq', 'fun': lambda x:  math.log(10) + (x[i] - defVals[i])} for i in range(len(defVals))] \
                        +[{'type': 'ineq', 'fun': lambda x:  math.log(10) - (x[i] - defVals[i])} for i in range(len(defVals))]
                
                op = minimize(fun,
                              x0,
                              #constraints=tuple(consts),
                              tol=1e-7,
                              #options={'disp': True, 'maxiter': 1000000}
                              options={'maxiter': 1000000}
                              )
                
                # print 'opt: ', op.message
                
                # TODO print op.success
                
                thetaRes = Theta.fromUnconstrainedVec(self._model, op.x)
                if self.Q(thetaRes) > maxFound:
                    maxFound = self.Q(thetaRes)
                    maxTheta = thetaRes
        
        return maxTheta, maxFound
        
   