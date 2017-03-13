import random
import math
import numpy as np
from scipy.optimize import basinhopping, minimize
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
from Containers import FreeParams, HmmTheta


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
        
        # TODO REMOVE float64 from project
        assert transitions.dtype == 'float64'
        assert emissions.dtype   == 'float64'
        
        # log( P(O|theta) ), where theta are the parameters used for the hidden-state inference
        # (ie, theta are the parameters used for the Baum-Welch expectation step)
        self.logL = logLikelihood
        
        # number of observed sequences (e.g. different chromosomes) being summarized
        self.nSequences = 1
        
        #TODO how to add these?
        self.gamma0 = gamma0
        
        # emissions[i,j] is the  proportion of emissions i->j (i a state, j an observed output type)
        # i.e., the entire emissions matrix sums to 1
        self.emissions   = emissions   / float(np.sum(emissions))
        
        # transitions[i,j] is the proportion of transitions i->j
        # i.e., the entire transitions matrix sums to 1
        self.transitions = transitions / float(np.sum(transitions))
        
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
    
    # calculate Q(theta* | Q) (ie evaluation for EM maximization step)
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

        
        # in our case (the model constraints the matrices & initial distribution), we need to numerically find a local (hopefully global) maximum
        else:
            freeParams = FreeParams(self._model)
            defVals = freeParams.defVals()
            x0 = [-random.expovariate(2) for _ in xrange(freeParams.nFreeParams)]
            fun = lambda x: -self.logLikelihood(freeParams.theta(x))
            
            consts = [{'type': 'ineq', 'fun': lambda x:  math.log(10) + (x[i] - defVals[i])} for i in range(len(defVals))] \
                    +[{'type': 'ineq', 'fun': lambda x:  math.log(10) - (x[i] - defVals[i])} for i in range(len(defVals))]
                    
            op = minimize(fun,
                          x0,
                          constraints=tuple(consts),
                          tol=1e-40,
                          options={'disp': True, 'maxiter': 1000000}
                          )
            
            print 'opt: ', op.message
            # TODO print op.success
            
            thetaRes = freeParams.theta(op.x)
        
        return thetaRes, self.Q(thetaRes)
        
        
'''
# bounds for optimization
# TODO REMOVE OR MOVE
class MyBounds(object):
    def __init__(self, xmax, xmin):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

# Summary statistics on an observed or inferred sequence of the hmm
class SequenceSummary(object):
    
    def __init__(self, model):
        
        self._model = model
        
        # loops_i is the number of transitions i->i
        self.loops        = [0 for _ in xrange(model.nStates)]
        
        # incFrom_i is the total number of transitions i->j where j>i
        self.incFrom      = [0 for _ in xrange(model.nStates)]
        
        # incTo_j is the total number of transitions i->j where i<j 
        self.incTo        = [0 for _ in xrange(model.nStates)]
         
        # decFrom_i is the total number of transitions i->j where j<i  
        self.decFrom      = [0 for _ in xrange(model.nStates)]
        
        # decTo_j is the total number of transitions i->j where i>j 
        self.decTo        = [0 for _ in xrange(model.nStates)]
        
        # homEmissions_i is the total number of homozygous emmisions from state i
        self.homEmissions = [0 for _ in xrange(model.nStates)]
         
        # hetEmissions_i is the total number of heterozygous emmisions from state i 
        self.hetEmissions = [0 for _ in xrange(model.nStates)]
        
    
    # Calculate the log-likelihood of the sequence, conditioned on the parameters defined by theta.
    def logLikelihood(self, theta):
        
        res  = EmissionProbs(self._model, theta).logLikelihood(self)
        res += TransitionProbs(self._model, theta).logLikelihood(self)
        
        return res
    
    # Normalize values by sequence length.
    # TODO do I need this function
    def normalize(self):
        length = 1.0/(math.fsum(self.homEmissions) + math.fsum(self.hetEmissions))
        
        self.loops        = [x*length for x in self.loops]
        self.incFrom      = [x*length for x in self.incFrom]
        self.incTo        = [x*length for x in self.incTo]
        self.decFrom      = [x*length for x in self.decFrom]
        self.decTo        = [x*length for x in self.decTo]
        self.homEmissions = [x*length for x in self.homEmissions]
        self.hetEmissions = [x*length for x in self.hetEmissions]
    
    # Return a set of parameters theta that maximizes the log-likelihood of a specifiv hidden-states sequence (i.e. Baum-Welch maximization step).
    def maximizeLogLikelihood(self, method='basinhopping'):
        
            
        freeParams = FreeParams(self._model)
        defVals = freeParams.defVals()
        x0 = [-random.expovariate(2) for _ in xrange(freeParams.nFreeParams)]
        # x0 = freeParams.defVals()
        fun = lambda x: -self.logLikelihood(freeParams.theta(x))
        def constrainedLikelihood(x):
            if mybounds(x_new=x):
                return fun(x)
            else:
                # TODO
                return 9999999999999999999999999999999999999999999999999999   
        
        if method == 'basinhopping':
            # TODO doc or remove
            minBounds = [x-math.log(10) for x in defVals]
            maxBounds = [x+math.log(10) for x in defVals]
            mybounds  = MyBounds(minBounds, maxBounds)
            
            op = basinhopping(lambda x: constrainedLikelihood(x),
                              x0,
                              niter = 3000,
                              niter_success = 300,
                              interval = 20,
                              stepsize = 5.0,
                              disp = False,
                              accept_test = mybounds)
                        
        else:
            
            if method in ['L-BFGS-B', 'TNC', 'SLSQP' ]:
                # use bounds
                op = minimize(fun, x0, bounds=[(x-math.log(10),x+math.log(10)) for x in defVals])
                
            elif method == 'COBYLA':
                # use constraints
                consts = [{'type': 'ineq', 'fun': lambda x:  math.log(10) + (x[i] - defVals[i])} for i in range(len(defVals))] \
                        +[{'type': 'ineq', 'fun': lambda x:  math.log(10) - (x[i] - defVals[i])} for i in range(len(defVals))]
                
                op = minimize(fun,
                              x0,
                              constraints=tuple(consts),
                              tol=1e-34,
                              options={'disp': True, 'maxiter': 100000}
                             )
            else:
                # optimization without bounds
                op = minimize(fun, x0)
            
        
        print 'opt: ', op.message
        # TODO print op.success
        
        return freeParams.theta(op.x)
'''        
        