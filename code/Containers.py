import math
import random
import numpy as np
from collections import namedtuple
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
from Logger import log

# default parameter values for r, u & lambda
DefValues = namedtuple('DefValues', 'r u lmb')

# piecewise: A container class specifying fixed parameters for a piecewise function on [0,infinity):
#            The number of segments, their sizes and boundaries.
#            The values of the function are not specified here.
class piecewise(object):
    
    # TODO don't take pi like that... 
    # n: number of segments.
    def __init__(self, n, pi):
        
        # number of segments
        assert n > 0
        self.n = n
        
        # Segments boundaties
        # Default is based on logarithmic quantiles; e.g. -log(1-i/n) for i=0,...,n where log(0) defined as -inf
        # self.boundaries = [-math.log1p(-1.0*i/self.n) for i in xrange(self.n)] + [np.inf]
        self.boundaries  = [-0.5*pi*math.log1p(-1.0*i/40.0) for i in xrange(11)]
        self.boundaries += [-0.5*pi*math.log1p(-1.0*i/40.0) for i in xrange(12,40,2)]
        self.boundaries += [np.inf]
        assert len(self.boundaries) == (n+1)
        log('intervals: ' + ','.join([str(x) for x in self.boundaries]))
        
        # Size of segments (notice: last segment size is np.inf)
        self.delta = np.diff(self.boundaries)
    
    # Find in which bin the input t falls, and return the bin's index.
    # Bins are indexed 0,1,...,n-1.
    def bin(self, t):
        assert (0.0 <= t and t < np.inf)
        
        ind = int(np.digitize(t, self.boundaries)) - 1
        assert (0 <= ind and ind < self.n)
        assert (self.boundaries[ind] <= t and t < self.boundaries[ind+1])
        return ind

# A container class specifying the fixed parameters of a HMM model
class HmmModel(object):
    
    # nStates: number of states in the model.
    # nEmissions: number of possible observations in the model (not chain length)
    def __init__(self, nStates, nEmissions):
        self.modelType  = 'basic'
        self.nStates    = nStates
        self.nEmissions = nEmissions

# TODO for M>2 add a state container with members (index,timeInd,branches); Let model make the conversions index->state, (timeInd,branches)->state

# model: A container class specifying all the fixed parameters of the model.
# TODO change name...
class Model(HmmModel):
    
    # fixedR      = True: assumes r (the recombination rate) is a fixed constant.
    # fixedLambda = true: assumes lambda(t) is a constant (independent of time).
    # fixedMu     = True: assumes u(t) is a constant (independent of time).
    # TODO constants values are specified in TODO.
    # TODO fixed here should be expanded to assume specific cons. / assume unknown constant (no change with time)
    # TODO also model could accept defVals as input....
    # TODO don't take pi like that - that's weird
    def __init__(self, pi, fixedR=True, fixedLambda=False, fixedMu=False):
        self.modelType  = 'full'
        
        # Fixed paramaters for the segments defining the discrete hmm states and underlying the piecewise functions lambda(t) & u(t).
        # TODO are the same segments really ideal for both the inference of u & lambda (in terms of power)?
        self.segments   = piecewise(25, pi)

        self.nStates    = self.segments.n
        self.nEmissions = 2
        
        self.fixedR      = fixedR
        self.fixedLambda = fixedLambda
        self.fixedMu     = fixedMu
        
        # number of unconstrained parameters of the model (also see Theta.fromFreeParams )
        self.nFreeParams = 0
        if not fixedR:
            self.nFreeParams += 1
        if not fixedLambda:
            self.nFreeParams += self.segments.n
        if not fixedMu:
            self.nFreeParams += self.segments.n
        
        # TODO change vals - rho should be 1...
        
        self.defVals = DefValues(.25, 1.0, (2.0/pi))

# HmmTheta: a container class for the non-fixed parameters of an HmmModel.
class HmmTheta(object):
    
    # model         : A HmmModel object specifying the fixed parameters of the model.
    # transitionMat : The transition mtrix of the chain.
    # initialDit    : The distribution of the first state.
    # emissionMat   : The emission probabilities matrix.
    def __init__(self, model, transitionMat, initialDist, emissionMat):
        
        assert transitionMat.shape == (model.nStates, model.nStates   )
        assert initialDist  .shape == (model.nStates,                 )
        assert emissionMat  .shape == (model.nStates, model.nEmissions)
        
        self._transitionMat = transitionMat
        self._initialDist   = initialDist
        self._emissionMat   = emissionMat
    
    # return the emission probabilities matrix.
    def emissionMat(self):
        return self._emissionMat
    
    # return the initial distribution of the chain, and the transition probabilities matrix.
    def chainDist(self):
        return self._initialDist, self._transitionMat
    
    # Initialize theta with random values
    # (all random probability vectors drawn from Dirichlet(1,...,1))
    @classmethod
    def random(cls, model):
        initialDist   = np.random.dirichlet(np.ones(model.nStates))
        transitionMat = np.empty( (model.nStates, model.nStates   ) )
        emissionMat   = np.empty( (model.nStates, model.nEmissions) )
        
        for i in xrange(model.nStates):
            transitionMat[i,:] = np.random.dirichlet(np.ones(model.nStates   ))
            emissionMat  [i,:] = np.random.dirichlet(np.ones(model.nEmissions))

        return cls(model, transitionMat, initialDist, emissionMat)


# theta: a container class for the non-fixed parameters of the model.
class Theta(HmmTheta):
    
    # model: a msmcMarkovModel object, containing the fixed parameters of the model.
    # r       : scaled (TODO explain!) recombination rate.
    # uV      : a vector of the scaled (TODO explain!) mutation rates in the different time-segments defined by the model.
    # lambdaV : a vector of the coalescence (TODO explain!) rates in the different time-segments defined by the model.
    # Parameters that are omitted (None) are set to the default values specified in defVals.
    def __init__(self, model, r=None, lambdaV=None, uV=None):
        
        # Scaled recombination rate r.
        if r == None:
            self.r = model.defVals.r
        else:
            self.r = r
        
        # Piecewise coalescense rates.
        if lambdaV == None:
            self.lambdaV = [model.defVals.lmb for _ in xrange(model.segments.n)]
        else:
            assert len(lambdaV) == model.segments.n
            self.lambdaV = lambdaV
        
        # Piecewise mutation rates.
        if uV == None:
            self.uV = [model.defVals.u for _ in xrange(model.segments.n)]
        else:
            assert len(uV) == model.segments.n
            self.uV = uV
        
        self._model = model
    
    # return the emission probabilities matrix.
    def emissionMat(self):
        
        return EmissionProbs(self._model,self).emissionMat()
    
    # return the initial distribution of the chain, and the transition probabilities matrix.
    def chainDist(self):
        
        transitionProbs = TransitionProbs(self._model,self)
        return transitionProbs.stationaryProb(), transitionProbs.transitionMat()

    @classmethod
    # expand a vector of unconstrained parameters to a Theta instance.
    # vec = log(r) || log(lambda_0), ..., log(lambda_n-1) || log(u_0), ..., log(u_n-1).
    # entries that are fixed in the model are ommited from the vector.
    def fromUnconstrainedVec(cls, model, vec):
        
        assert len(vec) == model.nFreeParams
                        
        index           = 0
        r, lambdaV, uV  = None, None, None
        
        # read recombination rate
        if not model.fixedR:
            r = math.exp(vec[0])
            index += 1
        
        # read lambda
        if not model.fixedLambda:
            lambdaV = []
            for i in xrange(index, index + model.segments.n ):
                lambdaV.append(math.exp(vec[i]))
            index  += model.segments.n
        
        # read u
        if not model.fixedMu:
            uV = []
            for i in xrange(index, index + model.segments.n):
                uV.append(math.exp(vec[i]))
            index += model.segments.n
        
        assert index == len(vec)
                                
        return cls(model, r=r, lambdaV=lambdaV, uV=uV)
        
    # Summarize the instance as a vector of unconstrained parameters (format defined above).
    def toUnconstrainedVec(self):
        res = []
        
        if not self._model.fixedR:
            res.append(math.log(self.r))
        
        if not self._model.fixedLambda:
            for i in xrange(0,len(self.lambdaV)):
                res.append(math.log(self.lambdaV[i]))
        
        if not self._model.fixedMu:
            for i in xrange(0,len(self.uV)):
                res.append(math.log(self.uV[i]))
        
        assert len(res) == self._model.nFreeParams
        
        return res
    
    # Initialize theta with random values
    # TODO describe better as not all are random.
    @classmethod
    def random(cls, model):
        
        r, lambdaV, uV  = None, None, None
        
        if not model.fixedR:
            r = random.uniform(0.5*model.defVals.r, 2*model.defVals.r)
        
        if not model.fixedLambda:
            lambdaV = [model.defVals.lmb]
            for i in xrange(1, model.segments.n):
                lambdaV.append(random.uniform(0.125*model.defVals.lmb, 8*model.defVals.lmb))
        
        if not model.fixedMu:
            uV = [model.defVals.u]
            for i in xrange(1, model.segments.n):
                uV.append(random.uniform(0.5*model.defVals.u, 2.0*model.defVals.u))
        
        return cls(model, r=r, lambdaV=lambdaV, uV=uV)
    
    # TODO Do I need this?
    def printVals(self):
        print 'Lambda: ', self.lambdaV
        print 'U:      ', self.uV
        print 'r:      ', self.r
        
    