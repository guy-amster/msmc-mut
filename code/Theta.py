from BasicHMM import HmmTheta
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
import re

# TODO explain
# theta: a container class for the non-fixed parameters of the model.
class Theta(HmmTheta):
    
    
    # TODO model   : a Model object, containing the fixed parameters of the model.
    # r       : scaled (TODO explain!) recombination rate.
    # uV      : a vector of the scaled (TODO explain!) mutation rates in the different time-segments defined by the model.
    # lambdaV : a vector of the coalescence (TODO explain!) rates in the different time-segments defined by the model.
    # Parameters that are omitted (None) are set to the default values specified in defVals.
    def __init__(self, model, r=None, lambdaV=None, uV=None):
        
        # Scaled recombination rate r.
        if r is None:
            self.r = model.defVals.r
        else:
            self.r = r
        
        # Piecewise coalescense rates.
        if lambdaV is None:
            self.lambdaV = [model.defVals.lmb for _ in xrange(model.segments.n)]
        else:
            assert len(lambdaV) == model.segments.n
            self.lambdaV = lambdaV
        
        # Piecewise mutation rates.
        if uV is None:
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
