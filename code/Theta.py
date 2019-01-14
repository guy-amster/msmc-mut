import re
import math
import numpy as np
from BaumWelch import HMM
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
from SiteTypes import siteTypes


# Segments: A container class specifying fixed parameters for a piecewise function on [0,infinity):
#           The number of segments, their sizes and boundaries.
#           The values of the function are not specified by this class.
class Segments(object):
    
    # boundaries: segments boundaries (eg, [0,1.0,2.3,inf]).
    def __init__(self, boundaries):
        
        # number of segments
        self.n = len(boundaries) - 1
        
        # assert input values are valid
        assert self.n > 0
        assert boundaries[0] == 0
        assert np.isinf(boundaries[-1])
        for i in xrange(self.n):
            assert boundaries[i] < boundaries[i+1]
        
        self.boundaries = boundaries
        
        # Size of segments (notice: last segment size is np.inf)
        self.delta = np.diff(self.boundaries)
    
    # Find in which bin the input t falls, and return the bin's index.
    # Bins are indexed as 0,1,...,n-1.
    def bin(self, t):
        
        assert 0.0 <= t < np.inf
        ind = int(np.digitize(t, self.boundaries)) - 1
        assert self.boundaries[ind] <= t < self.boundaries[ind+1]
        return ind


# Theta: A container class specifying parameters for the coalescence and mutational processes:
#        piecewise coalescence rates, piecewise mutation rates and recombination rate.
#        These parametrs define a hidden Markov model (the states corresponding to the segments defined by boundaries)
class Theta(HMM):
    
    # boundaries : Boundaries for the segments defining u(t) and lambda(t) as piecewise functions, e.g. [0, 1.0, 3.0, ..., inf].
    # TODO it's possible to specify different boundaries for u and lambda here
    # lmbVals    : Coalescence rate (lambda) values, per unit-time.
    # uVals      : Mutation rate values per unit-time.
    # r          : Recombination rate values per unit-time.
    def __init__(self, boundaries, lmbVals, uVals, r):
        
        # Initialize Segmnets instance
        self.segments = Segments(boundaries)
        assert self.segments.n == len(uVals) == len(lmbVals)
        
        # read parameter values
        self.lmbVals  = lmbVals
        self.uVals    = uVals
        self.r        = r
        
        # define HMM properties:
        self.nStates, self.nEmissions = self.segments.n, 2
        
        # for numerical reasons; it might be best to scale before initializing transitionProbs, emissionMat
        # (note that the BW model scales thetas such that this scaling here is not invoked in practice;
        #  however, eg when reading a hist file with r, u ~ 10^-8, one needs to scale first)
        if self.r < 0.00001:
            scaledTheta   = self.scale('r', 1.0)
            emissionMat   = scaledTheta.emissionMat
            transitionMat = scaledTheta.transitionMat
            initialDist   = scaledTheta.initialDist
            
        # if scaling is not required, calculate directly
        else:
            
            # TODO this should be a module not a class obviously
            emissionMat = EmissionProbs(self).emissionMat()
            # TODO check for loss of performance... decide what to do & change TransProbs accordingly
            transitionProbs = TransitionProbs(self)
            transitionMat, initialDist = transitionProbs.transitionMat(), transitionProbs.stationaryProb()
            
        HMM.__init__(self, self.segments.n, 2, transitionMat, initialDist, emissionMat)
    
    # Rescale theta, attaining an equivalent set of parameters.
    # (note that the full model is non-identifiable, so that such scaling is possible).
    # unit  : either '2N0', 'u0' or 'r'.
    # val   : target value for the scaled unit.
    # Returns an equivalent Theta object, which unit is equal to val.
    def scale(self, unit, val):
        
        assert val > 0
        
        # determine current value of the scaled parameter
        if unit == 'u0':
            currVal = self.uVals[0]
        elif unit == '2N0':
            currVal = self.lmbVals[0]
            # switch input val from 2N to lambda (the coalescence rate)
            val = 1.0/val
        else:
            assert unit == 'r'
            currVal = self.r
        
        # calculate the multiplicative scaling factor
        C = val/currVal
        
        # calculate scaled values
        # r *= C
        r = self.r*C
        # u *= C
        uVals = [x*C for x in self.uVals]
        # lambda *= C
        lmbVals  = [x*C for x in self.lmbVals]
        # boundaries /= C
        boundaries = [x/C for x in self.segments.boundaries]
        
        # return scaled instance
        return Theta(boundaries, lmbVals, uVals, r)
    
    # calculate pi
    # TODO support more than 2 emission types?
    def pi(self):
        pi = 0.0
        for i in xrange(self.nStates):
            pi += self.initialDist[i] * self.emissionMat[i,siteTypes.het]
        return pi
    
    # Retrun lambda at time t
    def lmb(self, t):
        return self.lmbVals[self.segments.bin(t)]
    
    # Retrun u at time t
    def u(self, t):
        return self.uVals[self.segments.bin(t)]
            
    # Unite time-segments for which all parameters are identical.
    # (note this changes the definition of the hmm; I'm using this method just for output printing purposes)
    def collapseSegments(self):
        r = self.r
        uVals = self.uVals[:1]
        lmbVals = self.lmbVals[:1]
        boundaries = self.segments.boundaries[:1]
        for u, lmb, bound in zip(self.uVals[1:], self.lmbVals[1:], self.segments.boundaries[1:-1]):
            if (u,lmb) != (uVals[-1],lmbVals[-1]):
                uVals.append(u)
                lmbVals.append(lmb)
                boundaries.append(bound)
        boundaries.append(np.inf)
        return Theta(boundaries, lmbVals, uVals, r)
                
            
    # Serialize class to human-readable string 
    def __str__(self):
        
        self._assertValidValues()
        res = '{0:<32}{1:<24}\n\n'.format('recombination rate:',self.r)
        res += 'Mutation & coalescence rate histories:\n'
        template = '\t{0:<24}{1:<24}{2:<24}{3:<24}\n'
        res += template.format('t_start', 't_end', 'coalescence rate', 'mutation rate')
        for i in xrange(self.segments.n):    
            res += template.format(self.segments.boundaries[i], self.segments.boundaries[i+1], self.lmbVals[i], self.uVals[i])
        return res
    
    
    # Deserialize class from string
    @classmethod
    def fromString(cls, inp):
        
        # define format pattern
        pattern = re.compile(r"""
                                 ^
                                  recombination\ rate:\ +(?P<r>.+)\n\n      # recombination rate
                                  .+\n\t.+\n                                # table headers
                                  (?P<tab>(\t.+\n)+)                        # table
                                 $
                              """, re.VERBOSE)
        # match input to pattern
        match = pattern.search(inp)
        assert match is not None
        
        # parse recombintion rate
        r = float(match.group("r"))
        
        # parse mutation & coalescence rates line by line
        boundaries, lmbVals, uVals = [], [], []
        tEndPrevious               = 0.0
        for line in match.group("tab").split('\n')[:-1]:
            
            tStart, tEnd, lmb, u = map(float, re.findall(r"([-|\w|\+|\.]+)", line))
            assert tStart == tEndPrevious
            tEndPrevious = tEnd
            
            boundaries.append(tStart)
            lmbVals.append(lmb)
            uVals.append(u)
        
        boundaries.append(tEnd)    
        res = cls(boundaries, lmbVals, uVals, r)
        res._assertValidValues()
        return res
    
    # Calculate distance metric from another parameter theta:
    # First, both thetas are scaled such that scaleUnit is equal to val.
    # Then, a different distance score is calculated for r, u, and N;
    # For each of these, d is the weighted absolute difference between the values in the 2 thetas,
    # where the weights correspond to self.initialDist (the stationary probabilities of the time segments under self).
    # (note that this 'metric' is not symmetric as self.initialDist and theta.initialDist are not the same).
    # TODO perhaps use standard metric for HMM's???
    def d(self, theta, scaleUnit, val = 1.0):
        
        # scale self, theta to have identical sacle units
        scSelf = self.scale(scaleUnit, val)
        scTheta = theta.scale(scaleUnit, val)
        
        # refine boundaries such that both sets of parameters are defined on same boundaries
        scSelf = scSelf.refine(scTheta.segments.boundaries)
        scTheta = scTheta.refine(scSelf.segments.boundaries)
        assert scSelf.segments.boundaries == scTheta.segments.boundaries
        
        # calculate distance for N, u and r
        selfN = np.array([0.5/lmb for lmb in scSelf.lmbVals])
        thetaN = np.array([0.5/lmb for lmb in scTheta.lmbVals])
        dn = np.sum(np.abs(selfN - thetaN) * scSelf.initialDist)
        du = np.sum(np.abs(np.array(scSelf.uVals) - np.array(scTheta.uVals)) * scSelf.initialDist)
        dr = abs(scSelf.r - scTheta.r)
        return dn, du, dr
    
    
    # refine the boundaries such that they also include all of the time points given in additionalBnds
    # returns a theta onject with the refined boundaries
    def refine(self, additionalBnds):
        
        # define refined boundaries
        refBnds = sorted(set(additionalBnds + self.segments.boundaries))
        
        # map refined segments to old segments
        inds = np.digitize(refBnds[:-1],self.segments.boundaries) - 1
        
        # re-define u and lambda on the refined segs
        uVals = [self.uVals[i] for i in inds]
        lmbVals = [self.lmbVals[i] for i in inds]
        
        return Theta(refBnds, lmbVals, uVals, self.r)
        
    
    # Sanity check: verify all values are valid
    def _assertValidValues(self):
        for val in self.lmbVals + self.uVals + [self.r]:
            assert 0 < val < np.inf
        assert self.segments.boundaries[0] == 0.0
        for i in xrange(0,len(self.segments.boundaries)-1):
            assert self.segments.boundaries[i] < self.segments.boundaries[i+1]
        assert np.isinf(self.segments.boundaries[-1])
        assert self.segments.n == len(self.uVals) == len(self.lmbVals)
        


    