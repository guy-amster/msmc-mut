import re
import numpy as np
from BaumWelch import HMM
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs

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
    # calcHmm    : Calculate hmm fields on initialization. 
    def __init__(self, boundaries, lmbVals, uVals, r, calcHmm=True):
        
        # Initialize Segmnets instance
        self.segments = Segments(boundaries)
        assert self.segments.n == len(uVals) == len(lmbVals)
        
        # read parameter values
        self.lmbVals  = lmbVals
        self.uVals    = uVals
        self.r        = r
        self._calcHmm = calcHmm
        
        # define HMM properties
        if self._calcHmm:
            self.nStates = self.segments.n
            self.nEmissions = 2
            # TODO this should be a module not a class obviously
            emissionMat = EmissionProbs(self).emissionMat()
            # TODO check for loss of performance... decide what to do & change TransProbs accordingly
            transitionProbs = TransitionProbs(self)
            transitionMat, initialDist = transitionProbs.transitionMat(), transitionProbs.stationaryProb()
            
            HMM.__init__(self, self.segments.n, 2, transitionMat, initialDist, emissionMat)
    
    
    # Scale the time-unit by a positive constant C.
    # (r, u and N are rescaled appropriately, so that the coalescence & mutation processes are invariant to this rescaling ).
    # Return value: a Theta instance with the scaled values
    def scale(self, C, calcHmm=True):
        
        assert C > 0
        
        # calculate scaled values
        r = self.r*C
        uVals = [x*C for x in self.uVals]
        lmbVals  = [x*C for x in self.lmbVals]
        boundaries = [x/C for x in self.segments.boundaries]
        
        # return scaled instance
        return Theta(boundaries, lmbVals, uVals, r, calcHmm=calcHmm)
    
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
        return Theta(boundaries, lmbVals, uVals, r, calcHmm=False)
                
            
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
    def fromString(cls, inp, calcHmm = True):
        
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
        res = cls(boundaries, lmbVals, uVals, r, calcHmm=calcHmm)
        res._assertValidValues()
        return res
    
    
    # Sanity check: verify all values are valid
    def _assertValidValues(self):
        for val in self.lmbVals + self.uVals + [self.r]:
            assert 0 < val < np.inf
        assert self.segments.boundaries[0] == 0.0
        for i in xrange(0,len(self.segments.boundaries)-1):
            assert self.segments.boundaries[i] < self.segments.boundaries[i+1]
        assert np.isinf(self.segments.boundaries[-1])
        assert self.segments.n == len(self.uVals) == len(self.lmbVals)
        


    