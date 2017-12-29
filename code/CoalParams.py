import re
import numpy as np
# TODO import segments

# CoalParams: A container class specifying parameters for the coalescence process:
#             piecewise coalescence rates, piecewise mutation rates and recombination rate.

class CoalParams(object):
    
    # segments : A Segments instance specifying the intervals on which the piecewise functions are defined.
    # lmbVals  : Coalescence rate (lambda) values, per unit-time.
    # uVals    : Mutation rate values per unit-time.
    # r        : Recombination rate values per unit-time.
    # TODO here I don't support Nulls!!! 
    def __init__(self, segments, lmbVals, uVals, r):
        
        assert segments.n == len(uVals) == len(lmbVals)
        
        self.segments = segments
        self.lmbVals  = lmbVals
        self.uVals    = uVals
        self.r        = r
    
    
    # Rescale the time-unit by a positive constant C.
    # (r, u and N are rescaled appropriately, so that the coalescence & mutation processes are invariant to this rescaling ).
    def rescale(self, C):
        
        assert C > 0
        
        self.r       *= C
        self.uVals    = [x*C for x in uVals  ]
        self.lmbVals  = [x*C for x in lmbVals]
        self.segments = Segments([x/C for x in self.segments.boundaries])
    
            
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
            
            boundaries.append(t_start)
            lmbVals.append(lmb)
            uVals.append(u)
        
        boundaries.append(tEnd)    
        res = cls(Segments(boundaries), lmbVals, uVals, r)
        res._assertValidValues()
        return res
    
    
    # Sanity check: verify all values are valid
    def _assertValidValues(self):
        for val in self.lmbVals + self.uVals + [self.r]:
            assert 0 < val < np.inf
        # TODO move to Segments.assert
        assert self.segments.boundaries[0] == 0.0
        for i in xrange(0,len(self.segments.boundaries)-1):
            assert self.segments.boundaries[i] < self.segments.boundaries[i+1]
        assert np.isinf(self.segments.boundaries[-1])
        assert self.segments.n == len(self.uVals) == len(self.lmbVals)
        


    