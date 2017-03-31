import itertools
import math
from collections import defaultdict
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
 

# calculates GOF statistic G_l (similar to the GOF metric used in the PSMC paper).
# NOTE: instead of measuring the distance (between observed dist. and fitted dist.) by relative entropy,
#       we use total variation distance.
class GOF(object):
    
    # observations: list of ObservedSequence instances.
    # l           : length (for calculating G_l)
    def __init__(self, model, observations, l):
        
        self._l     = l
        self._model = model

        # Implementation assumes only 2 emissions.
        assert model.nEmissions == 2
        
        # calculate the observed distribution
        M = 2**l
        # H - an histogram of l-sequences
        H = defaultdict(int)
        
        for obs in observations:
            
            # pos: last parsed pos
            pos = -1
            # n is a binary representation of a l-sequence
            n   =  0
            
            for nextPos, nextType in itertools.izip(obs.positions, obs.posTypes):
                
                # advance pos to nextPos
                while pos < nextPos:
                    
                    # jump all the way to (nextPos - 1) (but only if all zeroes all the way there)
                    if (pos >= (l-1)) and (n == 0):
                        jump  = nextPos - 1 - pos
                        H[0] += jump
                        pos  += jump
                        
                    # advance pos by one step :
                        
                    # shift n and enter the new bit as lsb
                    n  = (2*n)%M
                    if pos == (nextPos - 1):
                        assert nextType in [0,1]
                        n += nextType
                    
                    pos += 1
                    
                    # add to histogram (unless pos < (l-1))
                    if pos >= (l-1):
                        H[n] += 1
                
        # normalize H to distribution (i.e. to sum to one)
        s = float(sum(H.values()))
        self._pObs = dict()
        for k in H.keys():
            self._pObs[k] = float(H[k]) / s
        
            
    # return the statistic G_l comparing GOF between theta and the observed value
    # TODO is this really slow??? (I generate the transition matrix multiple times)
    def G(self, theta):
        
        M, Md2 = (2**self._l), (2**(self._l-1))
        res, sObs    = 0.0 ,0.0

        # calculate expected distribution of l-sequences under theta
        # first for sequences that were actually observed
        for n in self._pObs.keys():
            
            # break n into a binary sequence (where the first element id the MSB)
            seq = []
            assert n < M
            t = n
            for _ in xrange(self._l):
                msb = t/Md2
                assert msb in [0,1]
                seq.append(msb)
                t = (t*2)%M
            
            # represent n as an ObservedSequence object
            seq = ObservedSequence.fromEmissionsList(seq)
            
            # get the log-likelihood of the sequence given theta
            estP  = math.exp(BaumWelchExpectation(self._model, theta, seq).inferHiddenStates().logL)
            sObs += estP 
            
            # update total-variance distance
            res += ( 0.5 * abs(estP - self._pObs[n]) )
            
        # update total-variance distance for unobserved sequences
        assert (sObs <= 1.0)
        res += 0.5 *(1.0 - sObs)
        
        return res
