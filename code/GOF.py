import itertools
import math
from collections import defaultdict
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
from multiprocessing import Pool
from Logger import log

# Calculate the predicted probability of a specific subsequence seq given theta.
# (Function defined at the module level to allow calling from pool map)
# seq : ObservedSequence instance.
def _calcPredictedProb(inpDict):
    model = inpDict['model']
    theta = inpDict['theta']
    seq   = inpDict['seq']
    return math.exp(BaumWelchExpectation(model, theta, seq).inferHiddenStates().logL)


# Calculate the observed distribution of all length-l subsequences of an observed suquence.
# (Function defined at the module level to allow calling from pool map)
# l   : Subsequence-length.
# obs : ObservedSequence instance.
def _calcObservedDist(inpDict):
    l   = inpDict['l']
    obs = inpDict['obs']
    
    # H - an histogram of l-sequences
    H = defaultdict(int)
    
    M = 2**l
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
    
    return H


# calculates GOF statistic G_l (similar to the GOF metric used in the PSMC paper).
# NOTE: instead of measuring the distance (between observed dist. and fitted dist.) by relative entropy,
#       we use total variation distance.
class GOF(object):
    
    # observations: list of ObservedSequence instances.
    # l           : length (for calculating G_l)
    # nPrc        : number of processes to use
    def __init__(self, model, observations, l, nPrc):
        
        self._model = model
        self._nPrc  = nPrc

        # Implementation assumes only 2 emissions.
        assert model.nEmissions == 2
        
        # calculate a histogram of observed sequences in parallel
        inputs = [{'l': l, 'obs':obs} for obs in observations]
        p   = Pool(nPrc)
        res = p.map(_calcObservedDist, inputs)       
        p.close()
        
        # sum result histogram to one summary histogram H
        H = defaultdict(int)
        for d in res:
            for k,v in d.iteritems():
                H[k] += v
        
        # create two lists:
        # - _obsSequences, containing all observed l-sequences
        # - _obsProbs, containng the matching observed probabilities
        self._obsSequences, self._obsProbs = [], []
        
        s = float(sum(H.itervalues()))
        M, halfM = 2**l, 2**(l-1)
        
        # for all observed l-sequences:
        for n in H.keys():
            
            # normalize count to probability
            self._obsProbs.append(float(H[n]) / s)
            
            # map n to a binary sequence (where the first element is the msb)
            seq = []
            assert n < M
            t = n
            for _ in xrange(l):
                msb = t/halfM
                assert msb in [0,1]
                seq.append(msb)
                t = (t*2)%M
            
            # represent n as an ObservedSequence object
            self._obsSequences.append(ObservedSequence.fromEmissionsList(seq))
                
        log('Input sequences contain %d distinct %d-sequences (used for GOF statistics)'%(len(self._obsProbs),l))
        
            
    # return the statistic G_l comparing GOF between theta and the observed value
    def G(self, theta):
        
        # calculate the predicted probabilities of the observed sequences
        inputs    = [{'model': self._model, 'theta':theta, 'seq':seq} for seq in self._obsSequences]
        p         = Pool(self._nPrc)
        predProbs = p.map(_calcPredictedProb, inputs)       
        p.close()
        
        # calulate total-variation distance TV between observed and predicted distr. of l-sequences:
        
        # First, for l-sequences that weren't observed
        s         = sum(predProbs)
        assert s <= 1.0
        TVdist    = (0.5*(1.0-s))
        
        # Then, for l-sequences with positive observed probability  
        for predP,obsP in itertools.izip(predProbs, self._obsProbs):
            TVdist += 0.5*abs(predP - obsP)
        
        return TVdist
