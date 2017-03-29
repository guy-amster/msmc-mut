import random
import numpy as np
import math
from Containers import Model, Theta
from HiddenSeqSummary import HiddenSeqSummary
from ObservedSequence import ObservedSequence
from SiteTypes import siteTypes
from collections import defaultdict

# Simulates coalescence & mutational processes.
# Input:
#         - chrLength: Chromosome length (in base-pairs).
# Returns:
#         - Summary statistics on the simulated hidden state.
#         - An observed sequence object (if sampleObserved is True).
def sampleSequence(model, theta, chrLength):
    
    # Count the number of transitions from state i to j as sampleCount[i,j]
    transHist = np.zeros((model.nStates,model.nStates), dtype=np.uint64)
    
    # Sum the probability from state i to emissionHist[i,o], where o=0 stands for hom. and o=1 for het.
    emissionHist = np.zeros((model.nStates,2), dtype=np.uint64)
    homHist = defaultdict(int)
    hetHist = defaultdict(int)

    cummU = dict()
    cummU[0] = 0.0
    for i in xrange(1,model.segments.n):
        cummU[i] = cummU[i-1] - 2*model.segments.delta[i-1]*theta.uV[i-1]
    
    # number of states already sampled
    ind = -1
    
    segPositions = []
    
    # start with a very recent TMRCA, so a recombination event would result in a stationary sample
    currentTime = 0.00001*model.segments.boundaries[1]
    
    while ind < chrLength:
        
        # handle recombination event:
        # sample recombination time:
        u = random.uniform(0, currentTime)
        nextTime = u
        bounds = sorted([currentTime] + [b for b in model.segments.boundaries if b>u])
        for nextBound in bounds:
            rate = theta.lambdaV[model.segments.bin(nextTime)]
            # floating branch
            if nextTime < currentTime:
                rate *= 2
            t = random.expovariate(rate)
            if (nextTime + t) < nextBound:
                nextTime += t
                
                # with probability .5, coalescense is between floating branch and itself (no change to current time)
                if nextTime < currentTime:
                    if random.random() < .5:
                        nextTime = currentTime
                
                # we're done.        
                break
            else:
                nextTime = nextBound
        
        currentBin, nextBin = model.segments.bin(currentTime), model.segments.bin(nextTime)
        if ind >= 0:
            transHist[currentBin, nextBin] += 1
        ind += 1
        currentTime, currentBin = nextTime, nextBin
        
        # sample number of steps until next recombination event occurs
        p    = -math.expm1(-2*theta.r*currentTime)
        
        assert p >  0.0
        assert p <= 1.0
        
        jump = np.random.geometric(p) - 1
        jump = min(jump, chrLength - ind)
        
        # update number of steps
        ind += jump
        
        # update transition histogram
        transHist[currentBin, currentBin] += jump
        
        # update emissions
        pHom = math.exp(cummU[currentBin]-2*theta.uV[currentBin]*(currentTime-model.segments.boundaries[currentBin]))
        pHet = 1.0 - pHom
        assert pHet > 0.0
        assert pHet < 1.0
        nHet = 0
        pos  = ind - jump
        while pos <= ind:
            pos += (np.random.geometric(pHet) - 1)
            if pos <= ind:
                nHet += 1
                segPositions.append(pos)
                pos  += 1
        assert nHet <= (jump + 1)
        nHom = jump + 1 - nHet
        emissionHist[currentBin,siteTypes.hom] += nHom
        emissionHist[currentBin,siteTypes.het] += nHet
    
    obsSeq = ObservedSequence(chrLength, segPositions, [1 for _ in segPositions])
    
    transitions = np.zeros((model.nStates,model.nStates), dtype=np.float64)
    emissions   = np.zeros((model.nStates,2), dtype=np.float64)
    for i in xrange(model.nStates):
        for j in xrange(model.nStates):
            transitions[i,j] = float(transHist[i,j]) / float(chrLength - 1)
        for j in xrange(2):
            emissions[i,j]   = float(emissionHist[i,j]) / float(chrLength)
    
    hiddenSeq = HiddenSeqSummary(model, chrLength, transitions, emissions, np.zeros(model.nStates), None)
    
    return hiddenSeq, obsSeq
    