from Containers import Model, Theta
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
import scipy.integrate as integrate
import math
import numpy as np
import coalSim

def compVecs(v1, v2):
        assert len(v1) == len(v2)
        return [v1[i]/v2[i] for i in xrange(len(v1))]

model = Model()
fixedMu = False
theta = Theta(model).randomize(fixedMu)
theta.printVals()

for k in xrange(6,11):
    L = 3*(10**k)
    print '***********************************'
    print 'simulating with 3*(10**%d) bps...'%k
    
    obsTrans, obsEmiss, summ = coalSim.sampleSequence(model,theta,L)
    
    # sanity check: verify transition and emission probabilities match observed sequence
    AnProbs = TransitionProbs(model,theta)
    AnEmiss = EmissionProbs(model,theta)
    
    maxStd = 0.0
    for origin in xrange(model.nStates):
        for target in xrange(model.nStates):
            
            pAnl = AnProbs.transitionProb(origin,target)
            pObs = obsTrans[origin,target]
            
            std  = (pObs - pAnl) / math.sqrt(pAnl*(1.0-pAnl))
            n = summ.homEmissions[origin] + summ.hetEmissions[origin]
            
            assert n > 40
            std *= math.sqrt(n)
            maxStd = max(abs(std), maxStd)
            
            if abs(std) > 6:
                print '*****'
                print 'trs', origin, target, pAnl, pObs, n, std
        
    print 'max. std\'s for transition matrix: %f'%maxStd
        
    maxStd = 0.0
    for origin in xrange(model.nStates):
        pHomAnl = AnEmiss.homProb(origin)
        pHomObs = obsEmiss[origin]
        n = summ.homEmissions[origin] + summ.hetEmissions[origin]
        std = (pHomObs - pHomAnl) * math.sqrt(n) / math.sqrt(pHomAnl*(1.0-pHomAnl))
        maxStd = max(abs(std), maxStd)
        if abs(std) > 6:
            print '*****'
            print 'emission', origin, pHomAnl, pHomObs, std
    print 'max. std\'s for emission vec: %f'%maxStd
    
    for method in ['COBYLA']:
        for _ in xrange(1):
            print '********** ' + method + ' **********'
            # now maximize likelihood
            summ.normalize()
            trueLikelihood = summ.logLikelihood(theta)
            
            thetaMax = summ.maximizeLogLikelihood(method=method)
            MaxL = summ.logLikelihood(thetaMax)
            
            print 'Relative fit:'
            print 'lambda: ', compVecs(thetaMax.lambdaV, theta.lambdaV)
            print 'u     : ', compVecs(thetaMax.uV,      theta.uV)
            print 'r     : ', thetaMax.r/theta.r
            
            of = (MaxL-trueLikelihood)
            ofS = 'of (overfitting val: %f (found %f, true %f))'%(of,MaxL,trueLikelihood)
            if MaxL < trueLikelihood:
                ofS = 'NO ' + ofS
            print ofS
                                               