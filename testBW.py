from Containers import Model, Theta
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
from SiteTypes import siteTypes
import scipy.integrate as integrate
import math
import numpy as np
import coalSim


verifyProbs = True

def compVecs(v1, v2):
        assert len(v1) == len(v2)
        return [v1[i]/v2[i] for i in xrange(len(v1))]

model = Model()
fixedR=False
fixedLambda=False
fixedMu=False
theta = Theta(model).randomize(fixedR=fixedR, fixedLambda=fixedLambda, fixedMu=fixedMu)
theta.printVals()

for _ in xrange(4):
    k=6 
    L = 3*(10**k)
    print '***********************************'
    print 'simulating with 3*(10**%d) bps...'%k
    
    summ, _ = coalSim.sampleSequence(model,theta,L)
    
    # sanity check: verify transition and emission probabilities match observed sequence
    if verifyProbs:
        AnProbs = TransitionProbs(model,theta)
        AnEmiss = EmissionProbs(model,theta)
        
        maxStd = 0.0
        for origin in xrange(model.nStates):
            for target in xrange(model.nStates):
                
                pAnl = AnProbs.transitionProb(origin,target)
                pObs = summ.transitions[origin,target]/np.sum(summ.transitions[origin,:])
                
                std  = (pObs - pAnl) / math.sqrt(pAnl*(1.0-pAnl))
                n = L*np.sum(summ.transitions[origin,:])
                
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
            pHomObs = summ.emissions[origin,siteTypes.hom]/np.sum(summ.emissions[origin,:])
            n = L*np.sum(summ.emissions[origin,:])
            std = (pHomObs - pHomAnl) * math.sqrt(n) / math.sqrt(pHomAnl*(1.0-pHomAnl))
            maxStd = max(abs(std), maxStd)
            if abs(std) > 6:
                print '*****'
                print 'emission', origin, pHomAnl, pHomObs, std
        print 'max. std\'s for emission vec: %f'%maxStd
    
    print '********** ' 
    # now maximize likelihood
    # TODO why does it collapse without normalization?
    trueLikelihood = summ.logLikelihood(theta)
    
    thetaMax       = summ.maximizeLogLikelihood()
    MaxL           = summ.logLikelihood(thetaMax)
    
    print 'Relative fit:'
    print 'lambda: ', compVecs(thetaMax.lambdaV, theta.lambdaV)
    print 'u     : ', compVecs(thetaMax.uV,      theta.uV)
    print 'r     : ', thetaMax.r/theta.r
    
    of = (MaxL-trueLikelihood)
    ofS = 'of (overfitting val: %f (found %f, true %f))'%(of,MaxL,trueLikelihood)
    if MaxL < trueLikelihood:
        ofS = 'NO ' + ofS
    print ofS
                                               