from Containers import Model, Theta
from TransitionProbs import TransitionProbs
from EmissionProbs import EmissionProbs
from SiteTypes import siteTypes
import scipy.integrate as integrate
import math
import numpy as np
import coalSim

model = Model()


for i in xrange(1):
    theta = Theta(model).randomize()
    #theta.lambdaV[-1] = .0001
    # there seems to be a problem when lambda is very low at the last segment; see, for example,
    #theta.lambdaV = [0.6781413780161791, 0.5615490702054557, 1.1220031291049821, 0.37167886032058767, 1.7942202414843498, 0.5249107876651742, 2.4331399466661314, 0.04190780562189933]
    # when r is large - that's reasonable beacuase >1 rec. events become feasable.
    print theta.lambdaV
    for j in xrange(1):
        L = 1000000
        AnProbs = TransitionProbs(model,theta)
        emissAnl = EmissionProbs(model,theta)
        summ, _ = coalSim.sampleSequence(model,theta,L )
        
        s1, s2 = np.inf, np. inf
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
        
        print 'trans std', maxStd
        
        maxStd = 0.0
        for origin in xrange(model.nStates):
            pHomAnl = AnEmiss.homProb(origin)
            pHomObs = summ.emissions[origin,siteTypes.hom]/np.sum(summ.emissions[origin,:])
            n = L*np.sum(summ.emissions[origin,:])
            std = (pHomObs - pHomAnl) * math.sqrt(n) / math.sqrt(pHomAnl*(1.0-pHomAnl))
            maxStd = max(abs(std), maxStd)
            if abs(std) > 6:
                print 'emission', origin, pHomAnl, pHomObs, std
        print 'emission std', maxStd