from Containers import Model, Theta
from BaumWelch import BaumWelch
from BaumWelchExpectation import BaumWelchExpectation
import math
import numpy as np
import coalSim
from multiprocessing import Pool



model = Model()
theta = Theta.random(model)

L = 10**6
observations = []
for _ in xrange(2):
        _, obs = coalSim.sampleSequence(model,theta,L)
        observations.append(obs)

def calcExp(d):
        return BaumWelchExpectation(d['model'], d['theta'], d['obs']).inferHiddenStates()

inp = [{'model':model, 'theta':theta, 'obs':obs} for obs in observations]

p   = Pool(2)
res = p.map(calcExp, inp)       
p.close()

exp0 = BaumWelchExpectation(model, theta, observations[0]).inferHiddenStates()
exp1 = BaumWelchExpectation(model, theta, observations[1]).inferHiddenStates()

print exp0.transitions - res[0].transitions
print exp0.transitions
print exp1.transitions - res[1].transitions


'''
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
'''