from Containers import HmmModel, HmmTheta
import numpy as np
from BaumWelchExpectation import BaumWelchExpectation
from BaumWelch import BaumWelch
from ObservedSequence import ObservedSequence


# test BW implementation on a general HMM model

n, m      = 6, 6
model     = HmmModel(n,m)
trueTheta = HmmTheta.randomize(model)

L             = 10**7

s = 3
model     = HmmModel(s,s)
fs = float(s)
transitionMat = np.ones((s,s))*(.1/(fs-1))
emissionMat   = np.ones((s,s))*(.1/(fs-1))
iDist         = np.ones(s)/fs
for i in xrange(s):
        transitionMat[i,i] = 0.9
        emissionMat[i,i] = 0.9
trueTheta     = HmmTheta(model, transitionMat, iDist, emissionMat)
        
'''
transitionMat = np.array([[.95,.05],[.05,.95]])
emissionMat   = np.array([[.8,.2],[.2,.8]])
iDist         = np.array([.5,.5])
trueTheta     = HmmTheta(model, transitionMat, iDist, emissionMat)
'''
# TODO more emissions...
def randomSeq(model, theta, L):
        
        states    = range(model.nStates)
        emissions = range(model.nEmissions)
        
        res       = []        
        emissionMat = theta.emissionMat()
        iDist, transitionMat = theta.chainDist()
        
        # initial sample
        dist = iDist
        
        for pos in xrange(L):
                state = np.random.choice(states, p=dist)
                res.append(np.random.choice(emissions, p=emissionMat[state ,:]))
                dist = transitionMat[state, :]
        
        return res

obs = ObservedSequence.fromEmissionsList(randomSeq(model, trueTheta, L))
print 'simulation complete.'

while True:
        BaumWelch(model, [obs], 1, 30, trueTheta=trueTheta)
        raw_input()
