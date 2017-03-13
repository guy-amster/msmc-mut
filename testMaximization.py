from Containers import Model, Theta
import scipy.integrate as integrate
import numpy as np
from HiddenSeqSummary import HiddenSeqSummary

verifyProbs = True

def compVecs(v1, v2):
        assert len(v1) == len(v2)
        return [v1[i]/v2[i] for i in xrange(len(v1))]

model = Model()
trueTheta = Theta.random(model)
trueTheta.printVals()

L = 10**7

iDist, transProb = trueTheta.chainDist()
transProportion  = np.empty( (model.nStates, model.nStates) )
for i in xrange(model.nStates):
        for j in xrange(model.nStates):
                transProportion[i,j] = transProb[i,j] * iDist[i] * (L-1)

emmProb = trueTheta.emissionMat()
emmProportion = np.empty( (model.nStates, model.nEmissions) )
for i in xrange(model.nStates):
        for j in xrange(model.nEmissions):
                emmProportion[i,j] = emmProb[i,j] * iDist[i] * (L)

summ = HiddenSeqSummary(model, L, transProportion, emmProportion, iDist, None)

for _ in xrange(9):
        print '*********'
        newTheta, Qmax = summ.maximizeQ()
        Qtrue = summ.Q(trueTheta)
        print 'Qmax-Qtrue: ', Qmax-Qtrue
        newTheta.printVals()
        
        print 'Relative fit:'
        print 'lambda: ', compVecs(newTheta.lambdaV, trueTheta.lambdaV)
        print 'u     : ', compVecs(newTheta.uV,      trueTheta.uV)
        print 'r     : ', newTheta.r/trueTheta.r
        raw_input()