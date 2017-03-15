from Containers import HmmModel, HmmTheta, Theta
import numpy as np
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
import itertools
import time # TODO REMOVE
from multiprocessing import Pool


def _distPermute(A1, A2, B1, B2):
        dist = np.inf
        n    = A1.shape[0]
        for p in itertools.permutations(range(n)):
                A2P = A2[p,:][:,p]
                B2P = B2[p,:]
                
                d = np.sum(abs(A1 - A2P)) + np.sum(abs(B1 - B2P))
                
                if d < dist:
                        dist = d
                        pMin = p
        print A1
        print A2[p,:][:,p]
        print B1
        print B2[p,:]
        return dist

# auxiliary function, used just for debugging & simulation purposes (see below)
def _dist(theta1, theta2):
        _, trans1 = theta1.chainDist()
        _, trans2 = theta2.chainDist()
        emiss1    = theta1.emissionMat()
        emiss2    = theta2.emissionMat()
        res =  _distPermute(trans1, trans2, emiss1, emiss2)
        return res

def _calcExp(d):
        return BaumWelchExpectation(d['model'], d['theta'], d['obs']).inferHiddenStates()

def _parallelExp(model, theta, observations, nPrc):
        
        
        inp = [{'model':model, 'theta':theta, 'obs':obs} for obs in observations]
        
        p   = Pool(nPrc)
        res = p.map(_calcExp, inp)       
        p.close()
        
        resSum = res[0]
        for i in xrange(1, len(res)):
                resSum = resSum + res[i]
        return resSum
        
# model         : a (derived) HmmModel class
# observations  : a list of ObservedSequence objects
# nProcesses    : number of processors to parallelize on.
# nIterations   : number of BW iterations. # TODO is there's a standard stopping criteria?
# trueTheta     : for simulated data (will be used for printing statistics)
# thetaInit     : a theta value to initiate the BW process from; default is to use a random theta.
def BaumWelch(model, observations, nProcesses = 1, nIterations = 20, trueTheta = None, theta = None):
        
        # We can use at most one processor per sequence.
        nProcesses = min(nProcesses, len(observations))
                
        # initialize theta
        if theta == None:
                if model.modelType == 'basic':
                        theta = HmmTheta.random(model)
                else:
                        theta =    Theta.random(model)
                
        # we expect the log likelihood at the next iteration to be higher than this        
        bound = -np.inf
        
        if trueTheta != None:
                trueL = _parallelExp(model, trueTheta, observations, nProcesses).logL
                # TODO REMOVE trueL = BaumWelchExpectation(model, trueTheta, observations[0]).inferHiddenStates().logL
                print 'log-likelihood under true theta: %f'%trueL
                
        for iter in xrange(nIterations):
                
                # BW expectation step
                start = time.time()
                exp  = _parallelExp(model, theta, observations, nProcesses)
                # TODO REMOVE exp = BaumWelchExpectation(model, theta, observations[0]).inferHiddenStates()
                print 'timing exp: ', (time.time() - start)
                
                # sanity check: log(O|theta) has increased as expected in the last iteration
                if exp.logL < bound:
                        print 'WARNING **** BW error 1 %f %f'%(exp.logL, bound)
                
                Qtheta = exp.Q(theta)
                print 'iteration %d: log-likelihood %f '%(iter, exp.logL)
                
                # print statistics for simulated data
                if trueTheta != None:
                        # TODO EITHER REMOVE OR MOVE DIST TO THETA CLASSES
                        #x = _dist(trueTheta, theta)
                        #print 'distance from true theta: ', x
                        pass
                
                # sanity check (this is just Jensen's inequality... Q(theta | theta) = E( log(P(O,Z|theta) ) <= log( E(P(O,Z|theta)) ) = log( P(O|theta) ) 
                if Qtheta > exp.logL:
                        print 'WARNING **** BW error 2 %f %f'%(Qtheta, exp.logL)

                
                # maximization step
                start = time.time()
                newTheta, Qmax = exp.maximizeQ(nProcesses = nProcesses)
                print 'timing max: ', (time.time() - start)
                
                # sanity check: max_thetaStar Q(thetaStar | theta) >= Q(theta | theta)
                qDiff = Qmax - Qtheta
                #print 'Qdiff %f'%qDiff
                if qDiff < 0:
                        print 'WARNING **** BW error 3 %f %f'%(Qmax, Qtheta)
                
                if trueTheta != None:
                        if exp.Q(trueTheta) >= Qmax:
                                print 'WARNING **** BW ERROR 3 %f %f'%(exp.Q(trueTheta), Qmax)
        
                # the log likelihood of newTheta should be higher by at least qDiff
                # (this is the inequality you get in the standard proof showing EM converges to a local maximum)
                bound = exp.logL + qDiff
                
                # update theta
                theta = newTheta
                                
        if trueTheta != None:
                print 'log-likelihood under true theta: %f'%trueL
        
        print 'Baum-Welch done.'        
        return theta
                
