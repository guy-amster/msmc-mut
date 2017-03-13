from Containers import HmmModel, HmmTheta
import numpy as np
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
import itertools

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

# model         : a (derived) HmmModel class
# observations  : a list of ObservedSequence objects
# nProcesses    : number of processors to parallelize on.
# nIterations   : number of BW iterations. # TODO is there's a standard stopping criteria?
# trueTheta     : for simulated data (will be used for printing statistics)
def BaumWelch(model, observations, nProcesses, nIterations, trueTheta = None):
        
        # We can use at most one processor per sequence.
        nProcesses = min(nProcesses, len(observations))
        
        # TODO implement
        nProcesses = 1
        assert len(observations) == 1
        
        # initialize theta
        if model.modelType == 'basic':
                theta = HmmTheta.randomize(model)
        else:
                theta =    Theta.randomize(model)
                
        # we expect the log likelihood at the next iteration to be higher than this        
        bound = -np.inf
        
        if trueTheta != None:
                trueL = BaumWelchExpectation(model, trueTheta, observations[0]).inferHiddenStates().logL
                print 'log-likelihood under true theta: %f'%trueL
                
        for iter in xrange(nIterations):
                
                # BW expectation step
                exp = BaumWelchExpectation(model, theta, observations[0]).inferHiddenStates()
                
                # sanity check: log(O|theta) has increased as expected in the last iteration
                if exp.logL < bound:
                        print 'WARNING **** BW error 1 %f %f'%(exp.logL, bound)
                
                Qtheta = exp.Q(theta)
                print 'iteration %d: log-likelihood %f '%(iter, exp.logL)
                
                # print statistics for simulated data
                if trueTheta != None:
                        x = _dist(trueTheta, theta)
                        print 'distance from true theta: ', x
                
                # sanity check (this is just Jensen's inequality... Q(theta | theta) = E( log(P(O,Z|theta) ) <= log( E(P(O,Z|theta)) ) = log( P(O|theta) ) 
                if Qtheta > exp.logL:
                        print 'WARNING **** BW error 2 %f %f'%(Qtheta, exp.logL)

                
                # maximization step
                newTheta, Qmax = exp.maximizeQ()
                
                # sanity check: max_thetaStar Q(thetaStar | theta) >= Q(theta | theta)
                qDiff = Qmax - Qtheta
                #print 'Qdiff %f'%qDiff
                if qDiff < 0:
                        print 'WARNING **** BW error 3 %f %f'%(Qmax, Qtheta)
        
                # the log likelihood of newTheta should be higher by at least qDiff
                # (this is the inequality you get in the standard proof showing EM converges to a local maximum)
                bound = exp.logL + qDiff
                
                # update theta
                theta = newTheta
                                
        if trueTheta != None:
                print 'log-likelihood under true theta: %f'%trueL
        
        print 'Baum-Welch done.'        
        return theta
                
