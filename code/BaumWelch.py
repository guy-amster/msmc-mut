import itertools # TODO remove
import time
import numpy as np
from Containers import HmmModel, HmmTheta, Theta
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
from multiprocessing import Pool
from Logger import log, logError
from GOF import GOF


# TODO MOVE OR REMOVE
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

# TODO move to exp?
def _parallelExp(model, theta, observations, nPrc):
        
        inp = [{'model':model, 'theta':theta, 'obs':obs} for obs in observations]
        
        p   = Pool(nPrc)
        res = p.map(_calcExp, inp)       
        p.close()
        
        resSum = res[0]
        for i in xrange(1, len(res)):
                resSum = resSum + res[i]
        return resSum

def _logVals(i, theta, logL, QInit, QMax, gof):
        vals = [i]
        vals.append(','.join([str(x) for x in theta.lambdaV]))
        vals.append(theta.r)
        vals.append(','.join([str(x) for x in theta.uV]))
        vals += [logL, QInit, QMax]
        if gof is not None:
                start = time.time()
                for c in gof:
                        vals.append(c.G(theta))
                log('calculated gof statistics within %f seconds'%(time.time()-start))
        log('\t'.join([str(v) for v in vals]), filename='loop')
        
# model         : a (derived) HmmModel class
# observations  : a list of ObservedSequence objects
# nProcesses    : number of processors to parallelize on.
# nIterations   : number of BW iterations. # TODO is there's a standard stopping criteria?
# trueTheta     : for simulated data (will be used for printing statistics)
# theta         : a theta value to initiate the BW process from; default is to use a random theta.
# gof           : List of paramters l for GOF statistics G_l. If none, GOF statistics are not calculated.
def BaumWelch(model, observations, nProcesses = 1, nIterations = 20, trueTheta = None, theta = None, gof = None):
        
                        
        # initialize theta
        if theta is None:
                if model.modelType == 'basic':
                        theta = HmmTheta.random(model)
                else:
                        theta =    Theta(model)
                     
                
        # we expect the log likelihood at the next iteration to be higher than this        
        bound = -np.inf
                
        # log column names.
        colNames = ['iter', 'lambda', 'r', 'u', 'logL', 'Q-Init', 'Q-Max']
        if gof is not None:
                for l in gof:
                        colNames.append('G%d'%l)
        log('\t'.join(colNames), filename = 'loop')
        
        if gof is not None:
                start = time.time()
                gof = [GOF(model, observations, l, nProcesses) for l in gof]
                log('initialized gof statistics within %f seconds'%(time.time()-start))

        
        if trueTheta is not None:
                trueL = _parallelExp(model, trueTheta, observations, nProcesses).logL
                
                # log True theta vals
                _logVals('T', trueTheta, trueL, '.', '.', gof)
                
        for i in xrange(nIterations):
                
                log('starting BW iteration number %d'%(i + 1))
                
                # BW expectation step
                start = time.time()
                exp  = _parallelExp(model, theta, observations, nProcesses)
                log('finshed BW exp step within %f seconds'%(time.time()-start))
                
                # sanity check: log(O|theta) has increased as expected in the last iteration
                if exp.logL < bound:
                        logError('WARNING **** BW error 1 %f %f'%(exp.logL, bound))
                
                # print statistics for simulated data
                if trueTheta is not None:
                        # TODO EITHER REMOVE OR MOVE DIST TO THETA CLASSES
                        #x = _dist(trueTheta, theta)
                        #print 'distance from true theta: ', x
                        pass
                
                # sanity check (this is just Jensen's inequality... Q(theta | theta) = E( log(P(O,Z|theta) ) <= log( E(P(O,Z|theta)) ) = log( P(O|theta) ) 
                Qtheta = exp.Q(theta)
                if Qtheta > exp.logL:
                        logError('WARNING **** BW error 2 %f %f'%(Qtheta, exp.logL))

                # maximization step
                start = time.time()
                newTheta, Qmax = exp.maximizeQ(nProcesses = nProcesses, initTheta = theta)
                log('finshed BW max step within %f seconds'%(time.time()-start))

                # sanity check: max_thetaStar Q(thetaStar | theta) >= Q(theta | theta)
                qDiff = Qmax - Qtheta
                if qDiff < 0:
                        logError('WARNING **** BW error 3 %f %f'%(Qmax, Qtheta))
        
                # the log likelihood of newTheta should be higher by at least qDiff
                # (this is the inequality you get in the standard proof showing EM converges to a local maximum)
                bound = exp.logL + qDiff
                
                if trueTheta is not None:
                        QTrue = exp.Q(trueTheta)
                        if QTrue > Qmax:
                                logError('WARNING **** BW error 4 %f %f'%(exp.Q(trueTheta), Qmax))
                
                # log iteration
                _logVals(i, theta, exp.logL, Qtheta, Qmax, gof)

                # update theta
                theta = newTheta
                                
        # log final value of theta (for which some statistics are not calculated)
        _logVals(nIterations, theta, '.', '.', '.', gof)
        
        return theta
                
                
