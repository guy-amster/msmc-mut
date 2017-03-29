from Containers import HmmModel, HmmTheta, Theta
import numpy as np
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
import itertools
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

def _parallelExp(model, theta, observations, nPrc):
        
        if nPrc == 1:
                res = [BaumWelchExpectation(model, theta, obs).inferHiddenStates() for obs in observations]
        
        else:
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
                for c in gof:
                        vals.append(c.G(theta))
        log('\t'.join([str(v) for v in vals]))
        
# model         : a (derived) HmmModel class
# observations  : a list of ObservedSequence objects
# nProcesses    : number of processors to parallelize on.
# nIterations   : number of BW iterations. # TODO is there's a standard stopping criteria?
# trueTheta     : for simulated data (will be used for printing statistics)
# theta         : a theta value to initiate the BW process from; default is to use a random theta.
# gof           : List of paramters l for GOF statistics G_l. If none, GOF statistics are not calculated.
def BaumWelch(model, observations, nProcesses = 1, nIterations = 20, trueTheta = None, theta = None, gof = None):
        
                        
        # initialize theta
        if theta == None:
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
        log('\t'.join(colNames))
        
        if gof is not None:
                gof = [GOF(model, observations, l) for l in gof]

        
        if trueTheta != None:
                trueL = _parallelExp(model, trueTheta, observations, nProcesses).logL
                
                # log True theta vals
                _logVals('T', trueTheta, trueL, '.', '.', gof)
                
        for i in xrange(nIterations):
                
                print 'starting BW iteration number %d\n'%(i + 1)
                
                # BW expectation step
                exp  = _parallelExp(model, theta, observations, nProcesses)
                
                # sanity check: log(O|theta) has increased as expected in the last iteration
                if exp.logL < bound:
                        logError('WARNING **** BW error 1 %f %f'%(exp.logL, bound))
                
                # print statistics for simulated data
                if trueTheta != None:
                        # TODO EITHER REMOVE OR MOVE DIST TO THETA CLASSES
                        #x = _dist(trueTheta, theta)
                        #print 'distance from true theta: ', x
                        pass
                
                # sanity check (this is just Jensen's inequality... Q(theta | theta) = E( log(P(O,Z|theta) ) <= log( E(P(O,Z|theta)) ) = log( P(O|theta) ) 
                Qtheta = exp.Q(theta)
                if Qtheta > exp.logL:
                        logError('WARNING **** BW error 2 %f %f'%(Qtheta, exp.logL))

                # maximization step
                newTheta, Qmax = exp.maximizeQ(nProcesses = nProcesses, initTheta = theta)

                # sanity check: max_thetaStar Q(thetaStar | theta) >= Q(theta | theta)
                qDiff = Qmax - Qtheta
                if qDiff < 0:
                        logError('WARNING **** BW error 3 %f %f'%(Qmax, Qtheta))
        
                # the log likelihood of newTheta should be higher by at least qDiff
                # (this is the inequality you get in the standard proof showing EM converges to a local maximum)
                bound = exp.logL + qDiff
                
                if trueTheta != None:
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
                
                
