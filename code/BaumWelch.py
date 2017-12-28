import itertools # TODO remove
import time
import numpy as np
from Containers import HmmModel, HmmTheta, Theta
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
from Parallel import writeOutput, runParallel
from GOF import GOF
from functools import partial
from history import writeHistory



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

def _calcExp(obs, model, theta):
        return BaumWelchExpectation(model, theta, obs).inferHiddenStates()

def _parallelExp(model, theta, observations):
        
        res = runParallel(partial(_calcExp, model=model, theta=theta), observations)
        
        resSum = res[0]
        for i in xrange(1, len(res)):
                resSum = resSum + res[i]
        return resSum

def _logVals(header, theta, statsNames, stats, gof):
        # print header
        writeOutput(header,'loop')
        writeOutput('\n','loop')
        
        # print theta
        # TODO replace with object method... also, _model
        N_boundaries = theta._model.segments.boundaries
        if theta._model.fixedMu:
                u_boundaries = [0.0, np.inf]
                u_vals = [theta.uV[0]]
        else:
                u_boundaries = N_boundaries
                u_vals = theta.uV
        writeOutput(writeHistory(theta.r, u_boundaries, u_vals, N_boundaries, [0.5/x for x in theta.lambdaV]), 'loop')
        writeOutput('\n','loop')
        
        # calculate gof stats
        if len(gof) > 0:
                start = time.time()
                for c in gof:
                        stats.append(c.G(theta))
                writeOutput('calculated gof statistics within %f seconds'%(time.time()-start))        
        assert len(statsNames) == len(stats)

        temp = '\t'
        for i in xrange(len(statsNames)):
                temp += '{%d:<24}'%i
        writeOutput('Statistics:','loop')
        writeOutput(temp.format(*statsNames),'loop')
        writeOutput(temp.format(*stats),'loop')
        writeOutput('\n','loop')

# TODO remove
'''
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
                writeOutput('calculated gof statistics within %f seconds'%(time.time()-start))
        writeOutput('\t'.join([str(v) for v in vals]), 'loop')
'''
        
# model         : a (derived) HmmModel class
# observations  : a list of ObservedSequence objects
# nIterations   : number of BW iterations. # TODO is there's a standard stopping criteria?
# trueTheta     : for simulated data (will be used for printing statistics)
# theta         : a theta value to initiate the BW process from; default is to use a random theta.
# gof           : List of paramters l for GOF statistics G_l. If empty, GOF statistics are not calculated.
def BaumWelch(model, observations, nIterations = 20, trueTheta = None, theta = None, gof = []):
        
                        
        # initialize theta
        if theta is None:
                if model.modelType == 'basic':
                        theta = HmmTheta.random(model)
                else:
                        theta =    Theta(model)
                     
                
        # we expect the log likelihood at the next iteration to be higher than this        
        bound = -np.inf
                
        # print model specifications:
        writeOutput('Model specifications:', 'loop')
        writeOutput(model.printVals() + '\n', 'loop')
        
        # statistics to be collected
        statsNames = ['logL', 'Q-Init', 'Q-Max']
        for l in gof:
                statsNames.append('G%d'%l)
        
        # initialize GOF classes
        if len(gof) > 0:
                start = time.time()
                gof = [GOF(model, observations, l) for l in gof]
                writeOutput('initialized gof statistics within %f seconds'%(time.time()-start))

        
        if trueTheta is not None:
                trueL = _parallelExp(model, trueTheta, observations).logL
                
                # log True theta vals and statistics
                _logVals('True parameters:', trueTheta, statsNames, [trueL, '.', '.'], gof)
                
        for i in xrange(nIterations):
                
                writeOutput('starting BW iteration number %d'%(i + 1))
                
                # BW expectation step
                start = time.time()
                exp  = _parallelExp(model, theta, observations)
                writeOutput('finshed BW exp step within %f seconds'%(time.time()-start))
                
                # sanity check: log(O|theta) has increased as expected in the last iteration
                if exp.logL < bound:
                        writeOutput('WARNING **** BW error 1 %f %f'%(exp.logL, bound), 'ErrorLog')
                
                # sanity check (this is just Jensen's inequality... Q(theta | theta) = E( log(P(O,Z|theta) ) <= log( E(P(O,Z|theta)) ) = log( P(O|theta) ) 
                Qtheta = exp.Q(theta)
                if Qtheta > exp.logL:
                        writeOutput('WARNING **** BW error 2 %f %f'%(Qtheta, exp.logL), 'ErrorLog')

                # maximization step
                start = time.time()
                newTheta, Qmax = exp.maximizeQ(initTheta = theta)
                writeOutput('finshed BW max step within %f seconds'%(time.time()-start))

                # sanity check: max_thetaStar Q(thetaStar | theta) >= Q(theta | theta)
                qDiff = Qmax - Qtheta
                if qDiff < 0:
                        writeOutput('WARNING **** BW error 3 %f %f'%(Qmax, Qtheta), 'ErrorLog')
        
                # the log likelihood of newTheta should be higher by at least qDiff
                # (this is the inequality you get in the standard proof showing EM converges to a local maximum)
                bound = exp.logL + qDiff
                
                if trueTheta is not None:
                        QTrue = exp.Q(trueTheta)
                        if QTrue > Qmax:
                                writeOutput('WARNING **** BW error 4 %f %f'%(exp.Q(trueTheta), Qmax), 'ErrorLog')
                
                # log iteration
                _logVals('After %d iterations:'%i, theta, statsNames, [exp.logL, Qtheta, Qmax], gof)

                # update theta
                theta = newTheta
                                
        # log final value of theta (for which some statistics are not calculated)
        _logVals('After %d iterations:'%nIterations, theta, statsNames, ['.', '.', '.'], gof)

        
        return theta
                
                
