import itertools # TODO remove
import time
import numpy as np
from BaumWelchExpectation import BaumWelchExpectation
from ObservedSequence import ObservedSequence
from Parallel import writeOutput, runParallel
from GOF import GOF
from functools import partial

# HMM: a container class specifying a hidden Markov model.
class HMM(object):
    
    # nStates       : Number of hidden states.
    # nEmissions:   : Number of possible observations (not chain length, but the 'alphabet' length).
    # transitionMat : The transition matrix of the chain.
    # initialDit    : The distribution of the first state.
    # emissionMat   : The emission probabilities matrix.
    def __init__(self, nStates, nEmissions, transitionMat, initialDist, emissionMat):
        
        assert transitionMat.shape == (nStates, nStates)
        assert initialDist  .shape == (nStates)
        assert emissionMat  .shape == (nStates, nEmissions)
        
        self.nStates       = nStates
        self.nEmissions    = nEmissions
        self.transitionMat = transitionMat
        self.initialDist   = initialDist
        self.emissionMat   = emissionMat
    
    # prints contents to string
    def __str__(self):
        temp = '{0:<24}{1}\n'
        res  = temp.format('nStates:', self.nStates)
        res += temp.format('nEmissions:', self.nEmissions)
        res += temp.format('initialDist:', self.initialDist)
        temp = '{0:<24}'
        res += temp.format('transitionMat:') + ('\n'+temp.format('')).join(str(self.transitionMat).splitlines()) + '\n'
        res += temp.format('emissionMat:  ') + ('\n'+temp.format('')).join(str(self.emissionMat  ).splitlines()) + '\n'
        return res
        

# Implements Baum-Welsh algorithm for general HMMs.
class BaumWelch(object):
    
    # nStates       : Number of hidden states.
    # nEmissions:   : Number of possible observations (not chain length, but the 'alphabet' length).
    def __init__(nStates, nEmissions):
        self.nStates = nStates
        self.nEmissions = nEmissions
    
    # observations  : a list of ObservedSequence objects
    # nIterations   : number of BW iterations. # TODO is there's a standard stopping criteria?
    # trueTheta     : optional HMM instance; for simulated data, prints the log-likelihood of the data with the true parameters
    # initTheta     : optional HMM instance; a theta value to initiate the BW process from; default specified by BW._initTheta() below
    # gof           : list of paramters l for GOF statistics G_l. If empty, GOF statistics are not calculated.
    # TODO fix or remove gof
    def run(self, observations, nIterations, trueTheta = None, initTheta = None, gof = []):
        
        # initialize theta
        theta = initTheta
        if theta is None:
            theta = self._initTheta(observations)
                
        # we expect the log likelihood at the next iteration to be higher than this        
        bound = -np.inf
                
        # print model specifications (see __str__ below for details):
        writeOutput('Model specifications:', 'loop')
        writeOutput(self, 'loop')
        writeOutput('\n', 'loop')
        
        # statistics to be collected
        self._statsNames = ['logL', 'Q-Init', 'Q-Max']
        for l in gof:
            self._statsNames.append('G%d'%l)
        
        # initialize GOF classes
        if len(gof) > 0:
            start = time.time()
            gof = [GOF(model, observations, l) for l in gof]
            writeOutput('initialized gof statistics within %f seconds'%(time.time()-start))

        # print the log-likelihood of the data under the true parameters (if given; simulated data only)
        if trueTheta is not None:
            
            # use the forward-backward algorithm to calculate the log-lokelihood of the observed sequence under trueTheta
            trueL = self._parallelExp(trueTheta, observations).logL
            
            # log True theta vals and statistics
            self._logVals('True parameters:', trueTheta, [trueL, '.', '.'], gof, target='DBG')
                
        for i in xrange(nIterations):
                
            writeOutput('starting BW iteration number %d'%(i + 1))
            
            # BW expectation step
            start = time.time()
            inferredHiddenState = self._parallelExp(theta, observations)
            writeOutput('finshed BW exp step within %f seconds'%(time.time()-start))
            
            # sanity check: log(O|theta) has increased as expected in the last iteration
            if inferredHiddenState.logL < bound:
                writeOutput('WARNING **** BW error 1 %f %f'%(inferredHiddenState.logL, bound), 'ErrorLog')
            
            # sanity check (this is just Jensen's inequality... Q(theta | theta) = E( log(P(O,Z|theta) ) <= log( E(P(O,Z|theta)) ) = log( P(O|theta) ) 
            Qtheta = self._Q(theta, inferredHiddenState)
            if Qtheta > inferredHiddenState.logL:
                writeOutput('WARNING **** BW error 2 %f %f'%(Qtheta, inferredHiddenState.logL), 'ErrorLog')

            # maximization step
            start = time.time()
            newTheta, Qmax = self._maximizeQ(inferredHiddenState, theta)
            writeOutput('finshed BW max step within %f seconds'%(time.time()-start))

            # sanity check: max_thetaStar Q(thetaStar | theta) >= Q(theta | theta)
            qDiff = Qmax - Qtheta
            if qDiff < 0:
                writeOutput('WARNING **** BW error 3 %f %f'%(Qmax, Qtheta), 'ErrorLog')
    
            # the log likelihood of newTheta should be higher by at least qDiff
            # (this is the inequality you get in the standard proof showing EM converges to a local maximum)
            bound = inferredHiddenState.logL + qDiff
            
            # sanity check for simulated data: verify that Qmax > Q(truetheta); This just helps convince us that the maximizer did converge.
            if trueTheta is not None:
                QTrue = self._Q(trueTheta, inferredHiddenState)
                if QTrue > Qmax:
                    writeOutput('WARNING **** BW error 4 %f %f'%(QTrue, Qmax), 'ErrorLog')
            
            # log iteration
            self._logVals('After %d iterations:'%i, theta, [inferredHiddenState.logL, Qtheta, Qmax], gof)

            # update theta
            theta = newTheta
                                
        # log final value of theta (for which some statistics are not calculated)
        self._logVals('After %d iterations:'%nIterations, theta, ['.', '.', '.'], gof)

    
    # calculate Q(theta* | theta) (ie evaluation for EM maximization step)
    # where theta are the parameters that were used for the inference of hiddenStates (a HiddenSeqSummary instance).
    # Q = E( log( P(hidden-state sequence Z, observations O | theta* ) ) ), where
    #     the expactation is over the posterior distribution of Z conditioned on theta (ie Z ~ P(Z | O, theta) )
    # This is really just standard EM stuff.
    @staticmethod
    def _Q(thetaStar, hiddenState):
        
        res  = np.sum(np.log(thetaStar.transitionMat) * hiddenState.transitions)
        res += np.sum(np.log(thetaStar.emissionMat  ) * hiddenState.emissions  )
        res += np.sum(np.log(thetaStar.initialDist  ) * hiddenState.gamma0     )
        return res
    
    # calculate theta* that maximizes Q (ie EM maximization step).
    # returns (theta*, attained maximum value).
    # here we use the closed form for the global maximum (so initTheta is ignored); in the derived class we use initTheta
    # TODO nStartPoints 290?
    def _maximizeQ(self, hiddenState, initTheta):
        transitions = np.empty( (self.nStates, self.nStates) )
        for i in xrange(self.nStates):
            transitions[i,:] = hiddenState.transitions[i, :] / np.sum(hiddenState.transitions[i, :])
    
        emissions = np.empty( (self.nStates, self.nEmissions) )
        for i in xrange(self.nStates):
            emissions[i,:] = hiddenState.emissions[i, :] / np.sum(hiddenState.emissions[i, :])
        
        initial = hiddenState.gamma0 / np.sum(hiddenState.gamma0)
        
        maxTheta = HMM(self.nStates, self.nEmissions, transitions, initial, emissions)
        return maxTheta, self._Q(maxTheta)
    
    # expectation step
    def _parallelExp(self, theta, observations):
    
        # run expectation step for each sequence separately (in parallel)
        res = runParallel(runMemberFunc(self, '_calcExp'), [(theta, o) for o in observations])
        
        # combine results (__add__ is overriden by HiddenSeqSummary)
        return reduce(lambda x, y: x+y, res)
    
    # expectation step on a single observation
    @staticmethod
    def _calcExp(inp):
        theta, obs  = inp
        return BaumWelchExpectation(theta, obs).inferHiddenStates()
        
    def __str__(self):
        temp = '\t{0:<24}{1:<24}\n'
        res = temp.format('nStates:',self.nStates) + temp.format('nEmissions:',self.nEmissions)
        return res

    
    # log iteration results
    def _logVals(self, header, theta, stats, gof, target = 'loop'):
        # print header
        writeOutput(header,target)
        writeOutput('\n',target)
        
        # print theta
        writeOutput(theta, target)
        writeOutput('\n',target)
        
        # calculate gof stats
        if len(gof) > 0:
            start = time.time()
            for c in gof:
                    stats.append(c.G(theta))
            writeOutput('calculated gof statistics within %f seconds'%(time.time()-start))        
        assert len(self._statsNames) == len(stats)

        temp = '\t'
        for i in xrange(len(statsNames)):
                temp += '{%d:<24}'%i
        writeOutput('Statistics:',target)
        writeOutput(temp.format(*self._statsNames),target)
        writeOutput(temp.format(*stats),target)
        writeOutput('\n',target)
    
    # initialize theta with random values;
    # (here _initTheta doesn't use observations, but it does in the derived class)
    def _initTheta(self, observations):
        initialDist   = np.random.dirichlet(np.ones(self.nStates))
        transitionMat = np.empty( (self.nStates, self.nStates   ) )
        emissionMat   = np.empty( (self.nStates, self.nEmissions) )
        
        for i in xrange(self.nStates):
            transitionMat[i,:] = np.random.dirichlet(np.ones(self.nStates   ))
            emissionMat  [i,:] = np.random.dirichlet(np.ones(self.nEmissions))
        
        return HMM(self.nStates, self.nEmissions, transitionMat, initialDist, emissionMat)
    
# TODO name
class MSMCU(BaumWelch):
    
    # lmbPattern : [0,1,2] stands for 3 segments with independent coal. rates; [0,0,1] stand for 3 segments, where the first two have equal coal. rates.
    # uPattern   : same for mutation rate (e.g. [0,0,...,0] assumes fixed mutation rate; [0,1,2,...] assumes independent mutation rates)
    # scale      : either '2N0', 'u0' or 'r'. Sets the unit by which results are scaled (e.g. 'u0' sets u0 to 1.0).
    #              (Note you can scale by any parameter; e.g. uPattern = [1,0,2,...] and scaleBy = 'u0' would scale by the mutation rate in the second interval)
    def __init__(self, lmbPattern, uPattern, scale):
        
        # verify that exactly one is set to true
        assert scale in ['2N0', 'u0', 'r']
        
        # verify input validity
        nStates = len(lmbPattern)
        assert nStates == len(uPattern)
        for pattern in [lmbPattern, uPattern]:
            for i in xrange(len(set(pattern))):
                assert i in pattern
        
        self._scale      = scale
        self._lmbPattern = lmbPattern
        self._uPattern   = uPattern
        BaumWelch.__init__(nStates, 2)
        
        # number of parameters (accounting for pattern) and free parameters (also accounting for scale)
        self._nParamsLmb     = len(set(lmbPattern))
        self._nParamsU       = len(set(uPattern  ))
        self._nFreeParamsLmb = self._nParamsLmb - (scale == '2N0')
        self._nFreeParamsU   = self._nParamsU   - (scale == 'u0' )
        self._nFreeParamsR   = 1                - (scale == 'r'  )
        
        # total number of free parameters
        self._nFreeParams = self._nParamsLmb + self._nParamsU
        
        # inverse pattern
        self._uInvPattern   = [  uPattern.index(v) for v in xrange(self._nFreeParamsU  )]
        self._lmbInvPattern = [lmbPattern.index(v) for v in xrange(self._nFreeParamsLmb)]
    
    def _maximizeQ(self, hiddenState, initTheta):
        pass #B
    
    # Describe __init__ flags in string
    def __str__(self):
        template = '\t{0:<24}{1}\n'
        lmbPattern, uPattern, scale
        res  = template.format('lmbPattern', self._lmbPattern)
        res += template.format('uPattern', self._uPattern)
        res += template.format('scale', self._scale)
        
        return res
    
    # Construct class from string
    @classmethod
    def fromString(cls, inp):
        # TODO
        return cls(lmbPattern, uPattern, scale)
    
    # we initialize the Baum-Welch algorithm with a population of fixed size N, fixed mutation rate u and recombination rate r = u.
    # we choose N and u such that 4Nu = pi. 
    def _initTheta(self, observations):
        
        # calculate pi
        # TODO support missing sites, different site types etc
        het, length = 0, 0
        for obs in observations:
                length += obs.length
                het    += np.count_nonzero(obs.posTypes)
        pi = float(het)/float(length)
        
        # determine u, r, lmb in scaled units
        # (we assume r=u, and notice pi = 4Nu = 2u/lmb)
        if self._scale == '2N0':
            u, r, lmb = pi/2, pi/2, 1.0
        else:
            u, r, lmb = 1.0, 1.0, 2/pi
        
        # create vector
        vec = [u]*self._nFreeParamsU + [lmb]*self._nFreeParamsLmb + [r]*self._nFreeParamsR
        
        # translate to Theta
        return self._vecToTheta(np.log(vec))
    
    # expand a vector of unconstrained parameters to a Theta instance.
    # vec = log(u_0), ..., log(u_(nFreeParamsU-1)) || log(lambda_0), ..., log(lambda_(nFreeParamsLmb-1)) || log(r).
    # the scale parameter (u0, r or N0) is missing from the vector.
    def _vecToTheta(self, vec):
        
        assert len(vec) == self._nFreeParams
        
        # take exponent
        vec = np.exp(vec)
        
        # extract free parameters and add constant scale parameter
        uVec   =  [1.0][:self._scale == 'u0' ] + vec[:self._nFreeParamsU]
        lmbVec =  [1.0][:self._scale == '2N0'] + vec[self._nFreeParamsU:self._nFreeParamsU+self._nFreeParamsLmb]
        r      = ([1.0][:self._scale == 'r'  ] + vec[self._nFreeParamsU+self._nFreeParamsLmb:])[0]
        
        # calculate values based on patterns
        uVals   = [  uVec[self.  _uPattern[i]] for i in self.nStates]
        lmbVals = [lmbVec[self._lmbPattern[i]] for i in self.nStates]
        
        # boundaries are chosen such that the probability of coalescence at state i is 1/nStates
        boundaries = np.zeros(self.nStates + 1)
        for i in xrange(self.nStates - 1):
            delta = -math.log1p(-1.0/(self.nStates - i)) / lmbVals[i]
            boundaries[i+1] = boundaries[i] + delta
        boundaries[-1] = np.inf
        
        writeOutput(Theta(boundaries, lmbVals, uVals, r).initialDist, "DBG")
        return Theta(boundaries, lmbVals, uVals, r)
    
    # shrink a theta instance generated by _vecToTheta back to vector of free parameters 
    # (this is a left inverse function to _vecToTheta;
    #  we do not verify input validity: if theta isn't in the range of _vecToTheta, e.g. is not scaled, has other boundaries,
    #  or doesn't follow the pattern, we won't raise an exception).
    def _thetaToVec(self, theta):
        
        # 'shrink' vectors based on patterns, also removing the scale parameter
        u   = [theta.uVals[self._uInvPattern  [i]] for i in xrange(self._scale == 'u0' , self._nParamsU      )]
        lmb = [theta.uVals[self._lmvInvPattern[i]] for i in xrange(self._scale == '2N0', self._nParamsLmb    )]
        r   = [theta.r][self._scale == 'r' :]
        
        # concatenate values
        vec = np.append(u, np.append(lmb ,r))
        
        # take log and return
        return np.log(vec)


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




                        

                
                
