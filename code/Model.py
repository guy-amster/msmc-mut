import numpy as np
import re
import math
from BaumWelch import BaumWelch
from Parallel import writeOutput, runParallel, runMemberFunc
from scipy.optimize import minimize
from Theta import Theta
import scipy.stats as ss

# TODO name & doc
# This is the upper-level class implementing our model.
# Initialize with Model(...) (see below) and run model with .run(...) (see base class).
# Here we only implement the maximization step, as the other details are identical to the standard BW algorithm and implemented there.
class Model(BaumWelch):
    
    # lmbPattern : [0,1,2] stands for 3 segments with independent coal. rates; [0,0,1] stand for 3 segments, where the first two have equal coal. rates.
    # uPattern   : same for mutation rate (e.g. [0,0,...,0] assumes fixed mutation rate; [0,1,2,...] assumes independent mutation rates)
    # scale      : either '2N0', 'u0' or 'r'. Sets the unit by which results are scaled (e.g. 'u0' sets u0 to 1.0).
    #              (Note you can scale by any parameter; e.g. uPattern = [1,0,2,...] and scaleBy = 'u0' would scale by the mutation rate in the second interval)
    # TODO remove or doc boundaries; change 'scale' to 'scaleUnit' or whatever I'm using in Theta.scale(); '2N0' is confusing - use lmb0
    def __init__(self, lmbPattern, uPattern, scale, fixedBoundaries = None):
        
        # verify that exactly one is set to true
        # TODO remove or doc none 
        assert scale in ['2N0', 'u0', 'r', 'none']
        
        # verify input validity
        nStates = len(lmbPattern)
        assert nStates == len(uPattern)
        for pattern in [lmbPattern, uPattern]:
            for i in xrange(len(set(pattern))):
                assert i in pattern
        
        self.scale            = scale
        self._lmbPattern      = lmbPattern
        self._uPattern        = uPattern
        self._fixedBoundaries = fixedBoundaries
        BaumWelch.__init__(self, nStates, 2)
        
        # number of parameters (accounting for pattern) and free parameters (also accounting for scale)
        self._nParamsLmb     = len(set(lmbPattern))
        self._nParamsU       = len(set(uPattern  ))
        self._nFreeParamsLmb = self._nParamsLmb - (scale == '2N0')
        self._nFreeParamsU   = self._nParamsU   - (scale == 'u0' )
        self._nFreeParamsR   = 1                - (scale == 'r'  )
        
        # total number of free parameters
        self._nFreeParams = self._nFreeParamsLmb + self._nFreeParamsU + self._nFreeParamsR
        
        # inverse pattern
        self._uInvPattern   = [  uPattern.index(v) for v in xrange(self._nFreeParamsU  )]
        self._lmbInvPattern = [lmbPattern.index(v) for v in xrange(self._nFreeParamsLmb)]
    
    # calculate theta* that maximizes Q (ie EM maximization step).
    # returns theta* and the attained maximum value.
    # Here we find the maximum numerically, as there's no closed-form solution.
    # As starting points for the numerical optimizer, we use:
    #   (A) random start points
    #   (B) estimates of theta from previous iterations
    def _maximizeQ(self, hiddenState, initThetas):
        
        # TODO nStartPoints 290?
        # number of start points
        nStartPoints = 60
        
        # TODO remove
        # -Q is a positive measure we're trying to minimize... 
        refs = [-self._Q(t, hiddenState) for t in initThetas]
        for r in refs :
            assert r > 0
        
        # TODO move 'initTheta' to class field and see if it screws up performance.
        # initial points for optimizer; None is later converted to random init point
        # inputs = [self._thetaToVec(theta) for theta in initThetas] + [None for _ in xrange(nStartPoints - len(initThetas))]
        # inputs = [(x0, hiddenState) for x0 in inputs]
        inputs = [(self._thetaToVec(theta), hiddenState) for theta in [initThetas[0], initThetas[-1]]]
        
        # run self._maxQSingleStartPoint() on all items in inputs
        # Note: using partial(runMemberFunc, ...) to overcome Pool.map limitations on class methods.
        res = runParallel(runMemberFunc(self, '_maxQSingleStartPoint'), inputs)
            
        maxFound = -np.inf
        indices = []
        for i in xrange(len(res)): # TODO return to for theta, val in res
            theta, val = res[i]
            if val > maxFound:
                maxFound = val
                maxTheta = theta
            
            assert val < 0.0
            indices.append('{0}:{1}'.format(i, -val/refs[-1]))
        
        # TODO remove
        writeOutput('reference vals: ' + ','.join(str(r/refs[-1]) for r in refs), filename = 'DBG')
        writeOutput('indices: ' + ','.join(str(v) for v in ss.rankdata(indices)), filename = 'DBG')
    
        return maxTheta, maxFound
    
    # calculate theta* that maximizes Q numerically, given a specific init-point x0 for the optimizer
    def _maxQSingleStartPoint(self, inp):
        
        x0, hiddenState = inp
        if x0 is None:
            # our random start point for each parameter is def-value * e^r where r~uniform( -log(100), log(100) );
            # where the def-value corresponds to fixed population size & mutation rates
            # TODO 8? 100? at least have it symetric! (uniform in the log space...)
            ru = np.random.uniform(-np.log(0.5), np.log(2), self._nFreeParamsU)
            rl = np.random.uniform(-np.log(0.125), np.log(8), self._nFreeParamsLmb)
            rr = np.random.uniform(-np.log(0.125), np.log(8), self._nFreeParamsR)
            rand = np.log(np.append(ru, np.append(rl, rr)))
            x0 = self._defVec + rand
                
        # TODO DOC; why 10? why like this? this is ridicilous... 
        QNull = self._Q(self._vecToTheta(self._defVec), hiddenState)*2
        assert QNull < 0.0
        bound = math.log(10)
        def fun(x):
            z = np.max(np.abs(x-self._defVec))
            if z >= bound:
                return -QNull
            return -self._Q(self._vecToTheta(x), hiddenState)
        
        #consts = [{'type': 'ineq', 'fun': lambda x:  math.log(10) + (x[i] - defVals[i])} for i in range(len(defVals))] \
        #        +[{'type': 'ineq', 'fun': lambda x:  math.log(10) - (x[i] - defVals[i])} for i in range(len(defVals))]
        # TODO what algorithm?
        op = minimize(fun,
                      x0,
                      #constraints=tuple(consts),
                      tol=1e-5, #TODO -7
                      #options={'disp': True, 'maxiter': 1000000}
                      options={'maxiter': 10000}#TODO 00}
                      )
        # TODO remove writeOutput('optimizer output (val: %f)\n'%-op.fun + str(Theta.fromUnconstrainedVec(self._model, op.x)),'DBG')
        
        return (self._vecToTheta(op.x), -op.fun)
    
    
    # Describe __init__ flags in string
    def __str__(self):
        template = '\t{0:<24}{1}\n'
        res  = template.format('lmbPattern:', self._lmbPattern)
        res += template.format('uPattern:', self._uPattern)
        res += template.format('scale:', self.scale)
        # TODO update with other __init__ inputs
        return res
    
    # Construct class from string
    @classmethod
    def fromString(cls, inp):
        pattern = re.compile(r"""
                                 ^
                                  \t lmbPattern: \ * \[(?P<lmbPattern> (\d+[,\ ]*)+  )\] \ * \n
                                  \t uPattern:   \ * \[(?P<uPattern>   (\d+[,\ ]*)+  )\] \ * \n
                                  \t scale:      \ *  (?P<scale>       2N0|u0|r      )   \ * \n
                                 $
                             """, re.VERBOSE)

        match = pattern.match(inp)
        assert match is not None
        
        lmbPattern = [int(x) for x in match.group("lmbPattern").split(', ')]
        uPattern   = [int(x) for x in match.group("uPattern"  ).split(', ')]
        scale      = match.group("scale")
        
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
        if self.scale == '2N0':
            u, r, lmb = pi/2, pi/2, 1.0
        else:
            u, r, lmb = 1.0, 1.0, 2/pi
        
        # create vector
        self._defVec = np.log([u]*self._nFreeParamsU + [lmb]*self._nFreeParamsLmb + [r]*self._nFreeParamsR)
        
        # create Theta object and return
        return self._vecToTheta(self._defVec)
    
    # expand a vector of unconstrained parameters to a Theta instance.
    # vec = log(u_0), ..., log(u_(nFreeParamsU-1)) || log(lambda_0), ..., log(lambda_(nFreeParamsLmb-1)) || log(r).
    # the scale parameter (u0, r or N0) is missing from the vector.
    def _vecToTheta(self, vec):
        
        assert len(vec) == self._nFreeParams
        
        # take exponent
        vec = np.exp(vec)
        
        # extract free parameters and add constant scale parameter
        uVec   = np.append([1.0][:self.scale == 'u0' ], vec[:self._nFreeParamsU])
        lmbVec = np.append([1.0][:self.scale == '2N0'], vec[self._nFreeParamsU:self._nFreeParamsU+self._nFreeParamsLmb])
        r      = np.append([1.0][:self.scale == 'r'  ], vec[self._nFreeParamsU+self._nFreeParamsLmb:])[0]
        
        # calculate values based on patterns
        uVals   = [  uVec[self.  _uPattern[i]] for i in xrange(self.nStates)]
        lmbVals = [lmbVec[self._lmbPattern[i]] for i in xrange(self.nStates)]
        
        boundaries = self._fixedBoundaries
        if boundaries is None:
            # boundaries are chosen such that the probability of coalescence at state i is 1/nStates
            boundaries = np.zeros(self.nStates + 1)
            for i in xrange(self.nStates - 1):
                delta = -math.log1p(-1.0/(self.nStates - i)) / lmbVals[i]
                boundaries[i+1] = boundaries[i] + delta
            boundaries[-1] = np.inf
        
        return Theta(boundaries, lmbVals, uVals, r)
    
    # shrink a theta instance generated by _vecToTheta back to vector of free parameters 
    # (this is a left inverse function to _vecToTheta;
    #  we do not verify input validity: if theta isn't in the range of _vecToTheta, e.g. is not scaled, has other boundaries,
    #  or doesn't follow the pattern, we won't raise an exception).
    def _thetaToVec(self, theta):
        
        # 'shrink' vectors based on patterns, also removing the scale parameter
        u   = [theta.uVals  [self._uInvPattern  [i]] for i in xrange(self.scale == 'u0' , self._nParamsU      )]
        lmb = [theta.lmbVals[self._lmbInvPattern[i]] for i in xrange(self.scale == '2N0', self._nParamsLmb    )]
        r   = [theta.r][self.scale == 'r' :]
        
        # concatenate values
        vec = np.append(u, np.append(lmb ,r))
        
        # take log and return
        return np.log(vec)
