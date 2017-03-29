__author__ = 'Guy Amster'

import argparse
import sys
import numpy as np
import re
import math
from collections import defaultdict
from scipy.optimize import minimize

# specify flags & usage
parser = argparse.ArgumentParser(description='Simulate a time-dependent mutational process. \n (output positions are 0-based). ')
parser.add_argument('u', type=float,
                    help='Mutation rate at time 0 (per 4N_0 generations per-bp)')
parser.add_argument('-uT', dest='uT', metavar = ('time','u'), action='append', nargs=2, type=float, 
                   help='Assume a mutation rate u from \'time\' backwards (time measured in units of 4N_0, as in ms)')
parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help='Input trees as generated by ms. By default, input is read stdin.')
parser.add_argument('-c', nargs=1, type=int, default=[0],
                    help='First chromosome number (for output format; default = 0; for multiple input trees, will be incremented by 1)')
parser.add_argument('-r', action='store_true',
                    help='Estimate r (per-bp in units of 4N0) from input trees. When using this flag, no mutations will be generated.')


# read input flags
args = parser.parse_args()


# model the substitutuion rate as a piecewise linear function of time
class piecewiseLinearU(object):
    def __init__(self, u0, uT):
        if u0 <= 0:
            raise ValueError('negative input value is invalid')
        
        # self.times: list of interval boundaries (ie times in which the rate changes)
        # self.rates: per-gen mutation rates in each of these intervals
        # self.sumRates: list of accumulated rates, from present to each of the intervals strating point
        
        self.times, self.sumRates, self.rates = [0.0], [0.0], [u0]
        
        if uT != None:
            # sort uT  by time, and verify that all inputs are positive
            uT = sorted(uT,key=lambda l: l[0])
            for x in uT:
                for y in x:
                    if y <= 0:
                        raise ValueError('negative input value is invalid')
            
            self.times = self.times + [x[0] for x in uT]
            self.rates = self.rates + [x[1] for x in uT]
            for i in xrange(len(uT)):
                self.sumRates.append(self.sumRates[-1] + (self.times[i+1]-self.times[i])*self.rates[i])
    
    # return the total expected number of substitutions from time t to present    
    def subRate(self, t):
        if t < 0:
            raise ValueError('negative input value is invalid')
        if t == 0:
            return 0.0
        # find i such that times[i] < t <= times[i+1]
        i = np.searchsorted(self.times, t) - 1
        return self.sumRates[i] + (t - self.times[i])*self.rates[i]


# read tree structure from input:

# split independent samples based on the '//' line preceeding a new sample
rawSamples, samples = args.infile.read().split("//\n")[1:], []

# regex pattern for ms tree structure of a sample of two, e.g. '[17](1:0.203,2:203);'
pattern = re.compile(r"""
                     \[(?P<width>\d+)\]                        # [width]
                     \(1:(?P<depth>\d*\.\d+),2:(?P=depth)\);   # (1:depth,2:depth);
                     $                                         # end of line
                     """, re.VERBOSE)

# parse input
for rawSample in rawSamples:
    sample = []
    for line in rawSample.split():
        match = pattern.match(line)
        if match == None:
            raise ValueError('tree input invalid')
        tree = (int(match.group("width")), float(match.group("depth")))
        assert tree[0] >  0
        # note depth = 0.0 is possible due to limited output precision from ms (-p flag doesn't work with -T...)
        assert tree[1] >= 0

        sample.append(tree)
    samples.append(sample)

# simulate mutations
if not args.r:
    # use mutational parameters from input            
    mutationalModel = piecewiseLinearU(args.u, args.uT)

    res = []
    for sample in samples:
        # list of hetrozygous sites (0-based positions)
        treeStart, muts = 0, []
        chrSize         = 0
        for tree in sample:
            width, depth =  tree
            chrSize      += width
            i = 0
            if depth > 0.0:
                while i < width:
                    # probability of mutation per-bp in this tree
                    p = 1 - np.exp(-2.0 * mutationalModel.subRate(depth))
                    assert p >  0.0
                    assert p <= 1.0
                    # find the next mutation in this tree (if exists) and log the position
                    i = i + np.random.geometric(p)
                    if i <= width:
                        muts.append(treeStart+i-1)
                
            treeStart = treeStart + width
        res.append((muts, chrSize))
    
    # write output
    chNum = args.c[0]
    # NOTE all positions are 0-based
    
    for x in res:
        muts, chrSize = x
        with open('muts%d.txt'%chNum, 'w') as f:
            if len(muts) > 0:
                f.write('%d\t%d\t%d\tAC\n'%(chNum, muts[0], muts[0]+1))
                for i in xrange(1,len(muts)):
                    f.write('%d\t%d\t%d\tAC\n'%(chNum, muts[i], muts[i]-muts[i-1]))
                # also print last position...
                if muts[-1] != chrSize:
                    assert muts[-1] < chrSize
                    f.write('%d\t%d\t%d\tAA\n'%(chNum, chrSize - 1, chrSize - 1 - muts[-1]))
            else:
                sys.stderr.write('Warning: No mutations were generated.\nOutput file is empty.\n')
        chNum += 1

# estimate r:
else:
    d = defaultdict(int)
    A = 0.0
    R = []
    for sample in samples:
        previousT, previousW = None, None
        for tree in sample:
            width, depth =  tree
            A -= ( depth * (float(width) - 1.0) )
            if previousT != None:
                if previousT == 0.0:
                    previousT = 0.00005
                d[previousT] += 1
                R.append(float(previousW)*previousT)
            previousT = depth
            previousW = width
        
    def minusLogL(x):
        r = math.exp(x[0])
        res = 2.0*A*r
        for k, v in d.iteritems():
            res += float(v) * math.log(-math.expm1(-2.0*r*k))
        return (-res)
    
    print 'r estimate (based on avergae width*depth): %f'%(0.5/np.mean(R))
    x0 = (-4.0,)
    res = minimize(minusLogL, x0, method='TNC', tol=1e-10)
    res = math.exp(res.x[0])
    print 'MLE r estimate: ', res



    
