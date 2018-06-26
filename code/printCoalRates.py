#!/usr/bin/env python

import math
import argparse
import numpy as np
from readUtils import readHist

# define input structure using argparse
parser = argparse.ArgumentParser(description='Calculate coalescence rates per segment  \n ')
parser.add_argument('filename', help='input history file')

# read input flags
args = parser.parse_args()

# read history
theta = readHist(args.filename)
collapsedTheta = theta.collapseSegments()

def coalRates(theta):
    logq = 0.0
    rates, cond = [], []
    for i in xrange(len(theta.uVals)):
        p = -math.expm1(-theta.lmbVals[i]*theta.segments.delta[i])
        cond.append(p)
        rates.append(p*math.exp(logq))
        logq += -theta.lmbVals[i]*theta.segments.delta[i]
        
    print '\tcond', cond
    print '\tstat', rates        

print 'full:'
coalRates(theta)
print 'collapsed:'
coalRates(collapsedTheta)   
        
