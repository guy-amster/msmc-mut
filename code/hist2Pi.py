#!/usr/bin/env python

import os
import fnmatch
import argparse
import numpy as np
from ObservedSequence import ObservedSequence
from readUtils import readHist

# define input structure using argparse
parser = argparse.ArgumentParser(description='Calculate pi for a specific history; or, from observations  \n ')
parser.add_argument('-hist', dest='hist', metavar = ('filename'), 
                   help='input history file')
parser.add_argument('-obs', dest='obs', metavar = ('filename'), nargs='+', 
                   help='input observations file')

# read input flags
args = parser.parse_args()

if args.hist is not None:
    
    # read history
    hist = readHist(args.hist)
    
    # calculate pi
    pi = hist.pi()

# TODO this code is a duplicate... both reading the files & calculating pi
# either throw this away (preferable) or move to a shared util location
elif args.obs is not None:
    
    # read observations
    # read input dir & match all input files...
    files = []
    for inpPattern in args.obs:
            pathName   = os.path.dirname(inpPattern)
            if pathName == '':
                    pathName = os.curdir
            for f in os.listdir(pathName):
                    if fnmatch.fnmatch(f, os.path.basename(inpPattern)):
                            files.append(pathName + '/' + f)
    
    # read all input files (executed in parallel)
    assert len(files) > 0
    observations = [ObservedSequence.fromFile(f) for f in files]
    
    # calculate pi
    het, length = 0, 0
    for obs in observations:
            length += obs.length
            het    += np.count_nonzero(obs.posTypes)
    pi = float(het)/float(length)
else:
    print 'use either -obs or -hist'
    exit()
print 'pi: ', pi