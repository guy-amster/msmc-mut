#!/usr/bin/env python
from Model import Model
from ObservedSequence import ObservedSequence
from BaumWelch import BaumWelch
from Parallel import initParallel, runParallel, runMemberFunc, writeOutput, OutputWriter
import numpy as np 
import os
import sys
import fnmatch
import argparse
import multiprocessing
import time
# TODO remove
from Theta import Theta
from readUtils import readHist


# TODO THIS IS TEMPORARY
def calcPi(observations):
        het, length = 0, 0
        # TODO support missing sites, different site types etc
        for obs in observations:
                length += obs.length
                het    += np.count_nonzero(obs.posTypes) # remove np import
        pi = float(het)/float(length)
        return pi


# specify flags & usage
parser = argparse.ArgumentParser(description='Infer population size and mutation rate histories. \n ')

parser.add_argument('-o', default='', metavar = ('prefix'), 
                    help='Prefix to use for all output files.')
parser.add_argument('-psmcSegs', action='store_true',
                    help='#TODO')
parser.add_argument('-msmcSegs', action='store_true',
                    help='#TODO')
parser.add_argument('-finerMsmcSegs', action='store_true',
                    help='#TODO')
parser.add_argument('-fixedMu', action='store_true',
                    help='Assume constant mutation rate, and scale all outputs (time intervals, coalescence rates and recombination rate) by the mutation rate.')
parser.add_argument('-fixedN', action='store_true',
                    help='Assume constant population size, and scale all outputs (time intervals, coalescence rates and recombination rate) by the coalescence rate.')
parser.add_argument('-iter', type=int, default=50, metavar = ('nIterations'),
                    help='Number of Baum-Welch iterations (default = 50)')
# TODO remove gof ...
parser.add_argument('-gof', metavar = ('l'), action='append', type=int, 
                   help='Calculate the goodness-of-fit parameter G_l.')
parser.add_argument('-par', metavar = ('nProcesses'), type=int, default=multiprocessing.cpu_count(),
                   help='Number of processes to use (default: number of CPU\'s).')
parser.add_argument('input', metavar = ('inputFile'), nargs='+', 
                   help='Input files. Supports Unix filename pattern matching (eg, chr*.txt or inputfiles/chr*.txt ).')
parser.add_argument('-trueHist', metavar = ('filename'), nargs=1, 
                   help='True population history (for simulated data; used as a sanity check for the numerical maxmization step (this parameter would not change the inferred histories)')
# TODO remove
# parser.add_argument('-bnd', default='none', help='TODO REMOVE.')


# TODO make sure everuthing works with nPrc = 1

# read input flags
args = parser.parse_args()
assert args.iter > 0
assert args.par  > 0
if args.gof is not None:
        assert min(args.gof) > 0

# Init output-writer process and processes pool
initParallel(args.par, args.o)

# log command line
# TODO perhaps printOutput() ?
writeOutput(" ".join(sys.argv))
writeOutput('BW steps will be spanned over %d processes'%args.par)

# read input dir & match all input files...
files = []
for inpPattern in args.input:
        pathName   = os.path.dirname(inpPattern)
        if pathName == '':
                pathName = os.curdir
        for f in os.listdir(pathName):
                if fnmatch.fnmatch(f, os.path.basename(inpPattern)):
                        files.append(pathName + '/' + f)

# TODO proper error message if (a) file doesn't exist (b) file doesn't match format
# read all input files (executed in parallel)
assert len(files) > 0
observations = runParallel(runMemberFunc(ObservedSequence, 'fromFile'), files)

writeOutput('read %d input files (%s)'%(len(files), ','.join(files)))
# TODO handle exceptions well, path doesn't exist or file doesn't exist
# TODO indentation is somehow 4 spaces some places and 8 in others

if args.psmcSegs:
        t = [4]+[2]*25+[4]+[7]
        bounds = [0.1*np.exp(float(i)/64.0*np.log(151))-0.1 for i in xrange(65)] + [np.inf]
elif args.finerMsmcSegs:
        bounds = [-np.log1p(-i/100.0) for i in xrange(100)] + [np.inf]
        t = [4]*25
else:
        assert args.msmcSegs
        t = [1]*10 + [2]*15
        bounds = [-np.log1p(-i/40.0) for i in xrange(40)] + [np.inf]

nSegs = len(bounds) - 1
pattern = []
for i in xrange(len(t)):
        for _ in xrange(t[i]):
                pattern.append(i)
assert len(pattern)  == nSegs
dummpyPattern = [0]*nSegs

if args.fixedMu:
        uPattern, lmbPattern = dummpyPattern, pattern
        scale = 'u0'
        pi = calcPi(observations)
        bounds = Theta(bounds, [1.0]*nSegs, [pi/2.0]*nSegs, pi/2.0, calcHmm=False).scale(2.0/pi, False).segments.boundaries
        
else:
        assert args.fixedN
        lmbPattern, uPattern = dummpyPattern, pattern
        scale = '2N0'

start = time.time()

model = Model(lmbPattern, uPattern, scale, fixedBoundaries = bounds)
if args.gof is not None:
        gof = args.gof
else:
        gof = []
        
# TODO THIS DOESNT MAKE SENCE: TRUETHETA MIGHT NOT BE DEFINED BY THE MODEL SO ITS MEANINGLESS; ALSO NOT SCALED PROPERLY
if args.trueHist is None:
        trueTheta = None
else:
        trueTheta = readHist(args.trueHist)
        trueTheta = trueTheta.scale(1.0/trueTheta.r, calcHmm=False)
model.run(observations, args.iter, trueTheta = trueTheta)

# TODO direct stderr and stdout also to files???        
writeOutput('Done (overall execution time: %f minutes).'%((time.time()-start)/60))

