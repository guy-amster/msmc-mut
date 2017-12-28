#!/usr/bin/env python
from Containers import Model
from ObservedSequence import ObservedSequence
from BaumWelch import BaumWelch
from Parallel import initParallel, runParallel, runMemberFunc, writeOutput, OutputWriter
from history import readHistory, scale
import numpy as np 
import os
import sys
import fnmatch
import argparse
import multiprocessing
import time

# specify flags & usage
parser = argparse.ArgumentParser(description='Infer population size and mutation rate histories. \n ')

parser.add_argument('-o', default='', metavar = ('prefix'), 
                    help='Prefix to use for all output files.')
parser.add_argument('-fixedMu', action='store_true',
                    help='Assume constant mutation rate, and scale all outputs (time intervals, coalescence rates and recombination rate) by the mutation rate.')
parser.add_argument('-iter', type=int, default=50, metavar = ('nIterations'),
                    help='Number of Baum-Welch iterations (default = 50)')
# TODO remove gof ...
parser.add_argument('-gof', metavar = ('l'), action='append', type=int, 
                   help='Calculate the goodness-of-fit parameter G_l.')
parser.add_argument('-par', metavar = ('nProcesses'), type=int, default=multiprocessing.cpu_count(),
                   help='Number of processes to use (default: number of CPU\'s).')
parser.add_argument('input', metavar = ('inputFile'), nargs='+', 
                   help='Input files. Supports Unix filename pattern matching (eg, chr*.txt or inputfiles/chr*.txt ).')
# TODO remove
parser.add_argument('-bnd', default='none',
                   help='TODO REMOVE.')


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
observations = runParallel(runMemberFunc(ObservedSequence, 'fromFile'), files)

writeOutput('read %d input files (%s)'%(len(files), ','.join(files)))
# TODO handle exceptions well, path doesn't exist or file doesn't exist

# TODO THIS IS TEMPORARY
def calcPi(observations):
        het, length = 0, 0
        # TODO support missing sites, different site types etc
        for obs in observations:
                length += obs.length
                het    += np.count_nonzero(obs.posTypes) # remove np import
        pi = float(het)/float(length)
        return pi

pi = calcPi(observations)
if args.bnd == 'msmc':
        boundaries  = [-0.5*pi*math.log1p(-1.0*i/40.0) for i in xrange(11)]
        boundaries += [-0.5*pi*math.log1p(-1.0*i/40.0) for i in xrange(12,40,2)]
        boundaries += [np.inf]
elif args.bnd == 'u1':
        boundaries = [0.0, 0.00002732, 0.00010245, 0.003415, np.inf]
elif args.bnd == 'r1':
        boundaries = [0.0, 0.00000452, 0.00001695, 0.000565, np.inf]
elif args.bnd == 'cu1':
        boundaries = [0.0, 0.000111572, 0.000255413, 0.000458145, 0.000804719, np.inf]
elif args.bnd == 'cr1':
        boundaries = [0.0, 8.92574E-05, 0.00020433, 0.000366516, 0.000643775, np.inf]
else:
                assert args.fixedMu
                r, u_boundaries, u_vals, N_boundaries, N_vals = readHistory(args.bnd)
                r, u_boundaries, u_vals, N_boundaries, N_vals = scale(1.0/u_vals[0], r, u_boundaries, u_vals, N_boundaries, N_vals)
                #print N_boundaries
                boundaries = [b*u_vals[0] for b in N_boundaries]
                #print boundaries
                # exit()
        

start = time.time()

model = Model(pi, boundaries, fixedR=(not args.fixedMu), fixedLambda=False, fixedMu=args.fixedMu)
if args.gof is not None:
        gof = args.gof
else:
        gof = []

BaumWelch(model, observations, nIterations=args.iter, gof=gof)

# TODO direct stderr and stdout also to files???        
writeOutput('Done (overall execution time: %f minutes).'%((time.time()-start)/60))

