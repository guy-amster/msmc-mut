from Containers import Model
from ObservedSequence import ObservedSequence
from BaumWelch import BaumWelch
from Logger import log, logError, setLoggerPath
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
                    help='File prefix to use for all output files.')
parser.add_argument('-fixedMu', action='store_true',
                    help='Assume constant mutation rate, and scale all outputs (time intervals, coalescence rates and recombination rate) by the mutation rate.')
parser.add_argument('-iter', type=int, default=50, metavar = ('nIterations'),
                    help='Number of Baum-Welch iterations (default = 50)')
parser.add_argument('-gof', metavar = ('l'), action='append', type=int, 
                   help='Calculate the goodness-of-fit parameter G_l.')
parser.add_argument('-par', metavar = ('nProcesses'), type=int, default=multiprocessing.cpu_count(),
                   help='Number of processes to use (default: number of CPU\'s).')
parser.add_argument('input', metavar = ('inputFile'), nargs='+', 
                   help='Input files. Supports Unix filename pattern matching (eg, chr*.txt or inputfiles/chr*.txt ).')

# TODO make sure everuthing works with nPrc = 1
# TODO define pool here / somewhere shared, to avoid recreating pool each time???

# read input flags
args = parser.parse_args()
assert args.iter > 0
assert args.par  > 0
if args.gof is not None:
        assert min(args.gof) > 0

# set logger path
setLoggerPath(args.o)

# log command line
log(" ".join(sys.argv))

# read input dir & match all input files...
files = []
for inpPattern in args.input:
        pathName   = os.path.dirname(inpPattern)
        if pathName == '':
                pathName = os.curdir
        for f in os.listdir(pathName):
                if fnmatch.fnmatch(f, os.path.basename(inpPattern)):
                        files.append(pathName + '/' + f)


observations = [ObservedSequence.fromFile(f) for f in files]

log('read %d input files (%s)'%(len(files), ','.join(files)))
# TODO handle exceptions well, path doesn't exist or file doesn't exist

log('BW steps will be spanned over %d processes'%args.par)

# TODO move somewhere
def calcPi(observations):
        het, length = 0, 0
        # TODO support missing sites, different site types etc
        for obs in observations:
                length += obs.length
                het    += np.count_nonzero(obs.posTypes) # remove np import
        pi = float(het)/float(length)
        return pi

start = time.time()

model = Model(calcPi(observations), fixedR=(not args.fixedMu), fixedLambda=False, fixedMu=args.fixedMu)       
BaumWelch(model, observations, nProcesses=args.par, nIterations=args.iter, gof=args.gof)
        
log('Done (overall execution time: %f minutes).'%((time.time()-start)/60))

