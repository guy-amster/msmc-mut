# from Containers import Model, Theta
# from ObservedSequence import ObservedSequence
# from BaumWelch import BaumWelch
# from GOF import GOF
from Logger import log, logError, setLoggerPath
# import math
# import numpy as np
# import coalSim
# import os
# import fnmatch
import argparse
import sys
import multiprocessing

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


# read input flags
args = parser.parse_args()
assert args.iter > 0
assert args.par  > 0
if args.gof is not None:
        assert min(args.gof) > 0
# TODO log entire command line

# set logger path
setLoggerPath(args.o)

# read input dir & match all input files...
files = []
for inpPattern in args.input:
        pathName   = os.path.dirname(inpPattern)
        for f in os.listdir(pathName):
                if fnmatch.fnmatch(f, os.path.basename(inpPattern)):
                        files.append(pathName + '/' + f)


observations = [ObservedSequence.fromFile(f) for f in files]
# TODO log 'read X files (names)'
# TODO handle exceptions well, path doesn't exist or file doesn't exist

# TODO move somewhere
def calcPi(observations):
        het, length = 0, 0
        # TODO support missing sites, different site types etc
        for obs in observations:
                length += obs.length
                het    += np.count_nonzero(obs.posTypes)
        pi = float(het)/float(length)
        return pi

assert args.fixedMu
if args.fixedMu:
        model = Model(calcPi(observations), fixedR=False, fixedLambda=False, fixedMu=True)
        
        BaumWelch(model, observations, nProcesses=args.par, nIterations=args.iter, gof=args.gof)

