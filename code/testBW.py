from Containers import Model, Theta
from ObservedSequence import ObservedSequence
from BaumWelch import BaumWelch
from GOF import GOF
from Logger import log, logError
import math
import numpy as np
import coalSim
import os
import fnmatch

# TODO move somewhere
def calcPi(observations):
        het, length = 0, 0
        # TODO support missing sites, different site types etc
        for obs in observations:
                length += obs.length
                het    += np.count_nonzero(obs.posTypes)
        pi = float(het)/float(length)
        return pi

# read input dir & match all input files...
filenames = []
inpPattern = 'raw/muts*.txt' # TODO inpPattern = args.inputDir[0]
pathName   = os.path.dirname(inpPattern)
for f in os.listdir(pathName):
    if fnmatch.fnmatch(f, os.path.basename(inpPattern)):
        filenames.append(pathName + '/' + f)


observations = []

for filename in filenames:
        observations.append(ObservedSequence.fromFile(filename))

model = Model(calcPi(observations), fixedR=False, fixedLambda=False, fixedMu=True)
nPrc = 63

BaumWelch(model, observations, nProcesses=nPrc, nIterations=50, l=10)

