__author__ = 'Guy Amster'

import argparse
import sys
import numpy as np
import re
import random
import os
import fnmatch
from subprocess import call
import time

# read general input parameters from cmds[0]
parser = argparse.ArgumentParser(description='runs MSMC on all files muts*.txt in an input directory')
parser.add_argument('nProcesses', metavar='nProcesses', type=int, nargs=1,
                   help='number of processes to invoke in parallel')
parser.add_argument('inputDir', metavar='inputDir', nargs=1,
                   help='input directory')
parser.add_argument('outputDir', metavar='outputDir', nargs=1,
                   help='output directory')
parser.add_argument('-nIters', default=[20], type=int, nargs=1,
                   help='number of iterations (default: 20)')
args = parser.parse_args()

# read number of processes to use
nPrc = args.nProcesses[0]

# read input dir & match all input files...
inPath = args.inputDir[0]
inputFilenames = []
if inPath[-1] != '/':
    inPath += '/'
for f in os.listdir(inPath):
    if fnmatch.fnmatch(f, 'muts*.msmc'):
        inputFilenames.append(inPath + f)

# read output directory path
outPath = args.outputDir[0]
if outPath[-1] != '/':
    outPath += '/'

# run msmc
msmcCmd = ['/mnt/data/soft/msmc/build/msmc']

msmcOut  = outPath + 'msmc'
msmcStd  = msmcOut + '.stdout'
msmcErr  = msmcOut + '.stderr'
msmctime = msmcOut + '.time'

msmcCmd += ['-o', msmcOut, '-t', '%d'%nPrc, '-i', '%d'%args.nIters[0], '-R']

msmcCmd += inputFilenames

with open(msmcStd, 'w') as out, open(msmcErr,"w") as err, open(msmctime,"w") as t:
    start = time.time()
    t.write(' '.join(msmcCmd) + '\n')
    call(msmcCmd, stdout = out, stderr = err)
    tot   =  time.time() - start
    t.write('time ' + str(tot) + '\n')
for fName in [msmcStd, msmcErr]:
    if os.stat(fName).st_size == 0:
        os.remove(fName)






