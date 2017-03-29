__author__ = 'Guy Amster'

import argparse
import sys
import numpy as np
import re
import random
import os
from subprocess import call
from multiprocessing import Pool
from simChr import simulateChromosome, runAndDirect

# read imput commands:
subcommands = ['sim']
cmds, cmd = [], []
for arg in sys.argv[1:]:
    if arg in (subcommands):
        # start new command
        if cmd is not None:
            cmds.append(cmd)
        cmd = [arg]
    else:
        cmd.append(arg)
cmds.append(cmd)

# read general input parameters from cmds[0]
parser = argparse.ArgumentParser(description='MSMC simulator')
parser.add_argument('nProcesses', metavar='nProcesses', type=int, nargs=1,
                   help='number of processes to invoke in parallel')
parser.add_argument('outputDir', metavar='outputDir', nargs=1,
                   help='output directory')
args = parser.parse_args(cmds[0])

# read number of processes to use
nPrc = args.nProcesses[0]

# read output directory path
outPath = args.outputDir[0]
if outPath[-1] != '/':
    outPath += '/'

# if directory exists: verify it's empty
if os.path.exists(outPath):
    assert os.path.isdir(outPath)
    assert os.listdir(outPath) == []
    
# otherwise: create it
else:
    os.makedirs(outPath)
    
# create subdirectory
rawPath = outPath + 'raw/'
os.makedirs(rawPath)

# parser for simulation command
simParser = argparse.ArgumentParser(description='simulate sequences using ms')
simParser.add_argument('n', type=int,
                    help='number of chromosomes to simulate')
simParser.add_argument('N', type=float,
                    help='population size at time 0')

    
# process commands one by one
simIndex = 0

for cmd in cmds[1:]:
    if cmd[0] == 'sim':
        args = simParser.parse_args(cmd[1:])
        N = args.N
        n = args.n
        
        inp = [{'N':N, 'name':i, 'outPath':rawPath} for i in xrange(simIndex, simIndex+n)]
        p   = Pool(nPrc)
        res = p.map(simulateChromosome, inp)
       
        p.close()
        
        
        simIndex += n

# run msmc
msmcCmd = ['/mnt/data/msmc/build/msmc']

msmcOut = outPath + 'msmc'
msmcStd = outPath + 'stdout.msmc.txt'
msmcErr = outPath + 'stderr.msmc.txt'

msmcCmd += ['-o', msmcOut, '-t', '%d'%nPrc]

for i in xrange(0, simIndex):
    msmcCmd += [rawPath + 'muts%d.txt'%i]

runAndDirect(msmcCmd, msmcStd, msmcErr, measureTime=True)





