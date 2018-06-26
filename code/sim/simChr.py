__author__ = 'Guy Amster'

import argparse
import sys
import numpy as np
import re
import random
import os
from subprocess import call
from multiprocessing import Pool
import time

def runAndDirect(cmd, outName, errName, measureTime=False):
    with open(outName, 'w') as out, open(errName,"w") as err:
        start = time.time()
        call(cmd, stdout = out, stderr = err)
        tot   =  time.time() - start
        if measureTime:
            out.write('time ' + str(tot) + '\n')
    for fName in [outName, errName]:
        if os.stat(fName).st_size == 0:
            os.remove(fName)
    

def simulateChromosome(d):
    
    N0         = d['N']
    outPath    = d['outPath']
    outputName = str(d['name'])
        
    msPath   = '/mnt/data/msmc-bs/msdir/'
    mutPath  = '/mnt/data/msmc-mut/code/sim/'

    
    #### simulate TMRCA's using ms ####
    
    msCmd = [msPath + 'ms', '2', '1', '-T']
    
    # recombination flag
    chrSize  = 10000000
    r        = 10**(-8)
    msCmd   += ['-r', '%f'%(4*N0*r*chrSize), '%d'%chrSize]
    
    # seed flag (ms takes 3 seeds, each in the range [0,2**16)
    rng      = random.SystemRandom()
    seeds    = ['%d'%rng.randint(0, (2**16)-1) for _ in range(3)]
    msCmd   += ['-seeds'] + seeds
    
    
    outName = outPath + 'tree' + outputName
    outStd  = outName + '.txt'
    outErr  = outName + '.err'
    runAndDirect(msCmd, outStd, outErr)
        
    #### simulate mutations, conditioned on the gene trees, using mut.py ####
    
    mutCmd = ['python', mutPath + 'mut.py']
    
    # output name
    mutCmd += ['-c', outputName]
    
    # mutation rate
    u = 1.0 * (10**(-8))
    mutCmd += ['%f'%(4.0 * N0 * u)]
    
    # read tree
    mutCmd += [outStd]
    
    outName = outPath + 'muts' + outputName
    outStd  = outName + '.txt'
    outErr  = outName + '.err'
    runAndDirect(mutCmd, outStd, outErr)
    
    return outputName

    
    
    
    
    



