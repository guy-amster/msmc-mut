#!/usr/bin/env python

import argparse
import math
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from readUtils import readHist

# define input structure using argparse
parser = argparse.ArgumentParser(description='Plot population and mutation histories  \n ')
parser.add_argument('-p', dest='p', metavar = ('filename', 'label'), action='append', nargs=2, required=True,
                   help='input history file and corresponding label')
parser.add_argument('-o', dest='o', metavar = ('filename'), required=True, help='output name')

# read input flags
args = parser.parse_args()

# read histories
hists = [readHist(h[0]) for h in args.p]
labels = [h[1] for h in args.p]

# generate plot
sns.set_style("darkgrid")
f, axarr = plt.subplots(2, sharex=True)

# points for X-axis and their log-10 values
minLogX, maxLogX = 4, 7
# X  = np.arange(1000,10000000, step=50)
# Xl = np.log10(X)

evalFuncs = [lambda hist:[0.5/(10000.0*x) for x in hist.lmbVals], lambda hist:[u*100000000.0 for u in hist.uVals]]
titles = ['Population size (x $10^4$)', 'Mutation rate (x $10^{-8}$)']

for ind, title, evalFunc in zip([1,0], titles, evalFuncs):
    # Y-axis values:
    maxY = 0.0
    minY = np.inf
    # plot file by file
    for i in xrange(len(args.p)):
        # read segments
        # TODO choose exactly where to evaluate instead of using this huge grid
        boundaries = hists[i].segments.boundaries
        
        # translate from generations to years (assuming g=25)
        boundaries = [25.0*x for x in boundaries]
        
        # convert coalescence rates to scaled population size; or scale mutation rates
        vals = evalFunc(hists[i])
        
        X, Y = [], []
        for j in xrange(len(boundaries)-1):
            X.append(max(boundaries[j],1.0))
            X.append(min(boundaries[j+1]-1.0, boundaries[-2]+10**maxLogX))
            Y.append(vals[j])
            Y.append(vals[j])
                    
        maxY = max(maxY, max(Y))
        minY = min(minY, min(Y))
        axarr[ind].plot(np.log10(X), Y, label = labels[i])
        
    
    maxY = int(math.ceil(maxY))
    minY = int(math.floor(minY))
    axarr[ind].axis([minLogX, maxLogX, minY, maxY])
    axarr[ind].set_ylabel(title)
    axarr[ind].set_yticks(range(minY+1,maxY+1), ['%d'%i for i in range(minY+1,maxY+1)])


plt.legend(loc='right')
plt.xticks(range(4,8), ['$10^%d$'%i for i in range(4,8)])



#plt.title('Inferred parameters')
plt.xlabel('Years (G=25)')

# write plot to file
oFile = args.o
if oFile[-4:] != '.pdf':
    oFile = oFile + '.pdf'
plt.savefig(oFile)
