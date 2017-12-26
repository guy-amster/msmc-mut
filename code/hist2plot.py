import argparse
import math
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from history import readHistory


parser = argparse.ArgumentParser(description='Plot population and mutation histories  \n ')
parser.add_argument('-p', dest='p', metavar = ('filename', 'label'), action='append', nargs=2, required=True,
                   help='input history file and corresponding label')
parser.add_argument('-o', dest='o', metavar = ('filename'), required=True, help='output name')

# read input flags
args = parser.parse_args()

# read histories
hists  = [readHistory(h[0]) for h in args.p]
labels = [h[1]              for h in args.p]

# generate plot
sns.set_style("darkgrid")

# points for X-axis and their log-10 values
X  = np.arange(1000,10000000, step=50)
Xl = np.log10(X)

# Y-axis values:
maxY = 0.0
for i in xrange(len(args.p)):
    r, u_boundaries, u_vals, N_boundaries, N_vals = hists[i]
    
    # translate from generations to years (assuming g=25)
    N_boundaries = [25*x for x in N_boundaries]
    
    # evaluate N on the grid X
    condlist = [np.logical_and(X >= N_boundaries[j], X < N_boundaries[j+1]) for j in range(len(N_boundaries) - 1)]
    funclist = [(lambda z: lambda x: z)(val) for val in N_vals]
    Y        = np.piecewise(X, condlist, funclist)/10000.0

    maxY = max(maxY, max(Y))
    plt.plot(Xl, Y, label = labels[i])
    

yLimit = int(math.ceil(max(Y)))
plt.axis([4, 7, 0, yLimit])

plt.title('Population size history')
plt.xlabel('Years (G=25)')
plt.ylabel('Effective population size (x $10^4$)')
plt.legend(loc='upper right')
plt.xticks(range(4,8), ['$10^%d$'%i for i in range(4,8)])
plt.yticks(range(1,yLimit), ['%d'%i for i in range(1,yLimit)])

# write plot to file
oFile = args.o
if oFile[-4:] != '.pdf':
    oFile = oFile + '.pdf'
plt.savefig(oFile)
