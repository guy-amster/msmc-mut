# TODO 'script' this so I could run without python cre... ?
import argparse
import sys
import re
from history import writeHistory
# TODO remove?
import numpy as np

parser = argparse.ArgumentParser(description='Create hist object using ms\' and mut.py\' flags. \n ')

# TODO perhaps this 'history' module should serve instead of model & theta in a more general way?

# define input structure using argparse
# TODO these are not optional...
parser.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout, metavar='filename',
                    help='output file (defaults to stdout)')
parser.add_argument('-t', dest='t', metavar = ('theta'), type=float, 
                   help='see ms\' specifications')
parser.add_argument('-u', dest='u', metavar = ('u'), type=float, 
                   help='u (per-gen mutation rate)')
parser.add_argument('-r', dest='r', metavar = ('rho','nSites'), nargs=2, type=float, 
                   help='see ms\' specifications')
parser.add_argument('-eN', dest='eN', metavar = ('t','x'), action='append', nargs=2, type=float, 
                   help='see ms\' specifications')

# read input
args = parser.parse_args()

# parse parameters
u_boundaries = [0.0, np.inf]
u_vals       = [args.u]
nSites       = args.r[1]
_N0          = args.t/(4.0 * args.u * nSites)
r            = args.r[0]/(4.0 * _N0 * nSites)
eN           = sorted(args.eN, key=lambda x:x[0])
N_vals       = [_N0] + [x[1]*_N0 for x in eN]
N_boundaries = [0.0] + [x[0]*4*_N0 for x in eN] + [np.inf]

# print history:
args.o.write(writeHistory(r, u_boundaries, u_vals, N_boundaries, N_vals))