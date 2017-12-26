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
parser.add_argument('-t', dest='t', metavar = ('theta'), type=float, required=True,
                   help='see ms\' specifications')
parser.add_argument('-u', dest='u', metavar = ('u'), type=float, required=True,
                   help='u (per-gen mutation rate)')
parser.add_argument('-r', dest='r', metavar = ('rho','nSites'), nargs=2, type=float, required=True,
                   help='see ms\' specifications')
parser.add_argument('-eN', dest='eN', metavar = ('t','x'), action='append', nargs=2, type=float, 
                   help='see ms\' specifications')
parser.add_argument('-en', dest='en', metavar = ('t','i', 'x'), action='append', nargs=3, type=float, 
                   help='see ms\' specifications')


# read input
args = parser.parse_args()

# parse parameters
u_boundaries = [0.0, np.inf]
u_vals       = [args.u]
nSites       = args.r[1]
_N0          = args.t/(4.0 * args.u * nSites)
r            = args.r[0]/(4.0 * _N0 * nSites)

# combine eN and en flags and sort by t
eN = []
if args.eN is not None:
    eN += args.eN
if args.en is not None:
    for x in args.en:
        assert x[1] == 1.0
        eN.append((x[0],x[2]))
eN = sorted(eN, key=lambda x:x[0])

N_vals       = [_N0] + [x[1]*_N0 for x in eN]
N_boundaries = [0.0] + [x[0]*4*_N0 for x in eN] + [np.inf]

# print history:
args.o.write(writeHistory(r, u_boundaries, u_vals, N_boundaries, N_vals))