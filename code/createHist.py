#!/usr/bin/env python

import argparse
import sys
import re
import numpy as np
from CoalParams import CoalParams
# TODO import Segments

parser = argparse.ArgumentParser(description='Create hist object using ms\' and mut.py\' flags. \n ')

# TODO perhaps this 'history' module should serve instead of model & theta in a more general way?

# define input structure using argparse
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

_N = [_N0] + [x[1]*_N0 for x in eN]
lmbVals = [.5/x for x in _N]
uVals = [args.u for _ in _N]
boundaries = [0.0] + [x[0]*4*_N0 for x in eN] + [np.inf]

# print history:
args.o.write(CoalParams(Segments(boundaries), lmbVals, uVals, r))