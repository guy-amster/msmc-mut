#!/usr/bin/env python

import argparse
import sys
import re
import numpy as np
from Theta import Theta
# TODO import Segments

parser = argparse.ArgumentParser(description='Create hist object using ms\' and mut.py\' flags. \n ')

# TODO perhaps this 'history' module should serve instead of model & theta in a more general way?

# define input structure using argparse
parser.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout, metavar='filename',
                    help='output file (defaults to stdout)')
parser.add_argument('-t', dest='t', metavar = ('theta'), type=float, required=True,
                   help='see ms\' specifications')
parser.add_argument('-u', dest='u', metavar = ('u'), type=float, required=True,
                   help='u (per-gen mutation rate at time 0 per-bp)')
parser.add_argument('-r', dest='r', metavar = ('rho','nSites'), nargs=2, type=float, required=True,
                   help='see ms\' specifications')
parser.add_argument('-eN', dest='eN', metavar = ('t','x'), action='append', nargs=2, type=float, 
                   help='see ms\' specifications')
parser.add_argument('-en', dest='en', metavar = ('t','i', 'x'), action='append', nargs=3, type=float, 
                   help='compatible with psmc\' output')
parser.add_argument('-uT', dest='uT', metavar = ('time','u'), action='append', nargs=2, type=float, 
                   help='Assume a mutation rate u (per 4N_0 generations per-bp) from \'time\' backwards (time measured in units of 4N_0, as in ms)')


# read input
args = parser.parse_args()

# parse parameters
nSites       = args.r[1]
_N0          = args.t/(4.0 * args.u * nSites)
r            = args.r[0]/(4.0 * _N0 * nSites)

# combine eN, uT and en flags and sort by t
flags = []
if args.eN is not None:
    flags += [('N', v[0], v[1]) for v in args.eN]
if args.en is not None:
    for x in args.en:
        assert x[1] == 1.0
        v = (x[0],x[1])
        flags.append(('N', v[0], v[1]))
if args.uT is not None:
    flags += [('u', v[0], v[1]) for v in args.uT]
flags = sorted(flags, key=lambda x:x[1])

# combine flags to history
bound, u, N = 0.0, args.u, _N0
boundaries, Ns, us = [], [], []
for f in flags + [('u', np.inf, 0.0)]:
    typ, time, val = f
    time = time*4*_N0
    if time > bound:
        boundaries.append(bound)
        Ns.append(N)
        us.append(u)
        bound = time
    if typ == 'u':
        u = val/(4.0*_N0)
    else:
        assert typ == 'N'
        N = _N0 * val
    
boundaries.append(np.inf)
    
# print history:
args.o.write(str(Theta(boundaries, [.5/x for x in Ns], us, r)))