#!/usr/bin/env python
import argparse
import math
import sys
from history import readLoop, writeHistory, scale

# input -o output -iter n -su u -sr r
# define input structure using argparse
parser = argparse.ArgumentParser(description='Read and scale inferred parameters from loop file\n ')
parser.add_argument('input', help='input loop file')
parser.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout, metavar='filename',
                    help='output file (defaults to stdout)')
parser.add_argument('-iter', dest='iter', metavar = ('n'), default=-1,
                   help='Read parameters after the end of the nth iteration (default: last iteration)')
parser.add_argument('-su', dest='su', metavar = ('u'), type=float,
                    help='Scale parameters such that the mutation rate at time 0 will equal u')
parser.add_argument('-sr', dest='sr', metavar = ('r'), type=float,
                    help='Scale parameters such that the recombination rate will equal r')

# read input flags
args = parser.parse_args()

# read iteration from loop file
r, u_boundaries, u_vals, N_boundaries, N_vals = readLoop(args.input, args.iter)

# scale, if requested
if args.su is not None:
    r, u_boundaries, u_vals, N_boundaries, N_vals = scale(args.su/u_vals[0], r, u_boundaries, u_vals, N_boundaries, N_vals)
if args.sr is not None:
    r, u_boundaries, u_vals, N_boundaries, N_vals = scale(args.sr/r, r, u_boundaries, u_vals, N_boundaries, N_vals)
    
# write as output
args.o.write(writeHistory(r, u_boundaries, u_vals, N_boundaries, N_vals))