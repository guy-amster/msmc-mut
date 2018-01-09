#!/usr/bin/env python
import argparse
import math
import sys
from readUtils import readLoop

# define input structure using argparse
parser = argparse.ArgumentParser(description='Read and scale inferred parameters from loop file\n ')
parser.add_argument('filename', help='input loop file')
parser.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout, metavar='filename',
                    help='output file (defaults to stdout)')
parser.add_argument('-iter', dest='iter', metavar = ('n'), default=-1, type=int,
                   help='Read parameters after the end of the nth iteration (default: last iteration)')
parser.add_argument('-su', dest='su', metavar = ('u'), type=float,
                    help='Scale parameters such that the mutation rate at time 0 will equal u')
parser.add_argument('-sr', dest='sr', metavar = ('r'), type=float,
                    help='Scale parameters such that the recombination rate will equal r')

# read input flags
args = parser.parse_args()

# read iteration from loop file
_, thetas, _ = readLoop(args.filename)
theta = thetas[args.iter]

# scale, if requested
if args.su is not None:
    theta = theta.scale(args.su/theta.uVals[0])
if args.sr is not None:
    theta = theta.rescale(args.sr/theta.r)
    
# write as output
args.o.write(str(theta))