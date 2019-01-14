#!/usr/bin/env python
import argparse
from readUtils import readHist, readLoop

# define input structure using argparse
parser = argparse.ArgumentParser(description='Calculate distance from ref distribution  \n ')
parser.add_argument('ref', metavar = ('ref'), help='reference hist file')
parser.add_argument('loop', metavar = ('loop'), help='loop file')
parser.add_argument('-val', type=float, metavar = ('val'), default = 1.0, help='')

# TODO REMOVE parser.add_argument('scale', metavar = ('scale'), choices=['a','b','c'], help='scale unit; either 2N0, u0 or r')

# read input flags
args = parser.parse_args()

# read ref hist, loop file
ref = readHist(args.ref)
model, iterations, _, _ = readLoop(args.loop)
thetas = [i[0] for i in iterations]
scale = model.scale

template = '{0:<24}{1:<24}{2:<24}{3:<24}\n'
print template.format('iter', 'dN', 'du', 'dr')
_iter = 0
for theta in thetas:
    dN, du, dr = ref.d(theta, scale, val = args.val)
    print template.format(_iter, dN, du, dr)
    _iter += 1
