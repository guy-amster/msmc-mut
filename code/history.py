import re
import numpy as np

# TODO this should be merged with Theta class; 'history' change to params? 

def _writeTable(header, tabHeader, bounds, vals):
    
    # verify input validity
    assert bounds[0] == 0.0
    assert np.isinf(bounds[-1])
    assert len(bounds) == (len(vals) + 1)
    for i in xrange(len(bounds) - 1):
        assert bounds[i] < bounds[i+1]
    for i in xrange(len(vals)):
        assert vals[i] > 0.0
    
    res = header
    template = '\t{0:<24}{1:<24}{2:<24}\n'
    res += template.format('t_start', 't_end', tabHeader)
    for i in xrange(len(vals)):
        res += template.format(bounds[i], bounds[i+1], vals[i])
    return res
    

def writeHistory(r, u_boundaries, u_vals, N_boundaries, N_vals):
    
    res = ''
    res += 'recombination rate:\t{0}\n\n'.format(r)
    res += _writeTable('mutation rate history:\n'  , 'u', u_boundaries, u_vals) + '\n'
    res += _writeTable('population size history:\n', 'N', N_boundaries, N_vals)
    
    return res

# parse a table (either u or N table in the format defined by _writeTable)
# into array of boundaries and values
def _parseTable(tab):
    bounds, vals = [], []
    
    # parse table line by line
    for line in tab.split('\n')[:-1]:
        
        t = re.findall(r"([-|\w|\+|\.]+)", line)
        assert len(t) == 3
        t_start = float(t[0])
        if len(bounds) > 0:
            assert t_start == float(t_end)
            assert t_start  > bounds[-1]
        else:
            assert t_start == 0.0
        t_end, val = t[1], float(t[2])
        bounds.append(t_start)
        vals  .append(val    )
        
    assert t_end == 'inf'
    bounds.append(np.inf)
        
    return bounds, vals

# parse the output in the format defined by writeHistory
def _parseHistory(header, footer, inp):
    
    # define format pattern
    pattern = re.compile(r"""
                                 %s
                                  recombination\ rate:\t(?P<r>.+)\n\n       # recombination rate
                                  .+\n\t.+\n                                # mutaion rate table headers
                                  (?P<uTab>(\t.+\n)+) \n                    # mutation rate table
                                  .+\n\t.+\n                                # population size table headers
                                  (?P<nTab>(\t.+\n)+)                       # population size table
                                 %s
                             """%(header, footer), re.VERBOSE)
    # match input to pattern
    match = pattern.search(inp)
    assert match is not None
    
    # read values
    r = float(match.group("r"))
    u_boundaries, u_vals = _parseTable(match.group("uTab"))
    N_boundaries, N_vals = _parseTable(match.group("nTab"))
    
    return r, u_boundaries, u_vals, N_boundaries, N_vals
    
# read history file in the format defined above
def readHistory(filename):
    
    # read input file to string
    with open(filename, 'r') as f:
        inp = f.read()
    return _parseHistory('^', '$', inp)

# read history from iteration nIter in 'loop' output file
# (nIter = -1 corresponds to the last iteration)
def readLoop(filename, nIter):
    
    # read input file to string
    with open(filename, 'r') as f:
        inp = f.read()
    if nIter == -1:
        nIter = inp.count('iterations:\n') - 1
    return _parseHistory('After\ %d\ iterations:\\n\\n'%nIter, '', inp)

# scale parameters by C
def scale(C, r, u_boundaries, u_vals, N_boundaries, N_vals):
    assert C > 0.0
    r = r*C
    u_vals = [x*C for x in u_vals]
    N_vals = [x/C for x in N_vals]
    u_boundaries = [x/C for x in u_boundaries]
    N_boundaries = [x/C for x in N_boundaries]
    return r, u_boundaries, u_vals, N_boundaries, N_vals

    