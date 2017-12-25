import re
import numpy as np

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
    res += template.format('t_start (generations)', 't_end (generations)', tabHeader)
    for i in xrange(len(vals)):
        res += template.format(bounds[i], bounds[i+1], vals[i])
    return res
    

def writeHistory(r, u_boundaries, u_vals, N_boundaries, N_vals):
    
    res = ''
    res += 'recombination rate:\t{0}\n\n'.format(r)
    res += _writeTable('mutation rate history:\n'  , 'u', u_boundaries, u_vals) + '\n'
    res += _writeTable('population size history:\n', 'N', N_boundaries, N_vals)
    
    return res

# parse table (in the format defined by _writeTable) into array of boundaries and values
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

# read history file in the format defined above
def readHistory(filename):
    
    # read input file to string
    with open(filename, 'r') as f:
        inp = f.read()
    
    # define format pattern
    pattern = re.compile(r"""
                                 ^
                                  recombination\ rate:\t(?P<r>.+)\n\n       # recombination rate
                                  .+\n\t.+\n                                # mutaion rate table headers
                                  (?P<uTab>(\t.+\n)+) \n                    # mutation rate table
                                  .+\n\t.+\n                                # population size table headers
                                  (?P<nTab>(\t.+\n)+)                       # population size table
                                 $
                             """, re.VERBOSE)
    
    # match input to pattern
    match = pattern.match(inp)
    assert match is not None
    
    # read values
    r = float(match.group("r"))
    u_boundaries, u_vals = _parseTable(match.group("uTab"))
    N_boundaries, N_vals = _parseTable(match.group("nTab"))
    
    print r, u_boundaries, u_vals, N_boundaries, N_vals
    return r, u_boundaries, u_vals, N_boundaries, N_vals
    
    