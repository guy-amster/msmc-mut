from Theta import Theta
from Model import Model
import re
from collections import namedtuple

# statistics reported for each iteration
Stats = namedtuple('Stats', ['logL', 'QInit', 'QMax'])

# read Theta from file.
# return valus: model, list of Thetas (corresponding to iterations), list of reported stats (corresponding to iterations)
def readHist(filename):
    
    # read file
    with open(filename, 'r') as f:
        inp = f.read()
    return Theta.fromString(inp)

# read and parse loop file
# return value: model, iterations, trueTheta, trueThetaLikelihood
#               - model is a Model object as defined in the header of the loop file
#               - each entry in iterations is (theta, stats); correspponding to the inferred params and the stats for that iteration
#               - for simulated data, trueTheta is the true set of paramters, and trueLikelihood is the log-likelohood of the observations given trueTheta
#               - for non-simulated data, trueTheta, trueLikelihood are None
def readLoop(filename):
    
    # read file
    with open(filename, 'r') as f:
        inp = f.read()
    
    nIterations = inp.count('After') - 1
    assert nIterations >= 0
    
    # read model specs first
    pattern = re.compile(r"""
                             ^
                              Model\ specifications: \n             # header line
                              (?P<model> (.*\n)+ \t scale .+ \n) \n # model specs
                              (?P<rest>(After\ 0|True).+\n(.*\n)+)  # rest of file
                             $
                          """, re.VERBOSE)
    
    # match input to pattern
    match = pattern.search(inp)
    assert match is not None
        
    # parse model specs
    model = Model.fromString(match.group("model"))
    inp = match.group("rest")
    
    # pattern for a single iteration
    pattern = re.compile(r"""
                             ^
                              (?P<header>(.*)) \n\n                     # iteration header line
                              (?P<theta> rec.+\n\n .+\n (\t.+\n)+ ) \n  # Theta
                              Statistics:\n \t.+\n \t(?P<stats>.+)\n\n  # stats
                              (?P<rest> (.*\n)*)                        # rest of file
                             $
                          """, re.VERBOSE)
    
    # read iteration after iteration
    iterations, trueTheta, trueLikelihood = [], None, None
    while len(inp) > 0:
        
        # match input to pattern
        match = pattern.search(inp)
        assert match is not None
        
        theta = Theta.fromString(match.group("theta"))
        # treat missing values ('.') as nan
        def _castFloat(x):
            if x == '.':
                return float('nan')
            return float(x)
        stats = Stats(*map(_castFloat, re.findall(r"([-|\w|\+|\.]+)", match.group("stats"))))
        
        # read header line; determine if we're reading the trueTheta vals (if reported, for simulated theta), or inferred values
        if match.group("header") == 'True parameters:':
            trueTheta = theta 
            trueLikelihood = stats.logL
        else:
            assert match.group("header") == 'After %d iterations:'%len(iterations)
            iterations.append((theta, stats))
        
        inp = match.group("rest")
    
    assert len(iterations) == (nIterations+1)
    return model, iterations, trueTheta, trueLikelihood

    