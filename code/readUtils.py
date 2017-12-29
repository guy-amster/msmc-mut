from CoalParams import CoalParams

# read CoalParams from file.
# return valus: model, list of CoalParams (corresponding to iterations), list of reported stats (corresponding to iterations)
def readHist(filename):
    
    # read file
    with open(filename, 'r') as f:
        inp = f.read()
    return CoalParams.fromString(inp)

# read and parse loop file
def readLoop(filename):
    
    # read file
    with open(filename, 'r') as f:
        inp = f.read()
    
    # read model specs first
    pattern = re.compile(r"""
                             ^
                              Model\ specifications:\n (?P<model>[.*\n]+)\n  # model specs
                              (?P<rest>After\ 0\ iterations:\n[.*\n]+))      # rest of file
                             $
                          """, re.VERBOSE)
    # match input to pattern
    match = pattern.search(inp)
    assert match is not None
        
    # parse model specs
    model = Model.fromString(match.group("model"))
    
    # read iteration by iteration
    params, stats = [], []
    inp = match.group("rest")
    while (len(inp) > 0):
        
        # read coalParams & stats from next iteration
        # define pattern
        pattern = re.compile(r"""
                                 ^
                                  After\ (?P<nIter>\d+)\ iterations: \n\n   # iteration header
                                  (?P<theta> rec.+\n\n .+\n (\t.+\n)+ ) \n  # CoalParams
                                  Statistics:\n \t.+\n \t(?P<stats>.+)\n\n  # stats
                                  (?P<rest> [.*\n]*)                        # rest of file
                                 $
                              """, re.VERBOSE)
        # match input to pattern
        match = pattern.search(inp)
        assert match is not None

        assert int(match.group("nIter")) == len(params)
        params.append(CoalParams.fromString(match.group("theta")))
        
        # treat missing values ('.') as nan
        def _castFloat(x):
            if x == '.':
                return float('nan')
            return float(x)
        stats.append(map(_castFloat, re.findall(r"([-|\w|\+|\.]+)", match.group("stats"))))
        inp = match.group("rest")
    
    return model, params, stats

    