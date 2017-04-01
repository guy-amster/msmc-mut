from multiprocessing import Pool

# TODO Devevlop this into a state class\ module with fixed number of processes and access to logger...
# TODO module or class
# TODO logger safe threading
# TODO handle multiple args better
# TODO migrate all pool calls to here.
# TODO create p ONCE

# return instance.memberName(args0, *args);
# Note: Pool.map() only allows calling pickable functions (ie defined at top module level)
#       This allows to overcome this and invoke Pool.map() class functions 
#       (class function output returns to parent process, but any changes made to the class are lost)
def runMemberFunc(arg0, instance, memberName, *args):
    return getattr(instance, memberName)(arg0, *args)

# return [f(inp) for inp in inputs] using nPrc processes to make things faster.
def runParallel(f, inputs, nPrc):
    
    # Since exceptions within pool.map result in a non-informative exception in the parent process,
    #   for debugging purposes it's sometimes easier to avoid pool.map.
    # I prefer not to call pool.map if nPrc = 1.
    if nPrc == 1:
        res = [f(inp) for inp in inputs]
    
    else:
        p = Pool(nPrc)
        res = p.map(f, inputs)       
        p.close()
    
    return res
    
#TODO register at exit runs at every process or just the main one????