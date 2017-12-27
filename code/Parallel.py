# This module builds on Pool.map(), providing a pool of worker processes with threading-safe output writing.
# Usage:
#   - Init with initParallel();
#   - Write output messages with writeOutput();
#   - Run parallel tasks with runParallel().

import atexit
import multiprocessing as mp
import sys
import os
import traceback
from functools import partial

# TODO migrate all pool calls to here.
            
    
# Initialize a logger process that will handle all output writing;
# Initialize a pool of nPrc worker processes;
# Register their closure at exit.
def initParallel(nPrc, outputPrefix):
    
    global _nPrc
    _nPrc = nPrc
    
    # We avoid Pool.map when using one processor (see reasons in comment below).
    # TODO Do log exceptions and remove this???
    if nPrc == 1:
        
        # init outputWriter in this process
        global _outputWriter
        _outputWriter = OutputWriter(outputPrefix)
        atexit.register(_outputWriter.close)
    
    else:
        assert nPrc > 1
        
        # Initialize queue for output messages
        manager = mp.Manager()
        q = manager.Queue()
        
        # set queue (to be used for output messages from the current process)
        _setQueue(q)
        
        # create pool of processes: 1 output-writer, and nPrc worker processes
        # all processes set q at init
        global _p
        # TODO maxtasksperchild test
        _p = mp.Pool(nPrc + 1, initializer=_setQueue, initargs=(q,))
        
        # run output-writer process
        _p.apply_async(_runOutputWriter,(outputPrefix,))
        
        # shcedule cleanup of these resources when program is finished
        atexit.register(_atExit)

# Write output messages to file.
def writeOutput(line, filename = 'Log'):
    
    originProcess = mp.current_process().name
    
    if _nPrc == 1:
        _outputWriter.write(line, filename, originProcess)
    
    else:
        _q.put(('msg', line, filename, originProcess))

# Calculate [f(inp) for inp in inputs] in parallel.
def runParallel(f, inputs):
    
    # Since exceptions within pool.map result in a non-informative exception in the parent process,
    #   for debugging purposes it's sometimes easier to avoid pool.map.
    # I prefer not to call pool.map if nPrc = 1.
    if _nPrc == 1:
        return [f(inp) for inp in inputs]
    
    else:
        # run _runAndCatch(f, inp) for inp in inputs;
        # _runAndCatch(f, ) is just f with improved exception handling
        return _p.map(partial(_runAndCatch,f), inputs)

# Try & compute f(inp).
# If an exception is raised, print it from within the working process.
# (This function improves the poor builtin exception reporting of Pool.map)
def _runAndCatch(f, inp):
    try:
        return f(inp)
    except Exception as e:
        
        # print details
        writeOutput(traceback.format_exc(), 'ErrorLog')
        
        # re-throw e
        raise e
    

# Returns the function lambda x: instance.memberName(x);
# Note: Pool.map() only allows calling pickable functions (ie defined at top module level).
#       This function allows to overcome this and invoke Pool.map() on class functions.
#       (Any changes made to the class by the child process are obviously lost, but return value is attained).
def runMemberFunc(instance, memberName):
    return partial(_runMemberFunc, instance, memberName)
def _runMemberFunc(instance, memberName, arg0):
    return getattr(instance, memberName)(arg0)

# Sets queue q as global variable in this module.
def _setQueue(q):
    global _q
    _q = q

# Logic for output-writer process
def _runOutputWriter(outputPrefix):
    
    outputWriter = OutputWriter(outputPrefix)

    while True:
        mType, line, filename, originProcess = _q.get()
        if mType == 'stop':
            # TODO empty queue first???
            break
        else:
            outputWriter.write(line, filename, originProcess)
            
    outputWriter.close()

# Clean resources at exit
def _atExit():

    # order writer process to finish its task
    _q.put(('stop',None,None,None))
    # order pool processes to terminate
    _p.close()
    # wait for pool processes to terminate
    _p.join()
    
class OutputWriter(object):
    
    def __init__(self, outputPrefix):
        self._outputPrefix = outputPrefix
        self._files        = dict()
    
    def write(self, line, filename, originProcess):
        
        if line[-1] != '\n':
            line = line + '\n'
        # add process ID to debug and error prints:
        if filename in ['DBG', 'ErrorLog']:
            line = '(process %s) '%originProcess + line
    
        # if first use of this file, open it first 
        if filename not in self._files:
            
            # if a file with this name already exists, issue warning
            fullName = self._outputPrefix + filename + '.txt'
            if os.path.exists(fullName):
                warningMsg = 'File %s exists; previous version is being deleted.'%fullName
                if filename == 'ErrorLog':
                    line = line + warningMsg + '\n'
                else:
                    writeOutput(warningMsg, filename = 'ErrorLog')
            
            # open file (deleting existing copy if necessary)
            self._files[filename] = open(fullName, 'w')
        
        # write line to file & flush immediately
        f = self._files[filename]
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        
        # special file streams (log \ error) are also printed to stdout \ stderr
        if filename == 'ErrorLog':
            sys.stderr.write(line)
            sys.stderr.flush()
        elif filename == 'Log':
            sys.stdout.write(line)
            sys.stdout.flush()

    def close(self):
        writeOutput('Closing output-writer', 'DBG')
        for f in self._files.itervalues():
            f.close()
