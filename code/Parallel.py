# This module builds on Pool.map(), providing a pool of worker processes with threading-safe output writing.
# Usage:
#   - Init with initParallel();
#   - Write output messages with writeOutput();
#   - Run parallel tasks with runParallel().

import atexit
import multiprocessing as mp
import sys
import os

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
    
    if _nPrc == 1:
        _outputWriter.write(line, filename)
    
    else:
        _q.put(('msg', line, filename))

# Calculate [f(inp) for inp in inputs] in parallel.
def runParallel(f, inputs):
    
    # Since exceptions within pool.map result in a non-informative exception in the parent process,
    #   for debugging purposes it's sometimes easier to avoid pool.map.
    # I prefer not to call pool.map if nPrc = 1.
    if _nPrc == 1:
        return [f(inp) for inp in inputs]
    
    else:
        return _p.map(f, inputs)

# Return instance.memberName(args0, *args);
# Note: Pool.map() only allows calling pickable functions (ie defined at top module level).
#       This function allows to overcome this and invoke Pool.map() on class functions.
#       (Any changes made to the class by the child process are obviously lost).
def runMemberFunc(arg0, instance, memberName, *args):
    return getattr(instance, memberName)(arg0, *args)

# Sets queue q as global variable in this module.
def _setQueue(q):
    global _q
    _q = q

# Logic for output-writer process
def _runOutputWriter(outputPrefix):
    
    outputWriter = OutputWriter(outputPrefix)

    while True:
        mType, line, filename = _q.get()
        if mType == 'stop':
            # TODO empty queue first???
            break
        else:
            outputWriter.write(line, filename)
            
    outputWriter.close()

# Clean resources at exit
def _atExit():

    # order writer process to finish its task
    _q.put(('stop',None,None))
    # order pool processes to terminate
    _p.close()
    # wait for pool processes to terminate
    _p.join()
    
class OutputWriter(object):
    
    def __init__(self, outputPrefix):
        self._outputPrefix = outputPrefix
        self._files        = dict()
    
    def write(self, line, filename):

        line = line + '\n'
    
        # if first use of this file, open it first 
        if filename not in self._files:
            
            # if a file with this name already exists, issue warning
            fullName = self._outputPrefix + filename + '.txt'
            if os.path.exists(fullName):
                warningMsg = 'File %s exists; previous version is being deleted.'%fullName
                if filename == 'ErrorLog':
                    line = line + warningMsg + '\n'
                else:
                    self.write(warningMsg, filename = 'ErrorLog')
            
            # open file (deleting existing copy if necessary)
            self._files[filename] = open(fullName, 'w')
        
        # write line to file & flush immediately
        f = self._files[filename]
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        
        # special file streams (log \ error) are also printed to scrren
        if filename == 'ErrorLog':
            sys.stderr.write(line)
            sys.stderr.flush()
        elif filename == 'log':
            sys.stdout.write(line)
            sys.stdout.flush()

    def close(self):
        self.write('Closing output-writer', 'DBG')
        for f in self._files.itervalues():
            f.close()
