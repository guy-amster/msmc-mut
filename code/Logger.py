import atexit
import sys
import os

_path = ''

_logFiles = dict()

# set a default path for various log files
def setLoggerPath(path):
    global _path
    _path = path

# TODO add locks :(
# write line to aprropriate log-file (opening it first if hadn't already)
def log(line, filename = 'log'):
    
    line = line + '\n'
    
    # if first use of this file, open it first 
    if filename not in _logFiles:
        
        # if a file with this name already exists, issue warning
        fullPath = _path + filename + '.txt'
        if os.path.exists(fullPath):
            warningMsg = 'File %s exists; previous version is being deleted.'%fullPath
            if filename == 'errorLog':
                line = line + warningMsg
            else:
                log(warningMsg, filename = 'errorLog')
        
        # open file (deleting existing copy if necessary)
        _logFiles[filename] = open(fullPath, 'w')
    
    # write line to file & flush immediately
    f = _logFiles[filename]
    f.write(line)
    f.flush()
    os.fsync(f.fileno())
    
    # special file streams (log \ error) are also printed to scrren
    if filename == 'errorLog':
        sys.stderr.write(line)
        sys.stderr.flush()
    elif filename == 'log':
        sys.stdout.write(line)
        sys.stdout.flush()

def logError(line):
    log(line, filename='errorLog')
    
# close files on program exit
def _closeLogger():
    for f in _logFiles.values():
        f.close()

atexit.register(_closeLogger)
