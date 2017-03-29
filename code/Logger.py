import atexit
import sys

_path = ''

_logFiles = dict()

# set a default path for various log files
def setLoggerPath(path):
    global _path
    _path = path

# write line to aprropriate log-file (opening it first if hadn't already)
def log(line, filename = 'log'):
    if filename not in _logFiles:
        _logFiles[filename] = open(_path + filename + '.txt', 'w')
    _logFiles[filename].write(line + '\n')
    
    if filename == 'errorLog':
        sys.stderr.write(line + '\n')

def logError(line):
    log(line, filename='errorLog')
    
# close files on program exit
def _closeLogger():
    for f in _logFiles.values():
        f.close()

atexit.register(_closeLogger)
