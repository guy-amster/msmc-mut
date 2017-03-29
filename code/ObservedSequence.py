import numpy as np
import itertools
import re


# This class holds an observed sequence 
class ObservedSequence(object):
    

    # seqLength        : sequence length (eg chromosome length in base-pairs).
    # nonZeroPositions : A sorted list of all the (0-based) positions in which the emitted observation is NOT zero.
    #                    (the observation being a value between 0 and (nEmissions-1) )
    #                    (eg, all non-hom. positions, if hom. is defined as 0)
    # nonZeroValues    : A list of the emitted observations in the positions described by nonZeroPositions.
    def __init__(self, seqLength, nonZeroPositions, nonZeroValues, seqName = None):
        
        if seqName is not None:
            self.seqName = seqName
        
        # sequence length
        self.length = seqLength
        
        assert seqLength > 1
        # if larger positions are required, change uint32 type below...
        assert seqLength < (2**32)
        
        # assert nonZeroPositions list is valid
        assert len(nonZeroValues) == len(nonZeroPositions)
        if len(nonZeroPositions) > 0:
            assert nonZeroPositions[0] >= 0
            for i in xrange(1, len(nonZeroPositions)):
                assert nonZeroPositions[i-1] < nonZeroPositions[i]
            assert nonZeroPositions[-1] < seqLength   
        
            # if more than 256 observation types are required, change uint8 type below...
            assert max(nonZeroValues) <= 255
        
        # TODO optimize size for performance
        self.maxDistance = 1000
        
        # self.positions is a sorted list of positions such that:
        #                 - It includes all non-zero positions.
        #                 - The maximum distance between adjacent positions is self.maxDistance.
        #                 - It includes the leftmost (0) & rightmost (length - 1) positions.
        # self.posTypes describes the corresponding types of self.positions
        
        # the following code is somewhat non-pythonic:
        # (I want to use numpy arrays, and I don't want to use append for potentially large inputs)
        # - in the first pass, calculate the required arrays length & initialize arrays
        # - in the second pass, fill the arrays
        # TODO is this necessary? perhaps use append & cast...
        
        for firstPass in [True, False]:
            
            # n: next index to be filled in the target self.positions array
            n = 0
            
            # p: last position (in 0, ..., self.length - 1) that's already parsed
            p = -1
            
            # add position 0 if it's value is 0
            # (otherwise, it's treated like any other non-zero position)
            addZero = False
            if len(nonZeroPositions) == 0:
                addZero = True
            elif nonZeroPositions[0] > 0:
                addZero = True
            if addZero:
                if not firstPass:
                    self.posTypes [0]  = 0
                    self.positions[0]  = 0
                n = 1
                p = 0
            
            # iterate over all non-zero positions
            for nextNonZeroPos, nextNonZeroVal in itertools.izip(nonZeroPositions, nonZeroValues):
            
                # if the distance to the next non-zero position is too large, add zero positions
                while (p + self.maxDistance) < nextNonZeroPos:
                    p += self.maxDistance
                    if not firstPass:
                        self.positions[n] = p
                        self.posTypes [n] = 0
                    n += 1
                
                # add next non-zero position
                p  = nextNonZeroPos
                if not firstPass:
                    self.positions[n] = p
                    self.posTypes [n] = nextNonZeroVal
                n += 1

            # add zero positions until end of sequence is reached
            while p < (self.length - 1):
                p  = min(p + self.maxDistance, self.length - 1)
                if not firstPass:
                    self.positions[n] = p
                    self.posTypes [n] = 0
                n += 1
            
            # initilize arrays
            if firstPass:
                self.positions = np.empty(n, dtype=np.uint32)
                self.posTypes  = np.empty(n, dtype=np.uint8 )
        
        self.nPositions = n
        
        # update self.maxDistance (in case this specific sequence never reach the predefined maximum distance)
        self.maxDistance = 0
        for i in xrange(self.nPositions - 1):
            self.maxDistance = max(self.maxDistance, self.positions[i+1]-self.positions[i])
        
        
    # read sequence from file
    @classmethod
    def fromFile(cls, filename):
        
        # TODO elaborate on structure;  0 based positions; change format?
        # NOTICE last position must be given as input!
        pattern = re.compile(r"""
                                 ^
                                  (?P<chrN>\w+)\t        # Chromosome name
                                  (?P<pos>\d+)\t         # Position
                                  (?P<called>\d+)\t      # Number of called sites since last position
                                  (?P<seq>[ACGT]{2})     # Called sequence
                                 $
                             """, re.VERBOSE)
        
        chrName   = None
        positions = []
        lastPos  = -1
        
        with open(filename) as f:
            for line in f:
                
                match = pattern.match(line)
                
                # verify valid line structure
                assert match is not None
                
                chrN   =     match.group("chrN"  )
                pos    = int(match.group("pos"   ))
                called = int(match.group("called"))
                seq    =     match.group("seq"   )
                
                # read chr name from first line
                if chrName is None:
                    chrName = chrN
                # verify that all other lines refer to the same chromosome
                else:
                    assert chrName == chrN
                
                assert called        >= 1
                assert pos - lastPos >= called
                
                # TODO handle missing sites....
                assert pos - lastPos == called
                
                assert len(seq) == 2
                if seq[0] != seq[1]:
                    # segregating site
                    positions.append(pos)
                
                lastPos = pos
                
        return cls(lastPos+1, positions, [1 for _ in xrange(len(positions))], chrName)
    
    # read sequence from complete list of observed emissions
    @classmethod
    def fromEmissionsList(cls, observedEmissions):
        
        nonZeroPositions = [pos for pos in xrange(len(observedEmissions)) if observedEmissions[pos] != 0]
        nonZeroValues    = [observedEmissions[pos] for pos in nonZeroPositions]
        return cls(len(observedEmissions), nonZeroPositions, nonZeroValues)
        