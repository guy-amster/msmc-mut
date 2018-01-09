#!/bin/bash

# store simulated data in sub-folder
mkdir simData
cd simData

# simulate coalescence trees using ms; parameters correspond to sim-2 simulation in PSMC (see their supplement)
# original cmd: ms 2 100 -t 81960 -r 13560 30000000 -eN 0.01 0.05 -eN 0.0375 0.5 -eN 1.25 1
# here we generate trees using ms (thus -T replace -t) and mutations using simMuts.py;
# note that 4*N0*u per-bp = theta (4*N0*u*nSites) / nSites = 81960 / 30000000 = 0.002732
ms 2 100 -T -r 13560 30000000 -eN 0.01 0.05 -eN 0.0375 0.5 -eN 1.25 1
simMuts.py 0.002732 trees.txt -psmc -msmc 
cd ..

# write true history parameters to file
createHist.py -u 0.000000025 -t 81960 -r 13560 30000000 -eN 0.01 0.05 -eN 0.0375 0.5 -eN 1.25 1 -o true.hist

# run PSMC
psmc -N25 -t15 -r5 -p "4+25*2+4+6" -o out.psmc simData/chr.psmc

# extract results from PSMC
# NOTE: the perl scripts used here are distributed with PSMC
/mnt/data/msmc-mut/code/createHist.py -u 0.000000025 -o psmc.hist   \
$(/mnt/data/soft/psmc/utils/psmc2history.pl out.psmc        |       \
/mnt/data/soft/psmc/utils/history2ms.pl                     |       \
awk '{$1= ""; $2 = ""; $3 = ""; sub("-l",""); print $0}') 

# run MSMC

# extract results from MSMC

# run my program
msmc-mut.py -fixedMu -iter 20 simData/chr*.msmc 

# extract results from my program
loop2hist.py -o msmc_u.hist -su 0.000000025 loop.txt

# plot 4 histories
hist2plot.py -p true.hist "True history" -p msmc.hist "MSMC" -p psmc.hist "PSMC" -p msmc_u.hist "MSMC-U" -o sim-1

# TODO plot fit accuracy by iteration

# TODO plot rho-estimate by iteration, for MSMC & my program

