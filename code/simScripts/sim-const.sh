#!/bin/bash

# store simulated data in sub-folder
mkdir simData
cd simData

# simulate coalescence trees using ms; parameters correspond to constant population of size N=10^5/3; u = 1.5 10^-8; r = 10^-8.
# here we generate trees using ms (thus -T replace -t) and mutations using simMuts.py;
ms 2 1000 -T -r 4000 3000000 > trees.txt
sim/simMuts.py 0.002 trees.txt -psmc -msmc
# OR
sim/simMuts.py 0.0015 trees.txt -psmc -msmc  -uT 0.5 0.003
cd ..

# write true history parameters to file
createHist.py -u 0.000000015 -t 6000 -r 4000 3000000 -o true.hist
createHist.py -u 0.000000015 -t 6000 -r 4000 3000000 -o true.hist -uT 0.5 0.003

# run PSMC
psmc -N25 -t15 -r5 -p "4+25*2+4+6" -o out.psmc simData/chrs.psmc

# extract results from PSMC
# NOTE: the perl scripts used here are distributed with PSMC
/mnt/data/msmc-mut/code/createHist.py -u 0.000000015 -o psmc.hist $(/mnt/data/soft/psmc/utils/psmc2history.pl out.psmc | /mnt/data/soft/psmc/utils/history2ms.pl | awk '{$1= ""; $2 = ""; $3 = ""; sub("-l",""); print $0}') 

# run MSMC

# extract results from MSMC

# run my program
msmc-mut.py -fixedMu -iter 20 simData/chr*.msmc 

# extract results from my program
loop2hist.py -o msmc_u.hist -su 0.000000015 loop.txt
loop2hist.py -sn 33333.3 -o msmc_u.hist loop.txt

# plot 4 histories
hist2plot.py -p true.hist "True history" -p fu_msmc.hist "MSMC-segs" -p fu_psmc.hist "PSMC-segs" -p psmc.hist "PSMC" -o fu-plot -p fu_finer.hist "Finer"

# TODO plot fit accuracy by iteration

# TODO plot rho-estimate by iteration, for MSMC & my program

