import os

nruns = 30
nbins = [20]
l2regs = [0,1e-1,5e-1,1]
sdregs = [0,1e-4,1e-3,1e-2]

for nr in range(nruns):
    for nb in nbins:
        for l2reg in l2regs:
            for sdreg in sdregs:
                cmd = 'bsub -q atlas-t3 -W 80:00 -o $PWD/output/%(nr)s_%(nb)s $PWD/startup_anaconda.sh "python $PWD/traintoys.py --runNumber %(nr)s --nSamples %(nb)s --l2Reg %(l2reg)s --sdReg %(sdreg)s"'%{'nr':nr,'nb':nb,'l2reg':l2reg,'sdreg':sdreg}
                os.system(cmd)
