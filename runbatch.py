import os

nruns = 100
nbins = [20]
l2regs = [0]#,5e-1,6e-1]
sdregs = [0,1e-4,1e-3,1e-2]

for nr in range(nruns):
    for nb in nbins:
        for l2reg in l2regs:
            for sdreg in sdregs:
                cmd = 'bsub -q atlas-t3 -W 80:00 -o $PWD/output/%(nr)s_%(nb)s $PWD/startup_anaconda.sh "python $PWD/traintoys.py --runNumber %(nr)s --nSamples %(nb)s --l2Reg %(l2reg)s --sdReg %(sdreg)s"'%{'nr':nr,'nb':nb,'l2reg':l2reg,'sdreg':sdreg}
                os.system(cmd)

#nruns = 100
#nbins = [3,8,12,18,25]
#for nr in range(nruns):
#    for nb in nbins:
#        cmd = 'bsub -q atlas-t3 -W 80:00 -o $PWD/output/%(nr)s_%(nb)s $PWD/startup_anaconda.sh "python $PWD/traintoys.py --runNumber %(nr)s --nSamples %(nb)s"'%{'nr':nr,'nb':nb}
#        os.system(cmd)
