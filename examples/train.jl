using VQE4NPP
Pkg.add("Random")
STH = 1e-8
PTH = 1e-7
CTH = 1e-3
Niter = 5000

n = 8
Random.seed!(1234)
d = 4
s = 100
set = RandIntNumSet(n,s)
VQEtrain(set, depth = d, StopTH = STH, ConvergeTH=CTH, PerturbTH = PTH, PurtAmp = 0.02, niter=Niter, autoTrain=false)