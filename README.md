# VQE4NPP

[![Build Status](https://travis-ci.com/frankwswang/VQE4NPP.jl.svg?branch=master)](https://travis-ci.com/frankwswang/VQE4NPP.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/frankwswang/VQE4NPP.jl?svg=true)](https://ci.appveyor.com/project/frankwswang/VQE4NPP-jl)
[![Coverage](https://codecov.io/gh/frankwswang/VQE4NPP.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/frankwswang/VQE4NPP.jl)

__VQE4NPP__ is a julia package for applying a quantum-classical hybrid algorithm called variational quantum eigensolver (VQE) on number partitioning problems.

## Main functions
### RandIntNumSet
Generating a random number set that has an equi-partition solution.

### VQEtrain
Training the VQE to decrease the energy of a Hailtonian corresponding to the target number set. An auto-training option with multiple adjustable thresholds is available.

## Setup Guide
### Julia Environment
* [__Julia 1.4__](https://julialang.org)

### Installation
Type `]` in Julia REPL to enter `Pkg` mode, then type:
```
pkg> add https://github.com/frankwswang/VQE4NPP.jl
```

### Reference
Lucas, A. (2014). Ising formulations of many NP problems. Frontiers in Physics, 2, 5. ([DOI: 10.3389/fphy.2014.00005](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full))

## License
VQE4NPP.jl is released under MIT License.