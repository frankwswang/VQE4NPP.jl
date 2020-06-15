# VQE4NPP.jl
__VQE4NPP__ is a julia package for applying a quantum-classical hybrid algorithm called variational quantum eigensolver (VQE) on number partitioning problems.

## Main functions
### RandIntNumSet
Generating a random number set that has an equi-partition solution.

### VQEtrain
Training the VQE to decrease the energy of a Hailtonian corresponding to the target number set. An auto-train with user-defined thresholds is available.

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