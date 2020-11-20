# LogNormalGalaxies.jl


To install, first press `]` at the REPL to get into the package mode. Then, add the registry and the package,
```julia
pkg> regsitry add https://github.com/Wide-Angle-Team/WATCosmologyJuliaRegistry.git
...
pkg> add LogNormalGalaxies
```

A sample script is in `test/test_lognormals.jl`. On the REPL it can be called with
```julia
include("path/to/LogNormalGalaxies/test/test_lognormals.jl")
```

May need to configure [MPI.jl](https://juliaparallel.github.io/MPI.jl/stable/configuration/).
