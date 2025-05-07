# LogNormalGalaxies.jl


This Julia package implements a simple log-normal galaxies simulation. It is
heavily inspired by the
[log_normal_galaxies](https://bitbucket.org/komatsu5147/lognormal_galaxies/src/master/)
code introduced in [Agrawal et al. (2017)](https://arxiv.org/abs/1706.09195),
where more details on log-normal simulations in general can be found.


## Installation

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

May need to configure
[MPI.jl](https://juliaparallel.github.io/MPI.jl/stable/configuration/).
(Usually, `]build MPI` is sufficient.)


## Usage

To generate a log-normal catalog, use `simulate_galaxies()`:
```julia
julia> using LogNormalGalaxies

julia> nbar = 3e-4  # average number density per (Mpc/h)^3

julia> Lbox = 2048.0  # box side length in Mpc/h

julia> pk(k)  # a power spectrum.

julia> x, psi = simulate_galaxies(nbar, Lbox, pk; nmesh=256, bias=1.0)
```
The parameter `nmesh` is the size of the mesh and `bias` is the linear galaxy
bias. `x` is of size `(3,num_galaxies)` and the position vector of the galaxy
`i` is `x[:,i]`. `psi` is the displacement field.

There is a named parameter `f` that is used to multiply the displacement
field. It is best though of as a boolean. It should be left at `f=true`. If
`f=false`, then the relatively expensive calculation to generate `psi` can be
skipped (since `psi=0` in that case). This is useful if RSD is not needed.

Redshift space distortions can be added with code like
```julia
num_galaxies = size(x,2)
los = [0, 0, 1]
for i=1:num_galaxies
    x[:,i] .+= f * (psi[:,i]' * los) * los
end
```
where we assumed the line of sight along the z-axis, and `f` is the linear
growth rate `f=d\ln D/d\ln a`.

It is a good idea to generate a catalog in a box that is somewhat larger than
desired, and, after applying RSD, cut it down to the desired size. This is
because RSD will always leave a region near the edges of the box incomplete as
galaxies get scattered outside the box, but no galaxies get scattered inside
the box.


## List of Papers

This is a list of papers using this code:

1. [Grasshorn Gebhardt & Jeong (2020)](https://arxiv.org/abs/2008.08706): $C_\ell$ bins and RSD with FoG
2. [Grasshorn Gebhardt & Doré (2021)](https://arxiv.org/abs/2102.10079): [SuperFaB](https://github.com/hsgg/SphericalFourierBesselDecompositions.jl)
3. [Grasshorn Gebhardt & Doré (2021)](https://arxiv.org/abs/2109.13352): [CryoFunk/CryoFaBs.jl](https://github.com/hsgg/CryoFaBs.jl)
4. [Grasshorn Gebhardt & Doré (2024)](https://arxiv.org/abs/2310.17677): SuperFaB Validation on eBOSS
5. [Benabou & Sands *etal.* (2024)](https://arxiv.org/abs/2404.04811): Perturbative Wide-Angle effects in PS
6. [Wen *etal.* (2024)](https://arxiv.org/abs/2404.04812): Nonperturbative Wide-Angle effects in PS
