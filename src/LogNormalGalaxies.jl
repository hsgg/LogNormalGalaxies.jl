# The purpose of this package is to provide a quick way to generate a mock
# galaxy catalog.
#
# This version can use either FFTW or PencilFFTs.jl to distribute the density
# field over several nodes. PencilFFTs.jl, in turn, uses MPI for this task.

# TODO:
# * support general rectangular cuboid boxes


module LogNormalGalaxies

export simulate_galaxies,
       read_galaxies,
       write_galaxies


include("Splines.jl")

using Printf
using FFTW
using MPI
using PencilFFTs
using Statistics
using PoissonRandom
using TwoFAST
using Random
using Strided
using LinearAlgebra


using .Splines

using QuadOsc
using QuadGK


# common functions
j0(x) = sinc(x/π)  # this function is used in multiple locations

# intra-module include files
include("pk_to_pkG.jl")
include("arrays.jl")


######################## misc functions

estimate_memory(N::Integer) = estimate_memory([N,N,N])

function estimate_memory(nxyz::Array)
    nfloats = 9 * prod(nxyz)
    memory = nfloats * sizeof(Float64)
    return memory
end


# choose fft plan

function plan_with_fftw(nxyz)
    return plan_rfft(Array{Float64}(undef, nxyz...))
end

function plan_with_pencilffts(nxyz)
    rank, comm = start_mpi()

    proc_dims = MPI.Dims_create(MPI.Comm_size(comm), zeros(Int, 2))
    proc_dims = tuple(Int64.(proc_dims)...)
    transform = Transforms.RFFT()
    @show proc_dims typeof(proc_dims)

    @time rfftplan = PencilFFTPlan((nxyz...,), transform, proc_dims, comm)
    return rfftplan
end

const default_plan = plan_with_fftw
#const default_plan = plan_with_pencilffts



#################### draw deltak ###########################

function draw_phases(rfftplan; rng=Random.GLOBAL_RNG)
    deltar = allocate_input(rfftplan)
    #@show size(deltar),length(deltar)
    randn!(rng, parent(deltar))
    #@show mean(deltar),var_global(deltar)
    #@assert !isnan(mean(deltar))

    deltak_phases = rfftplan * deltar
    #@show mean(deltak_phases)
    #@assert !isnan(mean(deltak_phases))

    @strided @. deltak_phases /= abs(deltak_phases)
    #@show mean(deltak_phases)
    #@assert !isnan(mean(deltak_phases))
    #@show sizeof_global(deltak_phases)/1024^3
    #@show sizeof(rfftplan)
    return deltak_phases
end


function calc_kvec(i, j, k, localrange, kF, ny2, nz2, ny, nz)
    ig = localrange[1][i]  # global index of local index i
    jg = localrange[2][j]  # global index of local index j
    kg = localrange[3][k]  # global index of local index k
    kx = kF[1] * (ig - 1)
    ky = kF[2] * (jg <= ny2 ? jg-1 : jg-1-ny)
    kz = kF[3] * (kg <= nz2 ? kg-1 : kg-1-nz)
    return kx, ky, kz
end


function iterate_kspace(func, deltak, kF; usethreads=false)
    nx2, ny, nz = size_global(deltak)
    ny2 = div(ny,2) + 1
    nz2 = div(nz,2) + 1
    localrange = range_local(deltak)

    if usethreads
        Threads.@threads for k=1:size(deltak,3)
            for j=1:size(deltak,2), i=1:size(deltak,1)
                kvec = calc_kvec(i, j, k, localrange, kF, ny2, nz2, ny, nz)
                func(i, j, k, kvec)
            end
        end
    else
        for k=1:size(deltak,3), j=1:size(deltak,2), i=1:size(deltak,1)
            kvec = calc_kvec(i, j, k, localrange, kF, ny2, nz2, ny, nz)
            func(i, j, k, kvec)
        end
    end

    return deltak
end


function multiply_by_pk!(deltak, pkfn, kF, Volume)
    iterate_kspace(deltak, kF; usethreads=false) do i,j,k,kvec
        kx, ky, kz = kvec
        kmode = √(kx^2 + ky^2 + kz^2)
        pk = pkfn(kmode)  # not thread-safe
        deltak[i,j,k] *= √(pk * Volume)
    end
    return deltak
end


function calc_velocity_component!(deltak, kF, coord)
    iterate_kspace(deltak, kF; usethreads=true) do i,j,k,kvec
        kx, ky, kz = kvec
        kmode2 = kx^2 + ky^2 + kz^2
        if kmode2 == 0
            deltak[i,j,k] = 0
        else
            deltak[i,j,k] *= im * kvec[coord] / kmode2
        end
    end
    return deltak
end


##################### draw galaxies ###########################
function draw_galaxies_with_velocities(deltar, vx, vy, vz, Ngalaxies, Δx=[1.0,1.0,1.0];
        rng=Random.GLOBAL_RNG)
    T = Float32
    rsd = !(vx == vy == vz == 0)
    nx, ny, nz = size_global(deltar)
    Navg = Ngalaxies / (nx * ny * nz)
    xyzv = fill(T(0), 6*Ngalaxies)
    localrange = range_local(deltar)
    Ngalaxies_local_actual = 0
    for k=1:size(deltar,3), j=1:size(deltar,2), i=1:size(deltar,1)
        ig = localrange[1][i]  # global index of local index i
        jg = localrange[2][j]  # global index of local index j
        kg = localrange[3][k]  # global index of local index k
        Nthiscell = pois_rand(rng, (1 + deltar[i,j,k]) * Navg)
        g0 = 6 * Ngalaxies_local_actual  # g0 is the index in the xyzv 1D-array
        if g0 + 6*Nthiscell > length(xyzv)
            resize!(xyzv, length(xyzv) + 6*Nthiscell)
            xyzv[g0:end] .= 0  # in case rsd=false
        end
        for _=1:Nthiscell
            x = ig - 1 + rand(rng)
            y = jg - 1 + rand(rng)
            z = kg - 1 + rand(rng)
            xyzv[g0+1] = x*Δx[1]
            xyzv[g0+2] = y*Δx[2]
            xyzv[g0+3] = z*Δx[3]
            if rsd
                xyzv[g0+4] = vx[i,j,k]
                xyzv[g0+5] = vy[i,j,k]
                xyzv[g0+6] = vz[i,j,k]
            end # else xyzv[4:6] = 0
            g0 += 6
        end
        Ngalaxies_local_actual += Nthiscell
    end
    resize!(xyzv, 6 * Ngalaxies_local_actual)
    xyzv_out = reshape(xyzv, 6, :)
    return xyzv_out
end


######################### make galaxies file ######################

# write galaxies in same format as lognormal_galaxies
function write_galaxies(fname, LLL, xyzv)
    Ngalaxies = size(xyzv,2)
    fout = open(fname, "w")
    write(fout, convert(Vector{Float64}, LLL))
    write(fout, Int64(Ngalaxies))
    write(fout, convert(Matrix{Float32}, xyzv))
    close(fout)
end


# read galaxies in same format as lognormal_galaxies
function read_galaxies(fname; ncol=6)
    fin = open(fname, "r")
    Lx, Ly, Lz = read!(fin, Vector{Float64}(undef, 3))
    Ngalaxies = read(fin, Int64)
    xyzv = read!(fin, Matrix{Float32}(undef, ncol, Ngalaxies))
    close(fin)
    return (Lx, Ly, Lz), xyzv
end


# var_global(): Calculate variance of the given array, taking care of proper
# handling of distributed arrays such as PencilArrays.
function var_global(arr, comm=MPI.COMM_WORLD)
    n = length(arr)
    μ = mean(arr)
    v = var(arr)
    if MPI.Initialized()
        nn = MPI.Allgather(n, comm)
        μμ = MPI.Allgather(μ, comm)
        vv = MPI.Allgather(v, comm)
        n = sum(nn)
        μ = sum(@. nn / n * μμ)
        v = sum(@. (nn - 1) / (n - 1) * vv + nn / (n - 1) * (μμ - μ)^2)
    end
    return v
end


# Fourier transform of a unit sphere.
function W(x)
    if abs(x) < 2e-1
        return @evalpoly(x, 1, 0, -1/10, 0, 1/280, 0, -1/15120, 0, 1/1330560)
    else
        return 3 * (sin(x) - x * cos(x)) / x^3
    end
end


function calculate_sigmaGsq(pk, Vcell)
    R = cbrt(Vcell / (4*π/3))  # this is for spherical cells, but our cells are cubic
    @show Vcell R
    integrand(lnk::T) where {T<:Real} = begin
        k = exp(lnk)
        k3 = k^3
        if !isfinite(k3)
            return T(0)
        end
        return k3 * pk(k) * W(R*k)^2
    end
    I, E = quadgk(integrand, -Inf, Inf)
    σ² = I / (2 * π^2)
    @show σ² E
    return σ²
    #return log(1 + σ²)
end


function pixel_window!(deltak, nxyz)
    #nx = size(deltak)  # nx[1] is actually nx/2
    #nx2 = (nx[1], nx[2] ÷ 2 + 1, nx[3] ÷ 2 + 1)
    nx2 = @. nxyz ÷ 2 + 1
    p = 1  # NGP:1, CIC:2, TSC:3
    localrange = range_local(deltak)
    for k=1:size(deltak,3), j=1:size(deltak,2), i=1:size(deltak,1)
        ig = localrange[1][i]  # global index of local index i
        jg = localrange[2][j]  # global index of local index j
        kg = localrange[3][k]  # global index of local index k
        ikx = ig - 1
        iky = jg - 1 - (jg <= nx2[2] ? 0 : nxyz[2])
        ikz = kg - 1 - (kg <= nx2[3] ? 0 : nxyz[3])
        Wmesh = (j0(π*ikx/nxyz[1]) * j0(π*iky/nxyz[2]) * j0(π*ikz/nxyz[3]))^p
        deltak[i,j,k] /= Wmesh
    end
end


################## simulate_galaxies() ##################
# Here are multiple functions called 'simulate_galaxies()'. They only differ in
# their interface.

function deltak_to_galaxies!(deltarm, deltarg, nxyz, Lxyz, Ngalaxies, pkGm, pkGg; faH=true, rfftplan=default_plan(nxyz), rng=Random.GLOBAL_RNG)
    nx, ny, nz = nxyz
    Lx, Ly, Lz = Lxyz
    Volume = Lx * Ly * Lz
    Δx = Lxyz ./ nxyz
    kF = 2*π ./ Lxyz

    if faH != 0
        # Note: In this section we ignore the Volume/(nx*ny*nz) multiplication
        # because it cancels.

        # Note2: The velocity should be multiplied by faH. We do that later. In
        # any case, we are really calculating the displacement field, so faH=1.

        println("Calculate deltakm...")
        @time deltakm = rfftplan * deltarm
        #@time @strided @. deltakm *= Volume / (nx*ny*nz)
        deltarm = nothing

        println("Calculate vx...")
        @time vki = deepcopy(deltakm)
        @time calc_velocity_component!(vki, kF, 1)
        @time vx = rfftplan \ vki
        #@time @strided @. vx *= faH * (nx*ny*nz) / Volume

        println("Calculate vy...")
        @time @strided @. vki = deltakm
        @time calc_velocity_component!(vki, kF, 2)
        @time vy = rfftplan \ vki
        #@time @strided @. vy *= faH * (nx*ny*nz) / Volume

        println("Calculate vz...")
        @time @strided @. vki = deltakm
        @time calc_velocity_component!(vki, kF, 3)
        @time vz = rfftplan \ vki
        #@time @strided @. vz *= faH * (nx*ny*nz) / Volume

        vki = nothing  # free memory
        deltakm = nothing  # free memory
    else
        vx = vy = vz = 0
    end

    println("Draw galaxies...")
    @time xyzv = draw_galaxies_with_velocities(deltarg, vx, vy, vz, Ngalaxies, Δx; rng)

    if faH != 1 && faH != 0
        @time @strided @. xyzv[4:6,:] *= faH
    end

    @show Sys.maxrss() / 2^30
    return xyzv
end


# simulate galaxies
function simulate_galaxies(nxyz, Lxyz, Ngalaxies, pk, b, faH; rfftplan=default_plan(nxyz), rng=Random.GLOBAL_RNG, extra_phases=nothing)
    Volume = prod(Lxyz)
    nx, ny, nz = nxyz
    kF = 2*π ./ Lxyz

    println("Convert pk to log-normal pkG...")
    @time kGm, pkGm = pk_to_pkG(pk)
    @time kGg, pkGg = pk_to_pkG(k -> b^2 * pk(k))

    println("Draw random phases...")
    @time deltakm_init = draw_phases(rfftplan; rng)

    allphases = [0.0]
    if !isnothing(extra_phases)
        append!(allphases, extra_phases)
    end

    xyzv = []
    for phase in allphases
        println("Calculating phase=$phase...")
        deltakm = exp(im*phase) .* deltakm_init

        println("Calculate deltak{m,g}...")
        @time deltakg = deepcopy(deltakm)
        @time multiply_by_pk!(deltakm, pkGm, kF, Volume)
        @time multiply_by_pk!(deltakg, pkGg, kF, Volume)

        println("Calculate deltar{m,g}...")
        @time deltarm = rfftplan \ deltakm
        @time deltarg = rfftplan \ deltakg
        @time @strided @. deltarm *= (nx*ny*nz) / Volume
        @time @strided @. deltarg *= (nx*ny*nz) / Volume
        deltakm = nothing
        deltakg = nothing

        println("Transform G → δ...")
        @time σGm² = var_global(deltarm)
        @time σGg² = var_global(deltarg)
        σGm²_th = calculate_sigmaGsq(pkGm, prod(Lxyz ./ nxyz))
        σGg²_th = calculate_sigmaGsq(pkGg, prod(Lxyz ./ nxyz))
        @show σGm², σGg²
        @show σGm²_th, σGg²_th
        @time @strided @. deltarm = exp(deltarm - σGm²/2) - 1
        @time @strided @. deltarg = exp(deltarg - σGg²/2) - 1

        xyzvi = deltak_to_galaxies!(deltarm, deltarg, nxyz, Lxyz, Ngalaxies, pkGm, pkGg; faH, rfftplan, rng)
        push!(xyzv, xyzvi)
    end

    if isnothing(extra_phases)
        return xyzv[1]
    end
    return xyzv
end


function simulate_galaxies(nbar, Lbox, pk; nmesh=256, bias=1.0, f=false,
        rfftplanner=default_plan, rng=Random.GLOBAL_RNG, extra_phases=nothing,
        gather=true)

    if nmesh isa Number
        nxyz = nmesh, nmesh, nmesh
    else
        nxyz = nmesh
    end

    if Lbox isa Number
        Lxyz = Lbox, Lbox, Lbox
    else
        Lxyz = Lbox
    end

    Volume = prod(Lxyz)
    Ngalaxies = ceil(Int, Volume * nbar)

    @time rfftplan = rfftplanner(nxyz)

    @time xyzv = simulate_galaxies(nxyz, Lxyz, Ngalaxies, pk, bias, f;
                                   rfftplan, rng, extra_phases)

    if isnothing(extra_phases)
        xyzv = [xyzv]
    end

    println("Post-processing...")
    catalogs = ()
    for xv in xyzv
        @time x = @. xv[1:3,:] - Float32(Lbox / 2)
        @time v = xv[4:6,:]
        if gather
            x = concatenate_mpi_arr(x)
            v = concatenate_mpi_arr(v)
        end
        catalogs = (catalogs..., (x, v))
    end

    if isnothing(extra_phases)
        return catalogs[1]
    end
    return catalogs
end


end


# vim: set sw=4 et sts=4 :
