# The purpose of this package is to provide a quick way to generate a mock
# galaxy catalog.
#
# This version can use either FFTW or PencilFFTs.jl to distribute the density
# field over several nodes. PencilFFTs.jl, in turn, uses MPI for this task.


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


using .Splines

using QuadOsc
using QuadGK


# common functions
j0(x) = sinc(x/π)  # this function is used in some included files

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

    proc_dims = tuple(MPI.Dims_create!(MPI.Comm_size(comm), zeros(Int, 2))...)
    transform = Transforms.RFFT()
    @show proc_dims

    @time rfftplan = PencilFFTPlan((nxyz...,), transform, proc_dims, comm)
    return rfftplan
end

default_plan = plan_with_fftw
#default_plan = plan_with_pencilffts



#################### draw deltak ###########################

function draw_phases(rfftplan; rng=Random.GLOBAL_RNG)
    deltar = allocate_input(rfftplan)
    #@show size(deltar),length(deltar)
    randn!(rng, parent(deltar))
    #@show mean(deltar),var_global(deltar)
    @assert !isnan(mean(deltar))

    @time deltak_phases = rfftplan * deltar
    #@show mean(deltak_phases)
    @assert !isnan(mean(deltak_phases))

    @. deltak_phases /= abs(deltak_phases)
    #@show mean(deltak_phases)
    @assert !isnan(mean(deltak_phases))
    #@show sizeof_global(deltak_phases)/1024^3
    #@show sizeof(rfftplan)
    return deltak_phases
end


function calc_kmode(nx, ny, nz, kF, pencil_δk)
    nx2 = div(nx,2) + 1
    ny2 = div(ny,2) + 1
    nz2 = div(nz,2) + 1
    kmode = allocate_array(pencil_δk, Float64)
    localrange = range_local(kmode)
    for k=1:size(kmode,3), j=1:size(kmode,2), i=1:size(kmode,1)
        ig = localrange[1][i]  # global index of local index i
        jg = localrange[2][j]  # global index of local index j
        kg = localrange[3][k]  # global index of local index k
        ikx = ig - 1
        iky = jg <= ny2 ? jg-1 : jg-1-ny
        ikz = kg <= nz2 ? kg-1 : kg-1-nz
        kmode[i,j,k] = kF * √(ikx^2 + iky^2 + ikz^2)
    end
    return kmode
end


####################### calculate volocity field #################
function calculate_velocities_faH(deltak, kF)
    nx2, ny, nz = size_global(deltak)
    ny2 = div(ny,2) + 1
    nz2 = div(nz,2) + 1
    vkx = deepcopy(deltak)
    vky = deepcopy(deltak)
    vkz = deepcopy(deltak)
    localrange = range_local(deltak)
    for k=1:size(deltak,3), j=1:size(deltak,2), i=1:size(deltak,1)
        ig = localrange[1][i]  # global index of local index i
        jg = localrange[2][j]  # global index of local index j
        kg = localrange[3][k]  # global index of local index k
        ikx = ig - 1
        iky = jg <= ny2 ? jg-1 : jg-1-ny
        ikz = kg <= nz2 ? kg-1 : kg-1-nz
        ikmode = √(ikx^2 + iky^2 + ikz^2)
        if ikmode == 0
            vkx[i,j,k] = 0
            vky[i,j,k] = 0
            vkz[i,j,k] = 0
        else
            vk = im/ikmode^2/kF*deltak[i,j,k]
            vkx[i,j,k] = ikx*vk
            vky[i,j,k] = iky*vk
            vkz[i,j,k] = ikz*vk
        end
    end
    return vkx, vky, vkz
end


##################### draw galaxies ###########################
function draw_galaxies_with_velocities(deltar, vx, vy, vz, Ngalaxies, Δx=1.0;
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
        end
        for n=1:Nthiscell
            x = ig - 1 + rand(rng)
            y = jg - 1 + rand(rng)
            z = kg - 1 + rand(rng)
            xyzv[g0+1] = x*Δx
            xyzv[g0+2] = y*Δx
            xyzv[g0+3] = z*Δx
            if rsd
                xyzv[g0+4] = vx[i,j,k]
                xyzv[g0+5] = vy[i,j,k]
                xyzv[g0+6] = vz[i,j,k]
            end # else xyzv[4:6] = 0
            g0 += 6
        end
        Ngalaxies_local_actual += Nthiscell
    end
    xyzv_out = reshape(xyzv, 6, :)
    return xyzv_out[:,1:Ngalaxies_local_actual]
end


######################### make galaxies file ######################

# write galaxies in same format as lognormal_galaxies
function write_galaxies(fname, LLL, xyzv)
    Ngalaxies = size(xyzv,2)
    fout = open(fname, "w")
    write(fout, Array{Float64}(collect(LLL)))
    write(fout, Int64(Ngalaxies))
    write(fout, Array{Float32}(xyzv))
    close(fout)
end


# read galaxies in same format as lognormal_galaxies
function read_galaxies(fname; ncol=6)
    fin = open(fname, "r")
    Lx, Ly, Lz = read!(fin, Array{Float64}(undef, 3))
    Ngalaxies = read(fin, Int64)
    xyzv = read!(fin, Array{Float32,2}(undef, ncol, Ngalaxies))
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
    localrange = range_local(deltar)
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

# simulate galaxies
function simulate_galaxies(nxyz, Lxyz, Ngalaxies, pk, kF, Δx, b, faH; rfftplan=default_plan(nxyz), rng=Random.GLOBAL_RNG)
    nx, ny, nz = nxyz
    Lx, Ly, Lz = Lxyz
    Volume = Lx * Ly * Lz

    println("Convert pk to log-normal pkG...")
    #kln = readdlm("$root/data/fog_r1000_pkG.dat")[:,1]
    #pkGln = readdlm("$root/data/fog_r1000_pkG.dat")[:,2]
    @time kGm, pkGm = pk_to_pkG(pk)
    @time kGg, pkGg = pk_to_pkG(k -> b^2 * pk(k))
    #@show pkGm.([0.0,1.0])
    #@show pkGg.([0.0,1.0])

    println("Draw random phases...")
    @time deltak_phases = draw_phases(rfftplan; rng)
    #@show get_rank(),deltak_phases[1,1,1],mean(deltak_phases)

    println("Calculate kmode...")
    @time kmode = calc_kmode(nx, ny, nz, kF, pencil(deltak_phases))
    #@show get_rank(),kmode[1,1,1],mean(kmode)
    Volume = (2π / kF)^3

    println("Calculate deltak{m,g}...")
    deltakm = deepcopy(deltak_phases)
    deltakg = deltak_phases
    #@show get_rank(),deltakm[1,1,1],mean(deltakm)
    #@show get_rank(),deltakg[1,1,1],mean(deltakg)
    @time @. deltakm *= √(pkGm(kmode) * Volume)
    for i in eachindex(kmode)
        if pkGg(kmode[i]) < 0
	    @error "negative pkGg" i kmode[i] pkGg(kmode[i])
            error("negative pkGg")
        end
    end
    @time @. deltakg *= √(pkGg(kmode) * Volume)
    ##@time pixel_window!(deltakm, nxyz)
    #@time pixel_window!(deltakg, nxyz)
    deltak_phases = nothing
    kmode = nothing
    #@show get_rank(),deltakm[1,1,1],mean(deltakm)
    #@show get_rank(),deltakg[1,1,1],mean(deltakg)

    println("Calculate deltar{m,g}...")
    @time deltarm = rfftplan \ deltakm
    @time deltarg = rfftplan \ deltakg
    #@show get_rank(),"interim",deltarm[1,1,1],mean(deltakm)
    #@show get_rank(),"interim",deltarg[1,1,1],mean(deltakg)
    @. deltarm *= (nx*ny*nz) / Volume
    @. deltarg *= (nx*ny*nz) / Volume
    #@show get_rank(),deltarm[1,1,1],mean(deltakm)
    #@show get_rank(),deltarg[1,1,1],mean(deltakg)
    deltakg = nothing
    #@show mean(deltarm),std(deltarm)
    #@show extrema(deltarm)
    #@show mean(deltarg),std(deltarg)
    #@show extrema(deltarg)

    println("Transform G → δ...")
    @time σGm² = var_global(deltarm)
    @time σGg² = var_global(deltarg)
    #σGm²_th = calculate_sigmaGsq(pkGm, prod(Lxyz ./ nxyz))
    #σGg²_th = calculate_sigmaGsq(pkGg, prod(Lxyz ./ nxyz))
    #@show σGm²,σGm²_th,σGm²/σGm²_th
    #@show σGg²,σGg²_th,σGg²/σGg²_th
    #@show var(deltarg)
    #return
    @time @. deltarm = exp(deltarm - σGm²/2) - 1
    @time @. deltarg = exp(deltarg - σGg²/2) - 1
    #@show σGm² σGg²
    #@show mean(deltarm),std(deltarm)
    #@show extrema(deltarm)
    #@show mean(deltarg),std(deltarg)
    #@show extrema(deltarg)

    if faH != 0
        println("Calculate deltakm...")
        @time deltakm = rfftplan * deltarm
        @. deltakm *= Volume / (nx*ny*nz)
        deltarm = nothing
        println("Calculate v⃗(k⃗)...")
        @time vkx, vky, vkz = calculate_velocities_faH(deltakm, kF)
        deltakm = nothing  # free memory
        println("Calculate v⃗(r⃗)...")
        @time vx = rfftplan \ vkx
        @. vx *= faH * (nx*ny*nz) / Volume
        vkx = nothing  # free memory
        @time vy = rfftplan \ vky
        @. vy *= faH * (nx*ny*nz) / Volume
        vky = nothing  # free memory
        @time vz = rfftplan \ vkz
        @. vz *= faH * (nx*ny*nz) / Volume
        vkz = nothing  # free memory
    else
        vx = vy = vz = 0
    end

    println("Draw galaxies...")
    @time xyzv = draw_galaxies_with_velocities(deltarg, vx, vy, vz, Ngalaxies, Δx; rng)
    return xyzv
end


function simulate_galaxies(nbar, Lbox, pk; nmesh=256, bias=1.0, f=0.0,
        rfftplanner=default_plan, rng=Random.GLOBAL_RNG)
    aH = 1
    b = bias
    L = Lbox
    Ngalaxies = ceil(Int, L^3 * nbar)

    n = nmesh
    kF = 2π/L
    Δx = L/n

    nxyz = n, n, n
    Lxyz = L, L, L

    rfftplan = rfftplanner(nxyz)

    xyzv = simulate_galaxies(nxyz, Lxyz, Ngalaxies, pk, kF, Δx, b, f;
                                   rfftplan, rng)
    xyz = @. xyzv[1:3,:] - Float32(L / 2)
    v = xyzv[4:6,:]
    return xyz, v
end


end


# vim: set sw=4 et sts=4 :
