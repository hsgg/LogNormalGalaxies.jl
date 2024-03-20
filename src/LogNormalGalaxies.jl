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
       write_galaxies,
       apply_rsd!,
       apply_periodic_boundaries!


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


using Splines

using QuadOsc
using QuadGK


# common functions
j0(x) = sinc(x/π)  # this function is used in multiple locations

# intra-module include files
include("pk_to_pkG.jl")
include("arrays.jl")
include("LinearInterpolations.jl")
include("apply_rsd.jl")

using .LinearInterpolations


######################## misc functions

estimate_memory(N::Integer) = estimate_memory([N,N,N])

function estimate_memory(nxyz::Array)
    nfloats = 9 * prod(nxyz)
    memory = nfloats * sizeof(Float64)
    return memory
end


# choose fft plan

function plan_with_fftw(nxyz; kwargs...)
    return plan_rfft(Array{Float64}(undef, nxyz...); kwargs...)
end

function plan_with_pencilffts(nxyz; kwargs...)
    rank, comm = start_mpi()

    proc_dims = MPI.Dims_create(MPI.Comm_size(comm), zeros(Int, 2))
    proc_dims = tuple(Int64.(proc_dims)...)
    transform = Transforms.RFFT()
    @show proc_dims typeof(proc_dims)

    @time rfftplan = PencilFFTPlan((nxyz...,), transform, proc_dims, comm; kwargs...)
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
    NNN = prod(size(deltar))
    @strided @. deltak_phases /= √NNN
    #@show mean(deltak_phases)
    #@assert !isnan(mean(deltak_phases))

    #@show mean(deltar)
    #@show var(deltar)
    #@show mean(deltak_phases)
    #@show var(deltak_phases)
    #@show var(real(deltak_phases)) var(imag(deltak_phases))

    #@strided @. deltak_phases /= abs(deltak_phases)
    #@show mean(deltak_phases)
    #@show var(deltak_phases)
    #@show var(real(deltak_phases)) var(imag(deltak_phases))
    return deltak_phases
end


function calc_global_indices(ijk_local, localrange, nx2, ny2, nz2, nx, ny, nz)
    ig = localrange[1][ijk_local[1]]  # global index of local index i
    jg = localrange[2][ijk_local[2]]  # global index of local index j
    kg = localrange[3][ijk_local[3]]  # global index of local index k
    ig = ig <= nx2 ? ig-1 : ig-1-nx
    jg = jg <= ny2 ? jg-1 : jg-1-ny
    kg = kg <= nz2 ? kg-1 : kg-1-nz
    return ig, jg, kg
end


function iterate_kspace(func, deltak; usethreads=false, first_half_dimension=true)
    nx, ny, nz = size_global(deltak)
    nx2 = first_half_dimension ? nx : (div(nx,2) + 1)
    ny2 = div(ny,2) + 1
    nz2 = div(nz,2) + 1
    localrange = range_local(deltak)

    if usethreads
        Threads.@threads for k=1:size(deltak,3)
            for j=1:size(deltak,2), i=1:size(deltak,1)
                ijk_local = (i, j, k)
                ijk_global = calc_global_indices(ijk_local, localrange, nx2, ny2, nz2, nx, ny, nz)
                func(ijk_local, ijk_global)
            end
        end
    else
        for k=1:size(deltak,3), j=1:size(deltak,2), i=1:size(deltak,1)
            ijk_local = (i, j, k)
            ijk_global = calc_global_indices(ijk_local, localrange, nx2, ny2, nz2, nx, ny, nz)
            func(ijk_local, ijk_global)
        end
    end

    return deltak
end

iterate_rspace(args...; kwargs...) = iterate_kspace(args...; kwargs..., first_half_dimension=false)


function multiply_by_pk!(deltak, pkfn, kF::Tuple, Volume)
    iterate_kspace(deltak; usethreads=false) do ijk_local,ijk_global
        kx, ky, kz = kF .* ijk_global
        kmode = √(kx^2 + ky^2 + kz^2)
        pk = pkfn(kmode)  # not thread-safe
        deltak[ijk_local...] *= √(pk * Volume)
    end
    return deltak
end

multiply_by_pk!(deltak, pkfn, kF, Volume) = multiply_by_pk!(deltak, pkfn, (kF...,), Volume)


function calc_velocity_component!(deltak, kF::Tuple, coord)
    iterate_kspace(deltak; usethreads=true) do ijk_local,ijk_global
        kvec = kF .* ijk_global
        kx, ky, kz = kvec
        kmode2 = kx^2 + ky^2 + kz^2
        if kmode2 == 0
            deltak[ijk_local...] = 0
        else
            deltak[ijk_local...] *= im * kvec[coord] / kmode2
        end
    end
    return deltak
end

calc_velocity_component!(deltak, kF, coord) = calc_velocity_component!(deltak, (kF...,), coord)


##################### draw galaxies ###########################
function draw_galaxies_with_velocities(deltar, vx, vy, vz, Navg, Ngalaxies, Δx,
        ::Val{do_rsd}, ::Val{voxel_window_power}, ::Val{velocity_assignment};
        rng=Random.GLOBAL_RNG) where {do_rsd,voxel_window_power,velocity_assignment}
    T = Float64

    xyzv = fill(T(0), 6 * ceil(Int, Ngalaxies + 3 * √Ngalaxies))  # mean + 3 * stddev
    localrange = range_local(deltar)
    Ngalaxies_local_actual = 0

    pnvx = PeriodicNeighborView3D([0,0,0], vx)
    pnvy = PeriodicNeighborView3D([0,0,0], vy)
    pnvz = PeriodicNeighborView3D([0,0,0], vz)
    interp_vx = LinearInterpolation(3)
    interp_vy = LinearInterpolation(3)
    interp_vz = LinearInterpolation(3)

    for k=1:size(deltar,3), j=1:size(deltar,2), i=1:size(deltar,1)
        ig = localrange[1][i]  # global index of local index i
        jg = localrange[2][j]  # global index of local index j
        kg = localrange[3][k]  # global index of local index k

        Nthiscell = pois_rand(rng, (1 + deltar[i,j,k]) * Navg)

        g0 = 6 * Ngalaxies_local_actual  # g0 is the index in the xyzv 1D-array
        if g0 + 6 * Nthiscell > length(xyzv)
            resize!(xyzv, length(xyzv) + 6*Nthiscell)
            xyzv[(g0+1):end] .= 0  # in case do_rsd=false
        end

        for _=1:Nthiscell
            # center of cell ig=1 is at x=0.5*Δx
            x = ig - 0.5
            y = jg - 0.5
            z = kg - 0.5
            for _ = 1:voxel_window_power
                x += rand(rng) - 0.5
                y += rand(rng) - 0.5
                z += rand(rng) - 0.5
            end

            xyzv[g0+1] = x*Δx[1]
            xyzv[g0+2] = y*Δx[2]
            xyzv[g0+3] = z*Δx[3]

            if do_rsd
                if velocity_assignment == 0
                    # Current grid point
                    xyzv[g0+4] = vx[i,j,k]
                    xyzv[g0+5] = vy[i,j,k]
                    xyzv[g0+6] = vz[i,j,k]
                elseif velocity_assignment == 1
                    # Nearest grid point
                    # If voxel_window_power>=2, then maybe we want to use the
                    # nearest grid point for the velocity:
                    ijk = @. round(Int, (x,y,z) + 0.5) + 0
                    xyzv[g0+4] = vx[mod1(ijk[1],end), mod1(ijk[2],end), mod1(ijk[3],end)]
                    xyzv[g0+5] = vy[mod1(ijk[1],end), mod1(ijk[2],end), mod1(ijk[3],end)]
                    xyzv[g0+6] = vz[mod1(ijk[1],end), mod1(ijk[2],end), mod1(ijk[3],end)]
                elseif velocity_assignment == 2
                    # Tri-linear interpolation
                    ijk0 = @. floor(Int, (x,y,z) + 0.5)
                    x0 = @. ijk0 - 0.5
                    pnvx.ijk .= ijk0
                    pnvy.ijk .= ijk0
                    pnvz.ijk .= ijk0
                    LinearInterpolation!(interp_vx, x0, pnvx)
                    LinearInterpolation!(interp_vy, x0, pnvy)
                    LinearInterpolation!(interp_vz, x0, pnvz)
                    xyzv[g0+4] = interp_vx((x,y,z))
                    xyzv[g0+5] = interp_vy((x,y,z))
                    xyzv[g0+6] = interp_vz((x,y,z))
                elseif velocity_assignment == 3
                    # Average of 2x2x2 nearest cells
                    ijk0 = @. floor(Int, (x,y,z) + 0.5)
                    pnvx.ijk .= ijk0
                    pnvy.ijk .= ijk0
                    pnvz.ijk .= ijk0
                    for di=1:2, dj=1:2, dk=1:2
                        xyzv[g0+4] += pnvx[di,dj,dk] / 8
                        xyzv[g0+5] += pnvy[di,dj,dk] / 8
                        xyzv[g0+6] += pnvz[di,dj,dk] / 8
                    end
                elseif velocity_assignment == 4
                    # Average over 1x1x1 cloud
                    ijk0 = @. floor(Int, (x,y,z) + 0.5)
                    pnvx.ijk .= ijk0
                    pnvy.ijk .= ijk0
                    pnvz.ijk .= ijk0
                    tot_weight = 0.0
                    for di=1:2, dj=1:2, dk=1:2
                        xyz_neighbor = @. (ijk0 + (di-1,dj-1,dk-1) - 0.5)
                        xyz_overlap = @. 1 - abs((x,y,z) - xyz_neighbor)
                        volume_overlap = prod(xyz_overlap)
                        xyzv[g0+4] += pnvx[di,dj,dk] * volume_overlap
                        xyzv[g0+5] += pnvy[di,dj,dk] * volume_overlap
                        xyzv[g0+6] += pnvz[di,dj,dk] * volume_overlap
                        tot_weight += volume_overlap
                    end
                    #@show tot_weight
                    @assert tot_weight ≈ 1  # sanity check
                elseif velocity_assignment == 5
                    # Average over 3x3x3 nearest grids
                    ijk0 = @. round(Int, (x,y,z) + 0.5)
                    pnvx.ijk .= ijk0
                    pnvy.ijk .= ijk0
                    pnvz.ijk .= ijk0
                    tot_weight = 0.0
                    for di=0:2, dj=0:2, dk=0:2
                        volume_overlap = 1 / 27
                        xyzv[g0+4] += pnvx[di,dj,dk] * volume_overlap
                        xyzv[g0+5] += pnvy[di,dj,dk] * volume_overlap
                        xyzv[g0+6] += pnvz[di,dj,dk] * volume_overlap
                        tot_weight += volume_overlap
                    end
                    #@show tot_weight
                    @assert tot_weight ≈ 1  # sanity check
                elseif velocity_assignment == 6
                    # Average over 2x2x2 cloud
                    #xyz = (x,y,z)
                    xyz = (ig-0.5, jg-0.5, kg-0.5)
                    ijk0 = @. round(Int, xyz + 0.5)
                    pnvx.ijk .= ijk0
                    pnvy.ijk .= ijk0
                    pnvz.ijk .= ijk0
                    tot_weight = 0.0
                    for di=0:2, dj=0:2, dk=0:2
                        xyz_neighbor = @. (ijk0 + (di-1,dj-1,dk-1) - 0.5)
                        distance = @. abs(xyz - xyz_neighbor)
                        @assert all(distance .<= 1.5)
                        xyz_overlap = @. ifelse(distance > 0.5, 1.5 - distance, 1.0)
                        volume_overlap = prod(xyz_overlap) / 8
                        xyzv[g0+4] += pnvx[di,dj,dk] * volume_overlap
                        xyzv[g0+5] += pnvy[di,dj,dk] * volume_overlap
                        xyzv[g0+6] += pnvz[di,dj,dk] * volume_overlap
                        #println("===")
                        #@show (di,dj,dk) xyz xyz_neighbor distance xyz_overlap volume_overlap
                        tot_weight += volume_overlap
                    end
                    #@show tot_weight
                    @assert tot_weight ≈ 1  # sanity check
                end

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

const file_format = 2


function write_galaxies(fname, LLL, xyzv)
    Ngalaxies = size(xyzv,2)
    ncols = size(xyzv,1)
    fout = open(fname, "w")
    write(fout, Int64(file_format))
    write(fout, Int64(ncols))
    write(fout, Int64(Ngalaxies))
    write(fout, Vector{Float64}([LLL...]))  # explicitly convert Tuple to Vector
    write(fout, convert(Matrix{Float64}, xyzv))
    close(fout)
end


function read_galaxies(fname; ncols=6)
    fin = open(fname, "r")
    file_version = read(fin, Int64)
    if file_version == 2
        ncols = read(fin, Int64)
        Ngalaxies = read(fin, Int64)
        Lx, Ly, Lz = read!(fin, Vector{Float64}(undef, 3))
        xyzv = read!(fin, Matrix{Float64}(undef, ncols, Ngalaxies))
    else
        # read galaxies in same format as lognormal_galaxies
        close(fin)
        fin = open(fname, "r")
        Lx, Ly, Lz = read!(fin, Vector{Float64}(undef, 3))
        Ngalaxies = read(fin, Int64)
        xyzv = read!(fin, Matrix{Float32}(undef, ncols, Ngalaxies))
    end
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


# mean_global(): Calculate mean of the given array, taking care of proper
# handling of distributed arrays such as PencilArrays.
function mean_global(arr, comm=MPI.COMM_WORLD)
    n = length(arr)
    μ = mean(arr)
    if MPI.Initialized()
        nn = MPI.Allgather(n, comm)
        μμ = MPI.Allgather(μ, comm)
        n = sum(nn)
        μ = sum(@. nn / n * μμ)
    end
    return μ
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


function pixel_window!(deltak, nxyz; voxel_window_power=1)
    if voxel_window_power == 0
        return deltak
    end
    iterate_kspace(deltak; usethreads=true) do ijk_local, ijk_global
        Wmesh =  sinc(ijk_global[1] / nxyz[1])
        Wmesh *= sinc(ijk_global[2] / nxyz[2])
        Wmesh *= sinc(ijk_global[3] / nxyz[3])
        Wemsh = Wmesh ^ voxel_window_power
        deltak[ijk_local...] /= Wmesh  # acting on single δ
    end
end


################## simulate_galaxies() ##################
# Here are multiple functions called 'simulate_galaxies()'. They only differ in
# their interface.

# simulate galaxies
function simulate_galaxies(nxyz, Lxyz, nbar, pk, b, faH; rfftplan=default_plan(nxyz), rng=Random.GLOBAL_RNG, voxel_window_power=1, velocity_assignment=1, win=1, sigma_psi=0.0, fixed=false, gather=true)
    nx, ny, nz = nxyz
    Lx, Ly, Lz = Lxyz
    Volume = Lx * Ly * Lz
    Δx = Lxyz ./ nxyz
    kF = 2*π ./ Lxyz

    println("Convert pk to log-normal pkG...")
    @time kGm, pkGm = pk_to_pkG(pk)
    @time kGg, pkGg = pk_to_pkG(k -> b^2 * pk(k))

    println("Draw random phases...")
    @time deltakm = draw_phases(rfftplan; rng)
    if fixed
        @strided @. deltakm /= abs(deltakm)
    end

    println("Calculate deltak{m,g}...")
    @time deltakg = deepcopy(deltakm)
    @time multiply_by_pk!(deltakg, pkGg, kF, Volume)
    @time multiply_by_pk!(deltakm, pkGm, kF, Volume)
    #@time pixel_window!(deltakm, nxyz; voxel_window_power)
    #@time pixel_window!(deltakg, nxyz; voxel_window_power)

    println("Calculate deltar{m,g}...")
    @time deltarm = rfftplan \ deltakm
    @time deltarg = rfftplan \ deltakg
    #@show get_rank(),"interim",deltarm[1,1,1],mean(deltakm)
    #@show get_rank(),"interim",deltarg[1,1,1],mean(deltakg)
    @time @strided @. deltarm *= (nx*ny*nz) / Volume
    @time @strided @. deltarg *= (nx*ny*nz) / Volume
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
    @time @strided @. deltarm = exp(deltarm - σGm²/2) - 1
    @time @strided @. deltarg = exp(deltarg - σGg²/2) - 1
    #@show σGm² σGg²
    #@show mean(deltarm),std(deltarm)
    #@show extrema(deltarm)
    #@show mean(deltarg),std(deltarg)
    #@show extrema(deltarg)

    if faH != 0
        # Note: In this section we ignore the Volume/(nx*ny*nz) multiplication
        # because it cancels.

        # Note2: The velocity should be multiplied by faH. We do that later. In
        # any case, we are really calculating the displacement field, so faH=1.

        println("Calculate deltakm...")
        @time mul!(deltakm, rfftplan, deltarm)
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
        #@time @strided @. vki = deltakm
        @time vki = deltakm  # Note: last time we use deltakm, so can overwrite
        @time calc_velocity_component!(vki, kF, 3)
        @time vz = rfftplan \ vki
        #@time @strided @. vz *= faH * (nx*ny*nz) / Volume

        vki = nothing  # free memory
        deltakm = nothing  # free memory
        do_rsd = true
    else
        vx = vy = vz = 0
        do_rsd = false
    end

    println("Apply window...")
    @time @strided @. deltarg = (1 + deltarg) * win - 1

    println("Draw galaxies...")
    Ncells = prod(size_global(deltarg))
    Navg = nbar * prod(Δx)
    Ngalaxies = Navg * Ncells * mean_global(win)
    @time xyzv = draw_galaxies_with_velocities(deltarg, vx, vy, vz, Navg, Ngalaxies, Δx, Val(do_rsd), Val(voxel_window_power), Val(velocity_assignment); rng)

    # FoG: sigma_u = f * sigma_psi
    if sigma_psi != 0
        @. xyzv[4,:] += sigma_psi * randn()
        @. xyzv[5,:] += sigma_psi * randn()
        @. xyzv[6,:] += sigma_psi * randn()
    end

    if faH != 1 && faH != 0
        @time @strided @. xyzv[4:6,:] *= faH
    end

    if gather
        xyzv = concatenate_mpi_arr(xyzv)
    end

    @show Sys.maxrss() / 2^30
    return xyzv
end


@doc raw"""
    simulate_galaxies(nbar, Lbox, pk; nmesh=256, bias=1.0, f=false,
        rfftplanner=default_plan, rng=Random.GLOBAL_RNG, voxel_window_power=1,
        velocity_assignment=1, win=1, sigma_psi=0.0, fixed=false, gather=true)

Simulate galaxies using log-normal statistics.
"""
function simulate_galaxies(nbar, Lbox, pk; nmesh=256, bias=1.0, f=false,
        rfftplanner=default_plan, rng=Random.GLOBAL_RNG, voxel_window_power=1,
        velocity_assignment=1, win=1, sigma_psi=0.0, fixed=false, gather=true)

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

    @time if rfftplanner isa Function
        rfftplan = rfftplanner(nxyz)
    else
        rfftplan = rfftplanner
    end

    @time xyzv = simulate_galaxies(nxyz, Lxyz, nbar, pk, bias, f;
                                   rfftplan, rng, voxel_window_power,
                                   velocity_assignment, sigma_psi, win,
                                   fixed, gather)
    println("Post-processing...")
    @time xyz = @. xyzv[1:3,:] - Lbox / 2
    @time v = xyzv[4:6,:]

    return collect(xyz), collect(v)
end


end


# vim: set sw=4 et sts=4 :
