# The purpose of this package is to provide a quick way to generate a mock
# galaxy catalog.
#
# This version can use either FFTW or PencilFFTs.jl to distribute the density
# field over several nodes. PencilFFTs.jl, in turn, uses MPI for this task.


module LogNormalGalaxies

export simulate_galaxies,
       simulate_galaxies_mpi,
       read_galaxies,
       write_galaxies


#using PkSpectra
#using Cosmologies

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

#using QuadOsc
#using QuadGK


j0(x) = sinc(x/π)


#function xicalc00_quadosc(fn, r)
#    I,E = quadosc(k -> k^2 * fn(k) * j0(k*r), 0, Inf, n->π*n/r)
#    #I,E = quadgk(k -> k^2 * fn(k) * j0(k*r), 0, Inf)
#    I,E = (I,E) ./ (2 * π^2)
#    @show r,I
#    return I
#end


# PencilFFTs.jl needs 'allocate_input()', but FFTW doesn't provide it:
function PencilFFTs.allocate_input(plan::FFTW.FFTWPlan)
    T = eltype(plan)
    return Array{T}(undef, size(plan))
end


function plan_with_fftw(nxyz)
    return plan_rfft(Array{Float64}(undef, nxyz))
end

function plan_with_pencilffts(nxyz)
    rank, comm = start_mpi()

    proc_dims = tuple(MPI.Dims_create!(MPI.Comm_size(comm), zeros(Int, 2))...)
    transform = Transforms.RFFT()
    @show proc_dims

    @time rfftplan = PencilFFTPlan(nxyz, transform, proc_dims, comm)
    return rfftplan
end

#default_plan = plan_with_fftw
default_plan = plan_with_pencilffts


Base.deepcopy(pa::PencilArray) = PencilArray(pencil(pa), deepcopy(parent(pa)))


PencilFFTs.pencil(arr::AbstractArray) = size(arr)


allocate_array(shape, type::DataType) = Array{type}(undef, shape)

allocate_array(pen::Pencil, type::DataType) = begin
    @show pen typeof(pen)
    return PencilArray{type}(undef, Pencil(pen))
end


PencilFFTs.global_view(arr::AbstractArray) = arr

range_global(arr) = begin
    r = ()
    for s in size_global(arr)
        r = (r..., 1:s)
    end
    return r
end

PencilFFTs.range_local(arr::AbstractArray) = range_global(arr)


PencilFFTs.size_global(arr::AbstractArray) = size(arr)


function get_rank()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
end


#################### calculate P_G(k) ###########################
function pk_to_pkG(pkfn)
    r, xi = xicalc(pkfn, 0, 0; kmin=1e-10, kmax=1e10, r0=1e-5, N=2^15, q=2.0)
    #r2, xi2 = xicalc(pkfn, 0, 0; kmin=1e-25, kmax=1e25, r0=1e-25, N=4096, q=2.0)
    #r3 = range(0.001, 1e3, length=10000)
    #xi3 = xicalc00_quadosc.(pkfn, r3)
    #close("all")

    #figure()
    #plot(r,xi, "b", label=L"\xi(r)")
    #plot(r,-xi, "b--")
    #plot(r3,xi3, "g", label=L"\xi(r)")
    #plot(r3,-xi3, "g--")
    #xscale("log")
    #yscale("log")
    #legend()

    sel = @. 1e-5 <= r <= 1e7
    r = r[sel]
    xi = xi[sel]

    #sel2 = @. 1e-5 <= r2 <= 1e7
    #r2 = r2[sel2]
    #xi2 = xi2[sel2]

    xiG = @. log1p(xi)
    xiGfn = Spline1D(r, xiG, extrapolation=Splines.powerlaw)
    #xiG2 = @. log1p(xi2)
    #xiG3 = @. log1p(xi3)
    #xiG3fn = Spline1D(r3, xiG3, extrapolation=Splines.powerlaw)

    #figure()
    ##plot(r,xi, "k", label=L"\xi(r)")
    ##plot(r,-xi, "k--")
    ##plot(r,xiG, "b", label=L"\xi_G(r)")
    ##plot(r,-xiG, "b--")
    #plot(r,r.^3 .* xi, "k", label=L"r^3 \xi(r)")
    #plot(r,-r.^3 .* xi, "k--")
    ##plot(r2,r2.^3 .* xi2, "0.75")
    ##plot(r2,-r2.^3 .* xi2, c="0.75", ls="--")
    #plot(r,r.^3 .* xiG, "b", label=L"r^3 \xi_G(r)")
    #plot(r,-r.^3 .* xiG, "b--")
    #plot(r3,r3.^3 .* xiG3, "g")
    #plot(r3,-r3.^3 .* xiG3, "g--")
    #xscale("log")
    #yscale("log")
    #xlabel(L"r")
    #legend()

    #dlnr = log(r[3]/r[2])
    #pk_0 = 4π*sum(xi.*r.^3)*dlnr
    #pk_0² = 4π*sum(xi.^2 .* r.^3)*dlnr
    #pk_0³ = 4π*sum(xi.^3 .* r.^3)*dlnr
    #pk_0⁴ = 4π*sum(xi.^4 .* r.^3)*dlnr
    #@show pk_0 pk_0²/2 pk_0³/3 pk_0⁴/4
    #pkG_0 = 4π*sum(xiG.*r.^3)*dlnr
    #@show pkG_0

    #dlnr = log(r2[3]/r2[2])
    #pk_0 = 4π*sum(xi2.*r2.^3)*dlnr
    #pk_0² = 4π*sum(xi2.^2 .* r2.^3)*dlnr
    #pk_0³ = 4π*sum(xi2.^3 .* r2.^3)*dlnr
    #pk_0⁴ = 4π*sum(xi2.^4 .* r2.^3)*dlnr
    #@show pk_0 pk_0²/2 pk_0³/3 pk_0⁴/4
    #pkG_0 = 4π*sum(xiG2.*r2.^3)*dlnr
    #@show pkG_0

    #return r, xi

    #k, pk2 = xicalc((r,xi), 0, 0; r0=1/maximum(r), q=0.7)
    #k, pk3 = xicalc((r,xi), 0, 0; r0=1/maximum(r), q=0.8)
    #pk2 .*= (2π)^3
    #pk3 .*= (2π)^3
    #figure()
    #plot(k, pkfn.(k), "k", label=L"input $P(k)$")
    #plot(k, pk2, "b", label=L"round-trip $P(k)$")
    #plot(k, -pk2, "b--")
    #plot(k, pk3, "g", label=L"round-trip $P(k)$")
    #plot(k, -pk3, "g--")
    #xscale("log")
    #yscale("log")
    ##ylim(1e-10, 1e5)
    #legend()

    #kln = readdlm("$root/data/fog_r1000_pkG.dat")[:,1]
    #pkGln = readdlm("$root/data/fog_r1000_pkG.dat")[:,2]

    #k, pkG = xicalc((r,xiG), 0, 0; r0=1/maximum(r), q=1.5)
    k, pkG = xicalc(xiGfn, 0, 0; kmin=1e-10, kmax=1e10, r0=1e-10, N=2^18, q=1.5)
    #k2, pkG2 = xicalc(xiGfn, 0, 0; kmin=1e-10, kmax=1e10, r0=1e-5, N=2^18, q=1.5)
    #k3 = 10.0 .^ range(-5, 1, length=100)
    #pkG3 = xicalc00_quadosc.(xiG3fn, k3)
    pkG .*= (2π)^3
    #pkG2 .*= (2π)^3
    #pkG3 .*= (2π)^3
    #figure()
    ##plot(kln, pkGln, "k", lw=3, label=L"DJ's $P_G(k)$")
    ##plot(kln, -pkGln, "k--", lw=3)
    #plot(k, pkfn.(k), "k", L"P(k)")
    #plot(k,pkG, "b", label=L"$P_G(k)$")
    #plot(k,-pkG, "b--")
    #plot(k2,pkG2, "g", label=L"$P_G(k)$")
    #plot(k2,-pkG2, "g--")
    #plot(k3,pkG3, "r", label=L"$P_G(k)$")
    #plot(k3,-pkG3, "r--")
    #xscale("log")
    #yscale("log")
    ##ylim(1e-14, 1e5)
    #legend()

    # the extremes lead to overflow
    sel = @. (pkG > 0)
    k = k[sel][3:end-2]
    pkG = pkG[sel][3:end-2]

    sel = @. 1e-3 <= k <= 1e2
    k = k[sel]
    pkG = pkG[sel]

    #pkGfn = PkSpectrum(k, pkG)
    pkGfn = Spline1D(k, pkG, extrapolation=Splines.powerlaw)
    #@show pkGfn.kmin
    #@show pkGfn.kmax
    #@show pkGfn.nslo
    #@show pkGfn.nshi
    #@show pkGfn.kmin_norm
    #@show pkGfn.kmax_norm
    #@show pkGfn.kmax_a
    return k, pkGfn
end


#################### draw deltak ###########################

function draw_phases(rfftplan; reproducible=true)
    deltar = allocate_input(rfftplan)
    @time if reproducible && get_rank() == 0
        deltar_global = global_view(deltar)
        randn!(deltar_global)
    else
        randn!(deltar)
    end
    @time deltak_phases = rfftplan * deltar
    @time @. deltak_phases /= abs(deltak_phases)
    return deltak_phases
end


function calc_kmode(nx, ny, nz, kF, pencil_δk)
    nx2 = div(nx,2) + 1
    ny2 = div(ny,2) + 1
    nz2 = div(nz,2) + 1
    #kmode = PencilArray{Float64}(undef, Pencil(pencil_δk, Float64))  # allocate uninitialized array
    kmode = allocate_array(pencil_δk, Float64)
    kmode_global = global_view(kmode)
    localrange = range_local(kmode)
    for k=localrange[3], j=localrange[2], i=localrange[1]
        ikx = i - 1
        iky = j <= ny2 ? j-1 : j-1-ny
        ikz = k <= nz2 ? k-1 : k-1-nz
        kmode_global[i,j,k] = kF * √(ikx^2 + iky^2 + ikz^2)
    end
    return kmode
end


function draw_normal_deltak(deltak_phases, kmode, pk)
    return deltak
end


####################### calculate volocity field #################
function calculate_velocities_faH(deltak, kF)
    nx2, ny, nz = size_global(deltak)
    ny2 = div(ny,2) + 1
    nz2 = div(nz,2) + 1
    vkx = deepcopy(deltak)
    vky = deepcopy(deltak)
    vkz = deepcopy(deltak)
    deltak_global = global_view(deltak)
    vkx_global = global_view(vkx)
    vky_global = global_view(vky)
    vkz_global = global_view(vkz)
    localrange = range_local(deltak)
    for k=localrange[3], j=localrange[2], i=localrange[1]
        ikx = i - 1
        iky = j <= ny2 ? j-1 : j-1-ny
        ikz = k <= nz2 ? k-1 : k-1-nz
        ikmode = √(ikx^2 + iky^2 + ikz^2)
        if ikmode == 0
            vkx_global[i,j,k] = 0
            vky_global[i,j,k] = 0
            vkz_global[i,j,k] = 0
        else
            vk = im/ikmode^2/kF*deltak_global[i,j,k]
            vkx_global[i,j,k] = ikx*vk
            vky_global[i,j,k] = iky*vk
            vkz_global[i,j,k] = ikz*vk
        end
    end
    return vkx, vky, vkz
end


##################### draw galaxies ###########################
function draw_galaxies_with_velocities(deltar, vx, vy, vz, Ngalaxies, Δx=1.0; reproducible=true)
    nx, ny, nz = size_global(deltar)
    Navg = Ngalaxies / (nx * ny * nz)
    xyzv = Float32[]
    deltar_global = global_view(deltar)
    vx_global = global_view(vx)
    vy_global = global_view(vy)
    vz_global = global_view(vz)
    localrange = if reproducible
        if get_rank() == 0
            range_global(deltar)
        else
            (1:0, 1:0, 1:0)  # empty ranges, everything is done on rank=0
        end
    else
        range_local(deltar)
    end
    @time for k=localrange[3], j=localrange[2], i=localrange[1]
        Nthiscell = pois_rand((1 + deltar_global[i,j,k]) * Navg)
        for n=1:Nthiscell
            x = i - 1 + rand()
            y = j - 1 + rand()
            z = k - 1 + rand()
            push!(xyzv, x*Δx)
            push!(xyzv, y*Δx)
            push!(xyzv, z*Δx)
            push!(xyzv, vx_global[i,j,k])
            push!(xyzv, vy_global[i,j,k])
            push!(xyzv, vz_global[i,j,k])
        end
    end
    @time xyzv_out = reshape(xyzv, 6, :)
    return xyzv_out
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

################## simulate_galaxies() ##################
# Here are multiple functions called 'simulate_galaxies()'. They only differ in
# their interface.

# simulate galaxies
function simulate_galaxies(nxyz, Lxyz, Ngalaxies, pk, kF, Δx, b, faH, rfftplan=default_plan(nxyz))
    nx, ny, nz = nxyz
    Lx, Ly, Lz = Lxyz
    Volume = Lx * Ly * Lz

    println("Convert to log-normal...")
    #kln = readdlm("$root/data/fog_r1000_pkG.dat")[:,1]
    #pkGln = readdlm("$root/data/fog_r1000_pkG.dat")[:,2]
    kGm, pkGm = pk_to_pkG(pk)
    kGg, pkGg = pk_to_pkG(k -> b^2 * pk(k))

    println("Draw random phases...")
    deltak_phases = draw_phases(rfftplan)
    @show get_rank(),deltak_phases[1,1,1],mean(deltak_phases)
    return
    println("Calculate kmode...")
    @time kmode = calc_kmode(nx, ny, nz, kF, pencil(deltak_phases))
    @show get_rank(),kmode[1,1,1],mean(kmode)
    Volume = (2π / kF)^3
    println("Calculate deltak{m,g}...")
    @time deltakm = deepcopy(deltak_phases)
    @time deltakg = deltak_phases
    @time @. deltakm *= √(pkGm(kmode) * Volume)
    @time @. deltakg *= √(pkGg(kmode) * Volume)
    deltak_phases = nothing
    kmode = nothing
    @show get_rank(),deltakm[1,1,1],mean(deltakm)
    @show get_rank(),deltakg[1,1,1],mean(deltakg)
    println("Calculate deltar{m,g}...")
    @time deltarm = rfftplan \ deltakm
    @time deltarg = rfftplan \ deltakg
    @show get_rank(),"interim",deltarm[1,1,1],mean(deltakm)
    @show get_rank(),"interim",deltarg[1,1,1],mean(deltakg)
    @time @. deltarm *= (nx*ny*nz) / Volume
    @time @. deltarg *= (nx*ny*nz) / Volume
    deltakg = nothing
    @show get_rank(),deltarm[1,1,1],mean(deltakm)
    @show get_rank(),deltarg[1,1,1],mean(deltakg)
    @show mean(deltarm),std(deltarm)
    @show extrema(deltarm)
    @show mean(deltarg),std(deltarg)
    @show extrema(deltarg)

    # G -> δ
    println("Transform G → δ...")
    σGm² = var(deltarm)
    σGg² = var(deltarg)
    @time @. deltarm = exp(deltarm - 0.5σGm²) - 1
    @time @. deltarg = exp(deltarg - 0.5σGg²) - 1
    @show σGm² σGg²
    @show mean(deltarm),std(deltarm)
    @show extrema(deltarm)
    @show mean(deltarg),std(deltarg)
    @show extrema(deltarg)

    # calculate velocity field
    println("Calculate deltakm...")
    @time deltakm = rfftplan * deltarm
    @time @. deltakm *= Volume / (nx*ny*nz)
    deltarm = nothing
    println("Calculate v⃗(k⃗)...")
    @time vkx, vky, vkz = calculate_velocities_faH(deltakm, kF)
    deltakm = nothing  # free memory
    println("Calculate v⃗(r⃗)...")
    @time vx = rfftplan \ vkx
    @time @. vx *= faH * (nx*ny*nz) / Volume
    vkx = nothing  # free memory
    @time vy = rfftplan \ vky
    @time @. vy *= faH * (nx*ny*nz) / Volume
    vky = nothing  # free memory
    @time vz = rfftplan \ vkz
    @time @. vz *= faH * (nx*ny*nz) / Volume
    vkz = nothing  # free memory

    println("Draw galaxies...")
    #@time xyz = draw_galaxies(deltarg, Ngalaxies, Δx)
    @time xyzv = draw_galaxies_with_velocities(deltarg, vx, vy, vz, Ngalaxies, Δx)
    return xyzv
end


function start_mpi()
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    println("This is $(rank) of $(MPI.Comm_size(comm))")
    #rank == 0 || redirect_stdout(open("/dev/null", "w"))
    return rank, comm
end


function simulate_galaxies_mpi(dname::AbstractString, nxyz, Lxyz, Ngalaxies,
                           pk, kF, Δx, b, faH)
    rank, comm = start_mpi()

    rfftplan = plan_with_fftw(nxyz)
    #rfftplan = plan_with_pencilffts(nxyz)

    xyzv = simulate_galaxies(nxyz, Lxyz, Ngalaxies, pk, kF, Δx, b, faH, rfftplan)
    mkpath(dname)
    fname = @sprintf("%s/%04d.bin", dname, rank)
    @time write_galaxies(fname, Lxyz, xyzv)

    println("Saving all galaxies in one big file...")
    @time Ngalaxies = MPI.Reduce(size(xyzv,2), +, 0, comm)
    if rank == 0
        @time open("$dname.bin", "w") do fout
            write(fout, Array{Float64}(collect(Lxyz)))
            write(fout, Int64(Ngalaxies))
            @time write(fout, Array{Float32}(xyzv))  # rank = 0
            for r=1:MPI.Comm_size(comm)-1  # rank=1:end
                fname = @sprintf("%s/%04d.bin", dname, r)
                @time LLL, xyzv = read_galaxies(fname)
                @time write(fout, Array{Float32}(xyzv))
            end
        end
    end

    MPI.Finalize()
    return rank
end


function simulate_galaxies(nbar, Lbox, pk; nmesh=256, bias=1.0, f=0.0,
        lagrangian=false)
    aH = 1
    b = bias
    L = Lbox
    Ngalaxies = ceil(Int, L^3 * nbar)

    n = nmesh
    kF = 2π/L
    Δx = L/n

    nxyz = n, n, n
    Lxyz = L, L, L
    @time xyzv = simulate_galaxies(nxyz, Lxyz, Ngalaxies, pk, kF, Δx, b, f)
                                   #lagrangian=lagrangian)
    xyz = @. xyzv[1:3,:] - Float32(L / 2)
    v = xyzv[4:6,:]
    return xyz, v
end



end


# vim: set sw=4 et sts=4 :
