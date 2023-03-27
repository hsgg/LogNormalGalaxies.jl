#!/usr/bin/env julia

using Revise

# load current LogNormalGalaxies version:
using Pkg
Pkg.activate((@__DIR__)*"/../..")
using LogNormalGalaxies

# load other dependencies:
Pkg.activate(@__DIR__)
using DelimitedFiles
using Random
using Statistics
using Splines
using MeasurePowerSpectra
using YAML

include("lib.jl")


function apply_RSD!(x⃗, Ψ, f, los)
    Ngals = size(x⃗,2)
    for i=1:Ngals
        x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
    end
    return x⃗
end


function generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlzs; rfftplanner=LogNormalGalaxies.plan_with_fftw, sim_vox=0, est_vox=0, sim_velo=1, est_grid_assignment=1, fxshift_est=0, fxshift_sim=0, sigma_psi=0.0)
    LLL = [L, L, L]
    nnn_sim = [n_sim, n_sim, n_sim]
    nnn_est = [n_est, n_est, n_est]
    box_center = [0, 0, 0]

    #Lx, Ly, Lz = L, 0.7L, 0.5L
    #nx, ny, nz = n, floor(Int, 0.7n), floor(Int, 0.5n)

    sim_opts = (nmesh=n_sim, bias=b, rfftplanner=rfftplanner, voxel_window_power=sim_vox, velocity_assignment=sim_velo, sigma_psi=sigma_psi)
    est_opts = (nbar=nbar, lmax=4, do_mu_leakage=true, subtract_shotnoise=true, voxel_window_power=est_vox, grid_assignment=est_grid_assignment)

    # allocate outputs
    x⃗ = fill(0.0, 3, 1)
    km, pkm, nmodes = xgals_to_pkl_planeparallel(x⃗, LLL, nnn_est, box_center; est_opts...)
    pkm = fill(0.0, length(km), size(pkm,2), nrlzs)

    seeds = [rand(UInt64,4) for rlz=1:nrlzs]

    for rlz=1:nrlzs
        println("===== rlz = $rlz/$nrlzs")
        Random.seed!(seeds[rlz])
        rsd = (f != 0)

        # generate catalog
        println("Simulate galaxies...")
        @time x⃗, Ψ = simulate_galaxies(nbar, L, pk; sim_opts..., f=rsd)

        if rsd
            println("Apply RSD...")
            los = [0, 0, 1]
            apply_RSD!(x⃗, Ψ, f, los)
        end

        println("Shift grid by fractions $fxshift_est and $fxshift_sim...")
        @. x⃗ += fxshift_est * LLL / nnn_est + fxshift_sim * LLL / nnn_sim

        println("Apply periodic boundary...")
        @time x⃗ = MeasurePowerSpectra.periodic_boundaries(x⃗, LLL, box_center)

        # measure multipoles
        println("Measure power spectrum multipoles...")
        @time kmi, pkmi, nmodesi = xgals_to_pkl_planeparallel(x⃗, LLL, nnn_est, box_center; est_opts...)
        @assert km == kmi
        @assert nmodes == nmodesi
        @. pkm[:,:,rlz] = pkmi
    end

    pkm_err = std(pkm, dims=3)[:,:,1]
    pkm = mean(pkm, dims=3)[:,:,1]
    @show size(pkm) size(pkm_err)

    return km, pkm, nmodes, pkm_err
end


function main(args)
    cfg_fbase = splitext(args[1])[1]
    cfg = YAML.load_file(cfg_fbase * ".yml")
    fname = (@__DIR__)*"/../sim_results/"*basename(cfg_fbase)*"_pkest.tsv"
    mkpath(dirname(fname))

    b = cfg["bias"]
    D = cfg["D"]
    nbar = cfg["nbar"]
    L = cfg["L"]
    nrlzs = cfg["nrlzs"]

    n_sim = cfg["n_sim"]
    n_est = cfg["n_est"]

    f = cfg["f"]
    sigma_psi = cfg["sigma_psi"]
    sim_vox = cfg["sim_vox"]
    sim_velo = cfg["sim_velo"]
    est_vox = cfg["est_vox"]
    est_grid_assignment = cfg["est_grid_assignment"]
    fxshift_sim = cfg["fxshift_sim"]
    fxshift_est = cfg["fxshift_est"]

    Random.seed!(eval(Meta.parse(cfg["randomseed"])))

    data = readdlm((@__DIR__)*"/"*cfg["pkfname"], comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pk(k) = D^2 * _pk(k)

    println("Running...")
    km, pkm, nmodes, pkm_err = generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlzs; sim_vox, est_vox, sim_velo, est_grid_assignment, fxshift_est, fxshift_sim, sigma_psi)

    writedlm(fname, [km pkm nmodes pkm_err])

    println("Saved results to '$fname'.")
end


main(ARGS)



# vim: set sw=4 et sts=4 :
