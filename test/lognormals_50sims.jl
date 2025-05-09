# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


using LogNormalGalaxies
using Test

module lognormals


using LogNormalGalaxies
using MySplines
using MeasurePowerSpectra
using PythonPlot
using Random
using DelimitedFiles
#using MPI
using Statistics
using PlaneParallelRedshiftSpaceDistortions


include("lib.jl")


function generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlzs; rfftplanner=LogNormalGalaxies.plan_with_fftw, sim_vox=0, est_vox=0, sim_velo=1, grid_assignment=1, fxshift_est=0, fxshift_sim=0, generate=true, sigma_psi=0.0)
    LLL = [L, L, L]
    nnn_sim = [n_sim, n_sim, n_sim]
    nnn_est = [n_est, n_est, n_est]
    Vol = L^3
    box_center = [0, 0, 0]

    #Lx, Ly, Lz = L, 0.7L, 0.5L
    #nx, ny, nz = n, floor(Int, 0.7n), floor(Int, 0.5n)

    sim_opts = (nmesh=n_sim, bias=b, rfftplanner=rfftplanner, voxel_window_power=sim_vox, velocity_assignment=sim_velo, sigma_psi=sigma_psi)
    est_opts = (nbar=nbar, lmax=4, do_mu_leakage=true, subtract_shotnoise=true, voxel_window_power=est_vox, grid_assignment=grid_assignment)

    x⃗ = fill(0.0, 3, 1)
    km, pkm, nmodes = xgals_to_pkl_planeparallel(x⃗, LLL, nnn_est, box_center; est_opts...)
    pkm = fill(0.0, length(km), size(pkm,2), nrlzs)

    #Random.seed!(204213467893)
    #seeds = [rand(UInt64) for rlz=1:nrlzs]

    for rlz=1:nrlzs
        println("===== rlz = $rlz/$nrlzs")
        #Random.seed!(seeds[1] + rlz)
        #Random.seed!(seeds[rlz])
        rsd = (f != 0)

        # generate catalog
        fname = "out/gals_rlz$rlz.bin"
        if generate || !isfile(fname)
            @time x⃗, Ψ = simulate_galaxies(nbar, L, pk; sim_opts..., f=rsd)
            println("Gather galaxies...")
            @time x⃗ = LogNormalGalaxies.concatenate_mpi_arr(x⃗)
            @time Ψ = LogNormalGalaxies.concatenate_mpi_arr(Ψ)

            mkpath(dirname(fname))
            write_galaxies(fname, LLL, [x⃗; Ψ])
        end
        LLL, xyzv = read_galaxies(fname; ncols=6)
        x⃗ = collect(xyzv[1:3,:])
        Ψ = collect(xyzv[4:6,:])

        if rsd
            println("Apply RSD...")
            los = [0, 0, 1]
            apply_rsd!(x⃗, Ψ, f, los)
        end

        @. x⃗ += fxshift_est * LLL / nnn_est + fxshift_sim * LLL / nnn_sim

        println("Apply periodic boundary...")
        @time apply_periodic_boundaries!(x⃗, LLL, box_center)

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


function readdlm_cols(fname, cols)
    data = readdlm(fname)
    key = []
    for c in cols
        k = data[:,c]
        push!(key, k)
    end
    return key
end


function abbreviate(input)
    output = []
    for v in input
        if string(v) == "n_sim"
            v = "\$n_s\$"
        elseif string(v) == "n_est"
            v = "\$n_e\$"
        elseif string(v) == "sim_vox"
            v = "\$v_s\$"
        elseif string(v) == "est_vox"
            v = "\$v_e\$"
        elseif string(v) == "sim_velo"
            v = "\$vel_s\$"
        elseif string(v) == "grid_assignment"
            v = "\$g_e\$"
        end
        push!(output, v)
    end
    return output
end


function make_title(; kwargs...)
    t = join(abbreviate(keys(kwargs)), ", ")
    t *= " = "
    t *= join(values(kwargs), ", ")
    title(t)
end

function make_fname(prefix=""; kwargs...)
    fname = join(["$k=$v" for (k,v) in kwargs], "_")
    if length(prefix) > 0
        fname = prefix * "_" * fname
    end
    fname *= ".tsv"
    mkpath(dirname(fname))
    @show fname
    return fname
end


function agrawal_fig2()
    return quote
        b = 1.455
        f = 0.71
        #f = 0
        D = 1
        nbar = 3e-4
        L = 1e3
        n_sim = 320
        n_est = 320
        nrlzs = 1000
        sim_vox = 2
        est_vox = 0
        sim_velo = 1
        grid_assignment = 1
        fxshift_sim = 0
        fxshift_est = 0
        sigma_psi = 0
    end
end
function agrawal_fig6_fig7()
    return quote
        b = 1.455
        f = 0.71
        D = 1
        nbar = 1e-3
        L = 3e3
        n_sim = 512
        n_est = 512
        nrlzs = 100
        sim_vox = 2
        est_vox = 0
        sim_velo = 1
        grid_assignment = 1
        fxshift_sim = 0
        fxshift_est = 0
        sigma_psi = 0
    end
end


function main(fbase, rfftplanner=LogNormalGalaxies.plan_with_fftw)

    @eval $(agrawal_fig2())
    #@eval $(agrawal_fig6_fig7())
    generate = false

    #data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    #_pk = Spline1D(data[:,1], data[:,2], extrapolation=MySplines.powerlaw)
    in_k = readdlm(homedir() * "/MeasurePowerSpectra.jl/inputs/kh_camb_z_eff=0.38.csv")[:]
    in_pk = readdlm(homedir() * "/MeasurePowerSpectra.jl/inputs/matter_power_spectrum_pk_camb_z_eff=0.38.csv")[:]
    _pk = Spline1D(in_k, in_pk, extrapolation=MySplines.powerlaw)

    pk(k) = D^2 * _pk(k)

    fname = make_fname("out/"*fbase; nbar, b, D, f, L, n_sim, n_est, sim_vox, est_vox, sim_velo, grid_assignment, nrlzs, fxshift_est, fxshift_sim, sigma_psi)

    println("Running with $(rfftplanner)...")
    if generate || !isfile(fname)
        km, pkm, nmodes, pkm_err = generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlzs; rfftplanner, sim_vox, est_vox, sim_velo, grid_assignment, fxshift_est, fxshift_sim, generate, sigma_psi)
        writedlm(fname, [km pkm nmodes pkm_err])
    end
    km, pkm, nmodes, pkm_err = readdlm_cols(fname, [1, 2:6, 7, 8:12])
    pkm[1,2:end] .= 0
    pkm_err[1,2:end] .= 0
    @assert all(isfinite.(pkm_err))
    @assert all(pkm_err .>= 0)


    # theory
    pk_g = @. b^2 * pk(km)
    β = f / b
    #pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pk(km)
    pkl_kaiser = @. b^2 * Arsd_l_exp(km*f*sigma_psi, β, (0:4)') * pk(km)


    n = 1
    Δx_sim = L / n_sim
    Δx_est = L / n_est
    Wmesh_sim = @. sinc(km * Δx_sim / (2 * π))
    Wmesh_est = @. sinc(km * Δx_est / (2 * π))

    # plot
    figure()
    make_title(; L, D, f, n_sim, n_est, sim_vox, est_vox, sim_velo)#, grid_assignment)
    plot_pkl(km, pkm, pkm_err, pkl_kaiser, nbar; n, nrlzs, pk_g)
    savefig((@__DIR__)*"/$(fbase).pdf")

    figure()
    make_title(; L, D, f, n_sim, n_est, sim_vox, est_vox, sim_velo, #=grid_assignment,=# xshift=fxshift_est)
    plot_pkl_diff(km, pkm, pkm_err, pkl_kaiser, nbar; n, nrlzs)
    plot(km, Wmesh_sim.^7)
    plot(km, Wmesh_est.^6)
    savefig((@__DIR__)*"/$(fbase)_rdiff.pdf")
end


end # module

lognormals.main("sims_fftw", LogNormalGalaxies.plan_with_fftw)

if Sys.ARCH == :aarch64
    @test_skip "Skipping PencilFFTs on ARM64"
else
    lognormals.main("sims_pencilffts", LogNormalGalaxies.plan_with_pencilffts)
end


# vim: set sw=4 et sts=4 :
