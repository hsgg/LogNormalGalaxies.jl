using LogNormalGalaxies

module lognormals


using LogNormalGalaxies
using Splines
using MeasurePowerSpectra
using PyPlot
using Random
using DelimitedFiles
#using MPI
using Statistics


function Arsd_Kaiser(β, ℓ)
    if ℓ == 0
        return 1 + 2/3*β + 1/5*β^2
    elseif ℓ == 2
        return 4/3*β + 4/7*β^2
    elseif ℓ == 4
        return 8/35*β^2
    else
        return 0
    end
end


function generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlz; rfftplanner=LogNormalGalaxies.plan_with_fftw, sim_vox=0, est_vox=0, fxshift_est=0, fxshift_sim=0)
    LLL = [L, L, L]
    nnn_sim = [n_sim, n_sim, n_sim]
    nnn_est = [n_est, n_est, n_est]
    Vol = L^3
    box_center = [0, 0, 0]

    #Lx, Ly, Lz = L, 0.7L, 0.5L
    #nx, ny, nz = n, floor(Int, 0.7n), floor(Int, 0.5n)

    sim_opts = (nmesh=n_sim, bias=b, rfftplanner=rfftplanner, voxel_window_power=sim_vox)
    est_opts = (nbar=nbar, lmax=4, do_mu_leakage=true, subtract_shotnoise=true, voxel_window_power=est_vox)

    x⃗ = fill(0.0, 3, 1)
    km, pkm, nmodes = xgals_to_pkl_planeparallel(x⃗, LLL, nnn_est, box_center; est_opts...)
    pkm = fill(0.0, length(km), size(pkm,2), nrlz)

    #Random.seed!(204213467893)
    #seeds = [rand(UInt64) for rlz=1:nrlz]

    for rlz=1:nrlz
        println("===== rlz = $rlz/$nrlz")
        #Random.seed!(seeds[1] + rlz)
        #Random.seed!(seeds[rlz])
        rsd = (f != 0)

        # generate catalog
        fname = "out/gals_rlz$rlz.bin"
        if false
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

        # add RSD
        if rsd
            println("Apply RSD...")
            los = [0, 0, 1]
            @time x⃗ .+= f * (los' * Ψ) .* los
        end

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
        f = 0 #0.71
        D = 1
        nbar = 1e-3
        L = 3e3
        n_sim = 512
        n_est = 512
        nrlz = 10
        sim_vox = 0
        est_vox = 1
        fxshift_sim = 0
        fxshift_est = 0.5
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
        n_est = 256
        nrlz = 10
        sim_vox = 0
        est_vox = 1
        fxshift_est = 0.25
        fxshift_sim = 0
    end
end


function main(fbase, rfftplanner=LogNormalGalaxies.plan_with_fftw)

    @eval $(agrawal_fig2())
    #@eval $(agrawal_fig6_fig7())

    #data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    #_pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    in_k = readdlm(homedir() * "/MeasurePowerSpectra.jl/inputs/kh_camb_z_eff=0.38.csv")[:]
    in_pk = readdlm(homedir() * "/MeasurePowerSpectra.jl/inputs/matter_power_spectrum_pk_camb_z_eff=0.38.csv")[:]
    _pk = Spline1D(in_k, in_pk, extrapolation=Splines.powerlaw)

    pk(k) = D^2 * _pk(k)

    fname = make_fname("out/"*fbase; nbar, b, D, f, L, n_sim, n_est, sim_vox, est_vox, nrlz, fxshift_est, fxshift_sim)
    #fname = make_fname("out_linbias/"*fbase; nbar, b, D, f, L, n_sim, n_est, sim_vox, est_vox, nrlz, fxshift_est, fxshift_sim)
    #fname = make_fname("out_linbias2/"*fbase; nbar, b, D, f, L, n_sim, n_est, sim_vox, est_vox, nrlz, fxshift_est, fxshift_sim)

    println("Running with $(rfftplanner)...")
    km, pkm, nmodes, pkm_err = generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlz; rfftplanner, sim_vox, est_vox, fxshift_est, fxshift_sim)
    writedlm(fname, [km pkm nmodes pkm_err])
    km, pkm, nmodes, pkm_err = readdlm_cols(fname, [1, 2:6, 7, 8:12])

    # theory
    β = f / b
    pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pk(km)


    # plot
    #close("all")  # close previous plots to prevent plotcapolypse
    figure()
    make_title(; L, D, f, n_sim, n_est, sim_vox, est_vox)
    hlines(1/nbar, extrema(km)..., color="0.75", label="Shot noise")
    plot(km, b^2 .* pk.(km), "k", label="input \$k\\,P(k)\$")
    for m=1:size(pkm,2)
        errorbar(km, pkm[:,m], pkm_err[:,m], c="C$(m-1)", alpha=0.7)
        errorbar(km, pkm[:,m], pkm_err[:,m] ./ sqrt(nrlz), c="C$(m-1)", elinewidth=4, alpha=0.7)
        plot(km, pkm[:,m], "C$(m-1)-", label="\$P_{$(m-1)}(k)\$", alpha=0.7)
        plot(km, pkm_kaiser[:,m], "C$(m-1)--", alpha=0.7)
    end
    xlabel(L"k")
    ylabel(L"P_\ell(k)")
    xscale("log")
    xlim(right=0.6)
    legend(fontsize="small")
    savefig((@__DIR__)*"/$(fbase).pdf")


    figure()
    make_title(; L, D, f, n_sim, n_est, sim_vox, est_vox, xshift=fxshift_est)
    hlines(1, extrema(km)..., color="0.8")
    hlines([0.99,1.01], extrema(km)..., color="0.7", linestyle="--")
    hlines([0.95,1.05], extrema(km)..., color="0.6", linestyle=":")
    for l=[0,2]
        m = l+1
        ymid = pkm[:,m] ./ pkm_kaiser[:,m]
        yerr = pkm_err[:,m] ./ pkm_kaiser[:,m]
        #errorbar(km, ymid, yerr, c="C$(m-1)", alpha=0.7)
        #errorbar(km, ymid, yerr ./ sqrt(nrlz), c="C$(m-1)", elinewidth=4, alpha=0.7)
        errorbar(km, ymid, yerr ./ sqrt(nrlz), c="C$(m-1)", elinewidth=1, alpha=0.7)
        plot(km, ymid, "C$(m-1)-", label="\$P_{$(m-1)}(k)\$", alpha=0.7)
    end
    xlabel(L"k")
    ylabel(L"\hat P^{\rm pp}_\ell(k) / P^{\rm Kaiser}_\ell(k)")
    #xscale("log")
    xlim(right=0.15)
    ylim(0.9, 1.1)
    legend(fontsize="small")
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
