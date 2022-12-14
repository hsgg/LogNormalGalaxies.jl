using LogNormalGalaxies

module lognormals


using LogNormalGalaxies
using LogNormalGalaxies.Splines
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


function generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlz; rfftplanner=LogNormalGalaxies.plan_with_fftw)
    LLL = [L, L, L]
    nnn_sim = [n_sim, n_sim, n_sim]
    nnn_est = [n_est, n_est, n_est]
    Vol = L^3
    box_center = [0, 0, 0]

    #Lx, Ly, Lz = L, 0.7L, 0.5L
    #nx, ny, nz = n, floor(Int, 0.7n), floor(Int, 0.5n)

    opts = (nbar=nbar, lmax=4, do_mu_leakage=true, subtract_shotnoise=true, voxel_window_power=1)

    x⃗ = fill(0.0, 3, 1)
    km, pkm, nmodes = xgals_to_pkl_planeparallel(x⃗, LLL, nnn_est, box_center; opts...)
    pkm = fill(0.0, length(km), size(pkm,2), nrlz)

    Random.seed!(2041234567893)
    #seeds = [rand(UInt64) for rlz=1:nrlz]

    for rlz=1:nrlz
        println("===== rlz = $rlz/$nrlz")
        #Random.seed!(seeds[1] + rlz)
        #Random.seed!(seeds[rlz])

        # generate catalog
        @time x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n_sim, bias=b, f=true, rfftplanner)
        println("Gather galaxies...")
        x⃗ = LogNormalGalaxies.concatenate_mpi_arr(x⃗)
        Ψ = LogNormalGalaxies.concatenate_mpi_arr(Ψ)

        # add RSD
        los = [0, 0, 1]
        Ngals = size(x⃗,2)
        for i=1:Ngals
            x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
        end

        x⃗ = MeasurePowerSpectra.periodic_boundaries(x⃗, LLL, box_center)

        # cut the possibly incomplete (due to RSD) boundaries
        Lx, Ly, Lz = LLL
        sel = @. -Lx/2 <= x⃗[1,:] <= Lx/2
        @. sel &= -Ly/2 <= x⃗[2,:] <= Ly/2
        @. sel &= -Lz/2 <= x⃗[3,:] <= Lz/2
        x⃗ = x⃗[:,sel]

        # measure multipoles
        kmi, pkmi, nmodesi = xgals_to_pkl_planeparallel(x⃗, LLL, nnn_est, box_center; opts...)
        @assert km == kmi
        @assert nmodes == nmodesi
        @. pkm[:,:,rlz] = pkmi
    end

    pkm_err = std(pkm, dims=3)
    pkm = mean(pkm, dims=3)
    @show size(pkm) size(pkm_err)

    return km, pkm, nmodes, pkm_err
end


function main(fbase, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    b = 1.8
    f = 0.71
    D = 0.4  # deliberately high power spectrum amplitude for testing
    @show Sys.total_memory() / 1024^3
    @show Sys.free_memory() / 1024^3
    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pk(k) = D^2 * _pk(k)

    nbar = 3e-4
    L = 3e3
    n_sim = 256
    n_est = 64
    nrlz = 10

    println("Running with $(rfftplanner)...")
    km, pkm, nmodes, pkm_err = generate_sims(pk, nbar, b, f, L, n_sim, n_est, nrlz; rfftplanner)

    # theory
    β = f / b
    pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pk(km)

    # plot
    #close("all")  # close previous plots to prevent plotcapolypse
    figure()
    #title("Number of MPI Processes: $(MPI.Comm_size(MPI.COMM_WORLD))")
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
end


end # module

lognormals.main("sims_fftw", LogNormalGalaxies.plan_with_fftw)

lognormals.main("sims_pencilffts", LogNormalGalaxies.plan_with_pencilffts)


# vim: set sw=4 et sts=4 :
