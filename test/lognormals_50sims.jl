using LogNormalGalaxies

module lognormals


using LogNormalGalaxies
using LogNormalGalaxies.Splines
using MeasurePowerSpectra
using PyPlot
using Random
using DelimitedFiles
using MPI


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


function generate_sims(pk, nbar, b, f, L, n; rfftplanner=LogNormalGalaxies.plan_with_fftw)
    ΔL = 50.0  # buffer for RSD
    nrlz = 10

    Lx, Ly, Lz = L, L, L
    nx, ny, nz = n, n, n

    #Lx, Ly, Lz = L, 0.7L, 0.5L
    #nx, ny, nz = n, floor(Int, 0.7n), floor(Int, 0.5n)

    x⃗ = fill(0.0, 3, 1)
    @show size(x⃗)
    km, pkm, Mlm, Ngalaxies = xgals_to_pkm([Lx,Ly,Lz], [nx,ny,nz], x⃗; lmax=4)
    pkm .= 0
    Mlm .= 0
    Ngalaxies = 0

    #Random.seed!(1234567890)
    #Random.seed!(9876543210)
    seed = rand(UInt128)
    @show seed
    Random.seed!(seed)

    #seeds = [rand(UInt64) for rlz=1:nrlz]

    for rlz=1:nrlz
        println("===== rlz = $rlz")
        #Random.seed!(seeds[1] + rlz)
        #Random.seed!(seeds[rlz])

        # generate catalog
        @time x⃗, Ψ = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1, rfftplanner=rfftplanner)

        println("Gather galaxies...")
        x⃗ = LogNormalGalaxies.concatenate_mpi_arr(x⃗)
        Ψ = LogNormalGalaxies.concatenate_mpi_arr(Ψ)

        # add RSD
        los = [0, 0, 1]
        Ngals = size(x⃗,2)
        for i=1:Ngals
            x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
        end

        # cut the possibly incomplete (due to RSD) boundaries
        sel = @. -Lx/2 <= x⃗[1,:] <= Lx/2
        @. sel &= -Ly/2 <= x⃗[2,:] <= Ly/2
        @. sel &= -Lz/2 <= x⃗[3,:] <= Lz/2
        x⃗ = x⃗[:,sel]
        @show size(x⃗)

        # measure multipoles
        kmi, pkmi, Mlmi, Ngalaxiesi = xgals_to_pkm([Lx,Ly,Lz], [nx,ny,nz], x⃗; lmax=4)
        @assert km == kmi
        @. pkm = ((rlz - 1) * pkm + pkmi) / rlz
        @. Mlm = ((rlz - 1) * Mlm + Mlmi) / rlz
        Ngalaxies = ((rlz - 1) * Ngalaxies + Ngalaxiesi) / rlz
    end

    return km, pkm, Mlm, Ngalaxies
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
    n = 256
    #Random.seed!(8143083339)  # don't initialize all MPI processes with the same seed!

    println("Running with $(rfftplanner)...")
    km, pkm, Mlm, Ngalaxies = generate_sims(pk, nbar, b, f, L, n; rfftplanner=rfftplanner)

    # theory
    β = f / b
    pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pk(km)

    # plot
    close("all")  # close previous plots to prevent plotcapolypse
    figure()
    #title("Number of MPI Processes: $(MPI.Comm_size(MPI.COMM_WORLD))")
    hlines(1/nbar, extrema(km)..., color="0.75", label="Shot noise")
    plot(km, b^2 .* km.*pk.(km), "k", label="input \$k\\,P(k)\$")
    for m=1:size(pkm,2)
        plot(km, pkm[:,m], "C$(m-1)-", label="\$P_{$(m-1)}(k)\$")
        plot(km, pkm_kaiser[:,m], "C$(m-1)--")
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
