using Revise
using LogNormalGalaxies
using LogNormalGalaxies.Splines
using MeasurePowerSpectra
using DelimitedFiles
using Statistics
using PythonPlot
using Random
using Test
using LinearAlgebra

@testset "Orthogonal simulations" begin
    include("lib.jl")

    function apply_window(positions, box_center, rmin, rmax; dowin=true)
        if dowin
            Ngals = size(positions,2)
            select = fill(false, Ngals)
            for i=1:Ngals
                x = positions[1,i]
                y = positions[2,i]
                z = positions[3,i]
                x, y, z = @. (x,y,z) - box_center
                r = √(x^2 + y^2 + z^2)
                if rmin <= r <= rmax
                    select[i] = true
                end
            end
            positions = collect(positions[:,select])
        end
        return positions
    end


    nbar = 1e-3
    L = 4e3
    n_sim = 256
    n_est = 256
    bias = 1.0
    f = 0.7
    pk_matched = false
    fixed_amplitude = false
    fixed_phase = false

    dowin = true
    rmin = 0.3 * L
    rmax = 0.4 * L

    LLL = L .* [1,1,1]
    nnn_sim = n_sim .* [1,1,1]
    nnn_est = n_est .* [1,1,1]
    Δx = LLL ./ nnn_sim
    box_center_1 = [0,0,0]
    box_center_2 = [0,0,0]
    lmax = 4
    ikmin = 1

    kF = 2 * π / L
    kFarr = kF .* [1,1,1]

    Nrandoms = ceil(Int, 10 * prod(LLL) * nbar)
    nbar_rand = nbar * 10

    opts_est = (; nbar, lmax, do_mu_leakage=true, subtract_shotnoise=true,
                voxel_window_power=3)

    id = "matched: $pk_matched, fixed (A,phi) = ($fixed_amplitude, $fixed_phase)"

    kin = @. kF * (0:n_sim)
    pk = fill(0.0, n_sim)
    pk[100] = 30_000.0
    ell = 2

    # keq = 2e-2
    # c = 3 * keq^4
    # a = 3e4 * 4 * keq^3
    # pkfn(k) = a * k / (c + k^4)
    # pk = pkfn.(kin)


    Random.seed!(reinterpret(UInt64, time()))
    seed = rand(UInt64, 4)
    # seed = UInt64[0x01ca4eddf1fdc165, 0xd87d80f08f1e36f8, 0x0ecc4300d7f0a61c, 0xeb0ab21155f56174]
    @show seed

    rfftplan = LogNormalGalaxies.default_plan(nnn_sim)
    rng = Random.GLOBAL_RNG

    simulate(; box_center) = begin
        println("===> Running simulation!")
        Random.seed!(seed)

        @time deltakm = LogNormalGalaxies.draw_phases(rfftplan; rng)
        LogNormalGalaxies.scale_by_pk!(deltakm, pk, 1, (kFarr...,), prod(LLL); rfftplan)
        @time deltarm = rfftplan \ deltakm
        Ncells = prod(nnn_sim)
        vol = prod(LLL)
        @time @. deltarm *= Ncells / vol

        @time @. deltarm = exp(deltarm)
        # @show mean_global(deltarm), var_global(deltarm)
        # @show extrema(deltarm),deltarm[1,1,1]
        @time mean_expGm = 1 / LogNormalGalaxies.mean_global(deltarm)
        @time @. deltarm = deltarm * mean_expGm - 1

        println("Calculate velocities...")
        vx = fill(0.0, nnn_sim...)  # will store vpara in vz
        vy = vx
        vz = fill(0.0, nnn_sim...)
        @time deltakm = rfftplan * deltarm
        n_ell = ell - 1
        tesseral = true
        for m = -n_ell:n_ell
            println("  m = $m")
            deltakm_m = deepcopy(deltakm)
            Ynm_fn = MeasurePowerSpectra.get_ylm_fn(n_ell, m; tesseral)
            @time LogNormalGalaxies.iterate_kspace(deltakm_m; usethreads=true) do ijk_local, ijk_global
                n = norm(ijk_global)
                kx = kF * ijk_global[1]
                ky = kF * ijk_global[2]
                kz = kF * ijk_global[3]
                k = kF * n
                if n == 0
                    deltakm_m[ijk_local...] = 0
                else
                    deltakm_m[ijk_local...] *= im * Ynm_fn(kx, ky, kz) / k
                end
            end
            @time vz_m = rfftplan \ deltakm_m
            @time @. vz_m *= 4 * π / (2 * n_ell + 1)

            ylm = conj(Ynm_fn(0, 0, 1))
            @. vz_m *= ylm

            # box_corner = box_center - LLL ./ 2 .+ [0,0,1e8]
            # @time MeasurePowerSpectra.Ylm_mult!(vz_m, n_ell, m, box_corner, Δx; use_conjugate=true, tesseral)
            @time @. vz += vz_m
        end

        println("Draw galaxies...")
        Ncells = prod(LogNormalGalaxies.size_global(deltarm))
        Navg = nbar * prod(Δx)
        Ngalaxies = Navg * Ncells # * LogNormalGalaxies.mean_global(win)
        @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltarm, vx, vy, vz, Navg, Ngalaxies, Δx, Val(true), Val(2), Val(0); rng)

        x⃗ = collect(xyzv[1:3, :])
        vpara = collect(xyzv[4:6, :])

        return x⃗, vpara
    end

    @time x⃗1, vpara1 = simulate(; box_center=box_center_1)
    @time x⃗2, vpara2 = simulate(; box_center=box_center_2)
    @time x⃗rand = LLL .* (rand(3, Nrandoms) .- 1 // 2)
    @. x⃗1 -= LLL / 2
    @. x⃗2 -= LLL / 2

    # insert ell
    los = [0, 0, 1]
    apply_rsd!(x⃗1, vpara1, f, los)
    apply_rsd!(x⃗2, vpara2, f, los)
    @assert x⃗1 == x⃗2

    x⃗1 = MeasurePowerSpectra.periodic_boundaries!(x⃗1, LLL, box_center_1)
    x⃗2 = MeasurePowerSpectra.periodic_boundaries!(x⃗2, LLL, box_center_2)
    @show size(x⃗1)
    @show size(x⃗2)
    @show size(x⃗rand)
    @show extrema(x⃗1, dims=2)
    @show extrema(x⃗2, dims=2)
    @show extrema(x⃗rand, dims=2)

    x⃗1w = apply_window(x⃗1, box_center_1, rmin, rmax; dowin=false)
    x⃗2w = apply_window(x⃗2, box_center_2, rmin, rmax; dowin)
    x⃗randw1 = apply_window(x⃗rand .+ box_center_1, box_center_1, rmin, rmax; dowin=false)
    x⃗randw2 = apply_window(x⃗rand .+ box_center_2, box_center_2, rmin, rmax; dowin)
    @show size(x⃗1w)
    @show size(x⃗2w)
    @show size(x⃗randw1)
    @show size(x⃗randw2)
    @show extrema(x⃗1w, dims=2)
    @show extrema(x⃗2w, dims=2)
    @show extrema(x⃗randw1, dims=2)
    @show extrema(x⃗randw2, dims=2)

    vol1 = prod(LLL)
    vol2 = dowin ? 4*π/3 * (rmax^3 - rmin^3) : prod(LLL)
    nbar_1 = size(x⃗1w, 2) / vol1
    nbar_rand_1 = size(x⃗randw1, 2) / vol1
    nbar_2 = size(x⃗2w, 2) / vol2
    nbar_rand_2 = size(x⃗randw2, 2) / vol2
    @show nbar nbar_1 nbar_2
    @show nbar_rand nbar_rand_1 nbar_rand_2
    nbar_1 = nbar
    nbar_2 = nbar
    nbar_rand_1 = nbar_rand
    nbar_rand_2 = nbar_rand

    k1, pk1, nmodes1 = xgals_to_pkl_planeparallel(x⃗1w, x⃗randw1, LLL, nnn_est, box_center_1; opts_est..., nbar=nbar_1, nbar_rand=nbar_rand_1)
    k2, pk2, nmodes2 = xgals_to_pkl_planeparallel(x⃗2w, x⃗randw2, LLL, nnn_est, box_center_2; opts_est..., nbar=nbar_2, nbar_rand=nbar_rand_2)

    pk = pk[ikmin:length(k1)]

    k1 = k1[ikmin:end]
    k2 = k2[ikmin:end]
    pk1 = pk1[ikmin:end,:]
    pk2 = pk2[ikmin:end,:]
    nmodes1 = nmodes1[ikmin:end]
    nmodes2 = nmodes2[ikmin:end]

    @show length(k1) length(pk)

    figure()
    hlines(1/nbar, extrema(k1)..., color="0.75", label="Shotnoise")
    plot(k1, pk, "k-", lw=0.5, label=L"P_m(k)")
    plot(NaN, NaN, "k-", label="No win")
    plot(NaN, NaN, "k--", label="win")
    for l=4:lmax
        plot(NaN, NaN, label=" ", alpha=0)
    end
    gca().set_prop_cycle(nothing)
    @show size(pk1)
    plot(k1, pk1[:,:], "-", label=["\$\\ell=$l\$" for l=0:lmax])
    gca().set_prop_cycle(nothing)
    plot(k2, pk2[:,:], "--")
    xlim(0, 0.25)
    xlabel(L"k")
    ylabel(L"P_\ell(k)")
    legend(ncols=2)
    return

    figure()
    title("Real space: $id, dowindow=$dowin")
    text(0.01, 1.25, "\$\\chi_{0,1}^2=$chisq1_l0\$\n\$\\chi_{0,2}^2=$chisq2_l0\$\n\$\\chi_{0,12}^2=$chisq12_l0\$")
    text(0.07, 1.25, "\$\\chi_1^2=$chisq1\$\n\$\\chi_2^2=$chisq2\$\n\$\\chi_{12}^2=$chisq12\$")
    hlines(1, extrema(k1)..., color="0.75")
    plot(k1, pk1[:,1] ./ pk_real[:,1], "-", label=L"\phi=0")
    plot(k2, pk2[:,1] ./ pk_real[:,1], "-", label=L"\phi=\pi")
    plot(k2, middle.(pk1, pk2)[:,1] ./ pk_real[:,1], "k:", label="mean")
    xlim(0, 0.25)
    ylim(0.6, 1.4)
    ylabel(L"P^{\rm sim} / P^{theory}")
    xlabel(L"k")
    legend(loc=1)

    return

    #### to redshift space

end
