using Revise
using LogNormalGalaxies
using LogNormalGalaxies.Splines
using MeasurePowerSpectra
using DelimitedFiles
using Statistics
using PythonPlot
using Random
using Test

@testset "Complimentary simulations" begin
    include("lib.jl")

    nbar = 1e-4
    L = 4e3
    n_sim = 256
    n_est = 256
    bias = 1.5
    f = 0.7
    fixed_amplitude = false
    fixed_phase = false

    LLL = L .* [1,1,1]
    nnn_est = n_est .* [1,1,1]
    box_center = [0,0,0]
    lmax = 0
    ikmin = 1

    opts_sim = (; nmesh=n_sim, bias, f=(f!=0), fixed_amplitude, fixed_phase,
                voxel_window_power=2, velocity_assignment=6)

    opts_est = (nbar=nbar, lmax=lmax, do_mu_leakage=true, subtract_shotnoise=true,
                voxel_window_power=3)

    id = "fixed_amplitude=$fixed_amplitude, fixed_phase=$fixed_phase"

    #keq = 2e-2
    #c = 3 * keq^4
    #a = 2e4 * 4 * keq^3
    #pk(k) = a * k / (c + k^4)
    D = 1.0
    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pkfn(k) = D^2 * _pk(k)
    kin = @. (2 * π / L) * (0:n_sim)
    pk = pkfn.(kin)


    Random.seed!(reinterpret(UInt64, time()))
    seed = rand(UInt64, 4)
    #seed = UInt64[0x01ca4eddf1fdc165, 0xd87d80f08f1e36f8, 0x0ecc4300d7f0a61c, 0xeb0ab21155f56174]
    @show seed

    simulate_mean(; phase_shift) = begin
        Random.seed!(seed)
        @time x⃗, Ψ = simulate_galaxies(nbar, L, pk; opts_sim..., f=false, phase_shift)
        x⃗ = MeasurePowerSpectra.periodic_boundaries!(x⃗, LLL, box_center)
        ki, pki, nmodesi = xgals_to_pkl_planeparallel(x⃗, LLL, nnn_est, box_center; opts_est..., lmax=0)

        pk_in = deepcopy(pk)
        idxs = 1:length(pki[:,1])
        @. pk_in[idxs] = pk[idxs]^2 * bias^2 / pki[:,1]

        Random.seed!(seed)
        @time x⃗, Ψ = simulate_galaxies(nbar, L, pk_in; opts_sim..., phase_shift)
        x⃗ = MeasurePowerSpectra.periodic_boundaries!(x⃗, LLL, box_center)
        return x⃗, Ψ
    end

    mymean(a, b) = middle(a, b)

    @time x⃗1, Ψ1 = simulate_mean(phase_shift=0)
    @time x⃗2, Ψ2 = simulate_mean(phase_shift=π)

    k1, pk1, nmodes1 = xgals_to_pkl_planeparallel(x⃗1, LLL, nnn_est, box_center; opts_est...)
    k2, pk2, nmodes2 = xgals_to_pkl_planeparallel(x⃗2, LLL, nnn_est, box_center; opts_est...)

    pk = pk[ikmin:length(k1)]

    k1 = k1[ikmin:end]
    k2 = k2[ikmin:end]
    pk1 = pk1[ikmin:end,:]
    pk2 = pk2[ikmin:end,:]
    nmodes1 = nmodes1[ikmin:end]
    nmodes2 = nmodes2[ikmin:end]

    @show length(k1) length(pk)

    multipoles_array = fill(0, lmax + 1)
    multipoles_array[1] = 1
    pk_real = @. bias^2 * pk * multipoles_array'

    sig_pk = @. √(2 / nmodes1) * (pk_real[:,1] + 1 / nbar) * (2 * (0:lmax)' + 1)
    chisq1_i = @. 0.5 * (pk1 - pk_real)^2 / sig_pk^2
    chisq2_i = @. 0.5 * (pk2 - pk_real)^2 / sig_pk^2
    chisq12_i = @. 0.5 * (mymean(pk1, pk2) - pk_real)^2 / sig_pk^2
    idx_range = @. k1 <= 0.20
    chisq1_l0 = round(mean(chisq1_i[idx_range,1]), sigdigits=4)
    chisq2_l0 = round(mean(chisq2_i[idx_range,1]), sigdigits=4)
    chisq12_l0 = round(mean(chisq12_i[idx_range,1]), sigdigits=4)
    chisq1 = round(mean(chisq1_i[idx_range,:]), sigdigits=4)
    chisq2 = round(mean(chisq2_i[idx_range,:]), sigdigits=4)
    chisq12 = round(mean(chisq12_i[idx_range,:]), sigdigits=4)

    figure()
    title("Real space")
    hlines(1/nbar, extrema(k1)..., color="0.75")
    plot(k1, pk_real, "k-", lw=0.5)
    plot(k1, pk1[:,:], "-", label="phase=0")
    gca().set_prop_cycle(nothing)
    plot(k2, pk2[:,:], "--", label="phase=pi")
    plot(k1, mymean.(pk1, pk2), "k:")
    xlim(0, 0.25)
    legend()

    figure()
    title("Real space: $id")
    text(0.01, 1.25, "\$\\chi_{0,1}^2=$chisq1_l0\$\n\$\\chi_{0,2}^2=$chisq2_l0\$\n\$\\chi_{0,12}^2=$chisq12_l0\$")
    text(0.07, 1.25, "\$\\chi_1^2=$chisq1\$\n\$\\chi_2^2=$chisq2\$\n\$\\chi_{12}^2=$chisq12\$")
    hlines(1, extrema(k1)..., color="0.75")
    plot(k1, pk1[:,1] ./ pk_real[:,1], "-", label=L"\phi=0")
    plot(k2, pk2[:,1] ./ pk_real[:,1], "-", label=L"\phi=\pi")
    plot(k2, mymean.(pk1, pk2)[:,1] ./ pk_real[:,1], "k:", label="mean")
    xlim(0, 0.25)
    ylim(0.6, 1.4)
    ylabel(L"P^{\rm sim} / P^{theory}")
    xlabel(L"k")
    legend(loc=1)

    return

    #### to redshift space

    los = [0, 0, 1]
    apply_rsd!(x⃗1, Ψ1, f, los)
    apply_rsd!(x⃗2, Ψ2, f, los)
    x⃗1 = MeasurePowerSpectra.periodic_boundaries!(x⃗1, LLL, box_center)
    x⃗2 = MeasurePowerSpectra.periodic_boundaries!(x⃗2, LLL, box_center)
    k1, pk1, nmodes1 = xgals_to_pkl_planeparallel(x⃗1, LLL, nnn_est, box_center; opts_est...)
    k2, pk2, nmodes2 = xgals_to_pkl_planeparallel(x⃗2, LLL, nnn_est, box_center; opts_est...)
    k1 = k1[ikmin:end]
    k2 = k2[ikmin:end]
    pk1 = pk1[ikmin:end,:]
    pk2 = pk2[ikmin:end,:]
    nmodes1 = nmodes1[ikmin:end]
    nmodes2 = nmodes2[ikmin:end]

    β = f / bias
    pkm_kaiser = @. bias^2 * Arsd_Kaiser(β, (0:lmax)') * pk

    sig_pk = @. √(2 / nmodes1) * (pkm_kaiser[:,1] + 1 / nbar) * (2 * (0:lmax)' + 1)
    chisq1_i = @. 0.5 * (pk1 - pkm_kaiser)^2 / sig_pk^2
    chisq2_i = @. 0.5 * (pk2 - pkm_kaiser)^2 / sig_pk^2
    chisq12_i = @. 0.5 * (mymean(pk1, pk2) - pkm_kaiser)^2 / sig_pk^2
    idx_range = @. k1 <= 0.20
    chisq1_l0 = round(mean(chisq1_i[idx_range,1]), sigdigits=4)
    chisq2_l0 = round(mean(chisq2_i[idx_range,1]), sigdigits=4)
    chisq12_l0 = round(mean(chisq12_i[idx_range,1]), sigdigits=4)
    chisq1 = round(mean(chisq1_i[idx_range,:]), sigdigits=4)
    chisq2 = round(mean(chisq2_i[idx_range,:]), sigdigits=4)
    chisq12 = round(mean(chisq12_i[idx_range,:]), sigdigits=4)

    figure()
    title("Redshift space: $id")
    hlines(1/nbar, extrema(k1)..., color="0.75")
    plot(k1, pkm_kaiser[:,:], "-", lw=0.5)
    gca().set_prop_cycle(nothing)
    plot(k1, pk1[:,:], "-")
    gca().set_prop_cycle(nothing)
    plot(k2, pk2[:,:], "--")
    plot(k1, mymean.(pk1, pk2), "k:")
    xlim(0, 0.25)

    figure()
    title("Redshift space: $id")
    text(0.01, 1.25, "\$\\chi_{0,1}^2=$chisq1_l0\$\n\$\\chi_{0,2}^2=$chisq2_l0\$\n\$\\chi_{0,12}^2=$chisq12_l0\$")
    text(0.07, 1.25, "\$\\chi_1^2=$chisq1\$\n\$\\chi_2^2=$chisq2\$\n\$\\chi_{12}^2=$chisq12\$")
    hlines(1, extrema(k1)..., color="0.75")
    plot(k1, pk1[:,1] ./ pkm_kaiser[:,1], "-")
    plot(k2, pk2[:,1] ./ pkm_kaiser[:,1], "-")
    plot(k2, mymean.(pk1, pk2)[:,1] ./ pkm_kaiser[:,1], "k:")
    xlim(0, 0.25)
    ylim(0.6, 1.4)
end
