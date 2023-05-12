#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
using Revise
using PyPlot
using DelimitedFiles
using Splines
using YAML
using Statistics
using PlaneParallelRedshiftSpaceDistortions


function plot_pkl(k, pkl, pkl_err, pkl_kaiser, nbar; n=0, nrlzs=1, pk_g=nothing)
    #hlines(1/nbar, extrema(km)..., color="0.75", label="Shot noise")
    plot(k, k.^n ./ nbar, color="0.75", label="Shot noise")
    if !isnothing(pk_g)
        plot(k, k.^n .* pk_g, "k", label="input \$k^$n P(k)\$")
    end
    for m=1:size(pkl_kaiser,2)
        plot(k, k.^n.*pkl_kaiser[:,m], "C$(m-1)--", alpha=0.7)
    end
    for m=1:size(pkl,2)
        if !isnothing(pkl_err)
            errorbar(k, k.^n.*pkl[:,m], k.^n.*pkl_err[:,m], c="C$(m-1)", alpha=0.7)
        end
        if !isnothing(pkl_err) && nrlzs > 1
            errorbar(k, k.^n.*pkl[:,m], k.^n.*pkl_err[:,m] ./ sqrt(nrlzs), c="C$(m-1)", elinewidth=4, alpha=0.7)
        end
        plot(k, k.^n.*pkl[:,m], "C$(m-1)-", label="\$k^$n P_{$(m-1)}(k)\$", alpha=0.7)
    end
    xlabel(L"k")
    ylabel("\$k^$n P_\\ell(k)\$")
    xscale("log")
    ylim(top=1.1*maximum(k.^n.*pkl))
    #xlim(right=0.3)
    legend(fontsize="small")
end


function plot_pkl_diff(k, pkl, pkl_err, pkl_kaiser, nbar; n=0, nrlzs=1)
    hlines(1, extrema(k)..., color="0.8")
    hlines([0.99,1.01], extrema(k)..., color="0.7", linestyle="--")
    hlines([0.95,1.05], extrema(k)..., color="0.6", linestyle=":")
    for l=[0,2]
        m = l+1
        ymid = pkl[:,m] ./ pkl_kaiser[:,m]
        yerr = @. pkl_err[:,m] / abs(pkl_kaiser[:,m])
        errorbar(k, ymid, yerr, c="C$(m-1)", alpha=0.7)
        if nrlzs > 1
            errorbar(k, ymid, yerr ./ sqrt(nrlzs), c="C$(m-1)", elinewidth=4, alpha=0.7)
        end
        plot(k, ymid, "C$(m-1)-", label="\$P_{$(m-1)}(k)\$", alpha=0.7)
    end
    xlabel(L"k")
    ylabel(L"\hat P^{\rm pp}_\ell(k) / P^{\rm Kaiser}_\ell(k)")
    #xscale("log")
    xlim(left=0, right=0.25)
    ylim(0.9, 1.1)
end


function readdlm_cols(fname, cols)
    data, header = readdlm(fname; header=true)
    key = []
    for c in cols
        k = data[:,c]
        push!(key, k)
    end
    return key
end


function abbreviate(v)
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
    return v
end


#function make_title(; kwargs...)
#    t = join(abbreviate.(keys(kwargs)), ", ")
#    t *= " = "
#    t *= join(values(kwargs), ", ")
#    title(t)
#end


function make_title(; kwargs...)
    t = ""
    s = ""
    for (k,v) in kwargs
        k = abbreviate(k)
        t *= "$s$k=$v"
        s = ", "
    end
    title(t)
end


function plot_pkest(cfg_fname)
    cfg_fbase = splitext(cfg_fname)[1]
    cfg = YAML.load_file(cfg_fname)
    infname = (@__DIR__)*"/../sim_results/"*basename(cfg_fbase)*"_pkest.tsv"
    outfname1 = (@__DIR__)*"/../figs/"*basename(cfg_fbase)*"_pkest.pdf"
    outfname2 = (@__DIR__)*"/../figs/"*basename(cfg_fbase)*"_pkest_rdiff.pdf"
    mkpath(dirname(outfname1))

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

    data = readdlm((@__DIR__)*"/"*cfg["pkfname"], comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pk(k) = D^2 * _pk(k)

    km, pkm, nmodes, pkm_err = readdlm_cols(infname, [1, 2:6, 7, 8:12])
    pkm[1,2:end] .= 0
    pkm_err[1,2:end] .= 0
    #@assert all(isfinite.(pkm_err))
    #@assert all(pkm_err .>= 0)


    # theory
    pk_g = @. b^2 * pk(km)
    β = f / b
    #pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pk(km)
    pkl_kaiser = @. b^2 * Arsd_l_exp(km*f*sigma_psi, β, (0:4)') * pk(km)


    n = 0
    Δx_sim = L / n_sim
    Δx_est = L / n_est
    Wmesh_sim = @. sinc(km * Δx_sim / (2 * π))
    Wmesh_est = @. sinc(km * Δx_est / (2 * π))
    if est_grid_assignment == 2
        est_grid_assignment = 6
    end
    p = sim_vox + est_grid_assignment - est_vox
    @show sim_vox,est_grid_assignment,p

    # plot
    figure()
    make_title(; L, f, n_sim, n_est, sim_vox, est_vox, sim_velo)#, grid_assignment)
    plot_pkl(km, pkm, pkm_err, pkl_kaiser, nbar; n, nrlzs, pk_g)
    tight_layout()
    println(outfname1)
    savefig(outfname1, metadata=Dict("CreationDate"=>nothing))

    figure()
    make_title(; L, f, n_sim, n_est, sim_vox, est_vox, sim_velo, #=grid_assignment,=# xshift=fxshift_sim)
    plot_pkl_diff(km, pkm ./ Wmesh_est .^ (0*2*p), pkm_err, pkl_kaiser, nbar; n, nrlzs)
    plot(km, Wmesh_est.^(6-0*2*p), label="p=3 (\$W_e\$)")
    plot(km, Wmesh_sim.^(2*p-0*2*p), label="p=$p (\$W_s\$)")
    legend(fontsize="small")
    tight_layout()
    savefig(outfname2, metadata=Dict("CreationDate"=>nothing))
end


for cfg_fname in ARGS
    plot_pkest(cfg_fname)
end
