# some common functions that don't have a permanent place anywhere

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


function Statistics.middle(x...)
    return mean(x)
end


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
    legend(fontsize="small")
end
