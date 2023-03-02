module LinearInterpolations

export InterpolationOctants, InterpolationOctants!, LinearInterpolation


struct LinearInterpolation{Tr,D,Ti}
    x0::Vector{Ti}
    coeff::Array{Tr,D}
end


function LinearInterpolation(D)
    x0 = zeros(Int64, D)
    coeff = zeros(Float64, fill(2, D)...)
    return LinearInterpolation(x0, coeff)
end


function LinearInterpolation!(li, x0, fvals)
    li.coeff[1,1,1] = fvals[1,1,1]

    li.coeff[2,1,1] = fvals[2,1,1] - fvals[1,1,1]
    li.coeff[1,2,1] = fvals[1,2,1] - fvals[1,1,1]
    li.coeff[1,1,2] = fvals[1,1,2] - fvals[1,1,1]

    li.coeff[2,2,1] = fvals[2,2,1] - fvals[2,1,1] - fvals[1,2,1] + fvals[1,1,1]
    li.coeff[2,1,2] = fvals[2,1,2] - fvals[2,1,1] - fvals[1,1,2] + fvals[1,1,1]
    li.coeff[1,2,2] = fvals[1,2,2] - fvals[1,2,1] - fvals[1,1,2] + fvals[1,1,1]

    li.coeff[2,2,2] = (fvals[2,2,2]
                       - fvals[2,2,1] - fvals[2,1,2] - fvals[1,2,2]
                       + fvals[2,1,1] + fvals[1,2,1] + fvals[1,1,2]
                       - fvals[1,1,1])
    li.x0 .= x0
    return li
end


function evaluate(li::LinearInterpolation{T,3}, x) where {T}
    Δx = x[1] - li.x0[1]
    Δy = x[2] - li.x0[2]
    Δz = x[3] - li.x0[3]
    Δxyz = (Δx, Δy, Δz)
    result = sum(li.coeff[i+1,j+1,k+1] * prod(Δxyz .^ (i,j,k)) for i=0:1, j=0:1, k=0:1)
    return result
end

(li::LinearInterpolation)(x) = evaluate(li, x)



###########################################################################


struct InterpolationOctants{Toct,Tfarr}
    octant::Toct
    fvals::Tfarr
end


function InterpolationOctants(D)
    octants = [LinearInterpolation(D) for dii=0:1, djj=0:1, dkk=0:1]
    fvals = Array{Float64}(undef, fill(2, D)...)
    return InterpolationOctants(octants, fvals)
end


function InterpolationOctants!(interpo, i, j, k, farr)
    nx, ny, nz = size(farr)
    for ii=(i-1):i, jj=(j-1):j, kk=(k-1):k
        x0 = (ii, jj, kk)
        for dii=0:1, djj=0:1, dkk=0:1
            interpo.fvals[1+dii,1+djj,1+dkk] = farr[mod1(ii+dii,end), mod1(jj+djj,end), mod1(kk+dkk,end)]
        end
        li = interpo.octant[ii-i+2, jj-j+2, kk-k+2]
        LinearInterpolation!(li, x0, interpo.fvals)
    end
    return interpo
end


function evaluate(iq::InterpolationOctants, x::Tuple)
    ijk = @. Int(x >= 0) + 1
    li = iq.octant[ijk...]
    res = evaluate(li, x)
    return res
end

#evaluate(qi::InterpolationOctants, x) = evaluate(qi, (x...,))

(qi::InterpolationOctants)(x) = evaluate(qi, x)


################################################################################

function runtests()
end


end
