# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


module LinearInterpolations

export LinearInterpolation, LinearInterpolation!
export PeriodicNeighborView3D


struct LinearInterpolation{Tr,D,Ti}
    x0::Vector{Ti}
    coeff::Array{Tr,D}
end


function LinearInterpolation(D)
    x0 = zeros(Float64, D)
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
    #@show Δxyz
    result = sum(li.coeff[i+1,j+1,k+1] * prod(Δxyz .^ (i,j,k)) for i=0:1, j=0:1, k=0:1)
    return result
end

(li::LinearInterpolation)(x) = evaluate(li, x)



###########################################################################

struct PeriodicNeighborView3D{Tarr}
    ijk::Vector{Int}
    farr::Tarr
end

function Base.getindex(pnv::PeriodicNeighborView3D, i, j, k)
    return pnv.farr[mod1(pnv.ijk[1] + i - 1, end),
                    mod1(pnv.ijk[2] + j - 1, end),
                    mod1(pnv.ijk[3] + k - 1, end)]
end



end
