

function apply_rsd!(x⃗, Ψ, f, los=[0,0,1])
    Ngals = size(x⃗,2)
    for i=1:Ngals
        x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
    end
    return x⃗
end


function apply_periodic_boundaries!(x⃗, Lxyz::Tuple, box_center::Tuple)
    xshift = @. box_center - Lxyz / 2

    @. x⃗ = mod(x⃗ - xshift, Lxyz) + xshift

    return x⃗
end

apply_periodic_boundaries!(x⃗, Lxyz, box_center=(0,0,0)) = apply_periodic_boundaries!(x⃗, (Lxyz...,), (box_center...,))
