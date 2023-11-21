
function apply_rsd!(positions, velocities; los=:vlos)
    apply_rsd!(positions, velocities, los)
end

function apply_rsd!(positions, velocities, los=:vlos)
    apply_rsd!(positions, velocities, Val(los))
end


function apply_rsd!(positions, velocities, ::Val{los}) where {los}

    Ngals = size(positions,2)

    Threads.@threads for i=1:Ngals

        xvec = @view positions[:,i]
        velo = @view velocities[:,i]

        if los == :vlos
            mylos = xvec
        else
            mylos = los
        end

        r2 = mylos' * mylos

        uz_r = (velo' * mylos) / r2

        @. xvec += uz_r * mylos
    end

    return positions
end


# compatibility:
function apply_rsd!(xyz, psi, f, los)
    velo = @strided @. f * psi
    apply_rsd!(xyz, velo, los)
end


function apply_periodic_boundaries!(x⃗, Lxyz::Tuple, box_center::Tuple)
    xshift = @. box_center - Lxyz / 2

    @. x⃗ = mod(x⃗ - xshift, Lxyz) + xshift

    return x⃗
end

apply_periodic_boundaries!(x⃗, Lxyz, box_center=(0,0,0)) = apply_periodic_boundaries!(x⃗, (Lxyz...,), (box_center...,))
