

function apply_rsd!(x⃗, Ψ, f, los=[0,0,1])
    Ngals = size(x⃗,2)
    for i=1:Ngals
        x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
    end
    return x⃗
end


