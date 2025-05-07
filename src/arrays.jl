# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# In this file we define functions so that we can use our code either with FFTW
# or with PencilFFTs, without needing big changes. Generally, PencilFFTs need
# more support, so if it works with PencilFFTs, it should also work with FFTW.
# In the future, one might also consider supporting Joe's DistributedFFT.


######### functions for both FFTW and PencilFFTs array types

allocate_array(shape, T::DataType) = Array{T}(undef, shape...)
allocate_array(pen::Pencil, T::DataType) = PencilArray{T}(undef, Pencil(pen))

# allocate input, but allow a different type (useful to ensure the same topology is used)
allocate_array(p::FFTW.FFTWPlan, T::DataType) = allocate_array(size(p), T)
allocate_array(p::PencilFFTPlan, T::DataType) = allocate_array(PencilFFTs.pencil_input(p), T)


############### functions to extend PencilArrays ####

Base.deepcopy(pa::PencilArray) = PencilArray(pencil(pa), deepcopy(parent(pa)))
Strided.StridedView(a::PencilArray) = Strided.StridedView(parent(a))  # FIXME: incomplete if there are permutations. To fix, need to figure out how to get the permutated view. However, this should only matter for things like matrix multiplication, where it is NOT just element-wise.


############### functions to extend base Arrays ####

# this is *un*like 'size_local()', because a pencil also has info about the
# other processes. It is used for 'allocate_array()'.
PencilFFTs.pencil(arr::AbstractArray) = size(arr)

PencilFFTs.global_view(arr::AbstractArray) = arr

PencilFFTs.size_global(arr::AbstractArray) = size(arr)

PencilFFTs.sizeof_global(arr::AbstractArray) = sizeof(arr)

PencilFFTs.range_local(arr::AbstractArray) = begin
    r = ()
    for s in size_global(arr)
        r = (r..., 1:s)
    end
    return r
end


############### functions to extend FFTW ####

# PencilFFTs.jl needs 'allocate_input()', but FFTW doesn't provide it:
PencilFFTs.allocate_input(plan::FFTW.FFTWPlan{T}) where {T} = Array{T}(undef, size(plan))


############### functions to extend PencilFFTs ####
# none!


############### iterate_kspace()

function calc_global_indices(ijk_local, localrange, nx2, ny2, nz2, nx, ny, nz; wrap)
    ig = localrange[1][ijk_local[1]] - 1  # global index of local index i
    jg = localrange[2][ijk_local[2]] - 1  # global index of local index j
    kg = localrange[3][ijk_local[3]] - 1  # global index of local index k
    if wrap
        ig = ig < nx2 ? ig : ig-nx
        jg = jg < ny2 ? jg : jg-ny
        kg = kg < nz2 ? kg : kg-nz
    end
    return ig, jg, kg
end


# https://discourse.julialang.org/t/conditional-multithreading/32421/12?u=hsgg
macro maybe_threads(usethreads, expr)
    return quote
        if $(usethreads)
            Threads.@threads $(expr)
        else
            $(expr)
        end
    end |> esc
end


function iterate_kspace(func, deltak; usethreads=false, first_half_dimension=true, wrap=true)
    nx, ny, nz = size_global(deltak)
    nx2 = first_half_dimension ? nx : (div(nx,2) + 1)
    ny2 = div(ny,2) + 1
    nz2 = div(nz,2) + 1
    localrange = range_local(deltak)

    @maybe_threads usethreads for k=1:size(deltak,3)
        for j=1:size(deltak,2), i=1:size(deltak,1)
            ijk_local = (i, j, k)
            ijk_global = calc_global_indices(ijk_local, localrange, nx2, ny2, nz2, nx, ny, nz; wrap)
            func(ijk_local, ijk_global)
        end
    end

    return deltak
end

# The index (1,1,1) maps to x⃑ = (0,0,0).
iterate_rspace(args...; kwargs...) = iterate_kspace(args...; first_half_dimension=false, wrap=false, kwargs...)



# vim: set sw=4 et sts=4 :
