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


############### using @strided with GPU arrays ####
#
# Strided.jl works on GPU arrays from v2.5 on: StridedGPUArraysExt is keyed on
# GPUArrays (which every GPU backend depends on) so it loads by itself, and from
# v2.5 an allocating `@strided` broadcast allocates its result with
# `similar(parent, ...)`, keeping it on the device. Earlier 2.x allocated a host
# Array instead, which would silently mix a host destination with device
# sources, hence the compat lower bound.
#
# One rule has to be respected at the call sites: `@strided` does not evaluate
# its expression, it captures it into a `Strided.CaptureArgs` tree that is
# passed to the kernel as an argument. Anything that is not a bitstype
# therefore cannot appear inside the expression. In particular a type
# conversion written inline,
#
#     @strided @. deltak /= T(√NNN)          # T is a DataType => not isbits
#
# fails to compile with "passing non-bitstype argument". Compute such scalars
# into a local first and reference the local:
#
#     norm_factor = T(√NNN)
#     @strided @. deltak /= norm_factor
#
# Plain functions are fine, since they are singletons: `√(pkG * vol)` compiles.


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

# 'allocate_input()' is the one place where the array type of the whole pipeline
# is decided: 'draw_phases()' calls it once, and every other array is derived
# from that one by transforming, 'similar()', or 'copy()'. A backend therefore
# only needs a method here to be usable throughout -- or, if adding a method is
# undesirable (e.g. because the backend is a test-only dependency), the array
# can be handed to 'draw_phases(rfftplan; deltar)' directly.


############### element types ####

# match_eltype(): Return `x` with the real element type used by `arr`, so that a
# user-supplied array (typically Float64) can be fed to a pipeline running at a
# different precision. Returns `x` itself when no conversion is needed, so the
# Float64 path is untouched.
function match_eltype(arr, x::AbstractArray)
    R = real(eltype(arr))
    T = eltype(x) <: Complex ? complex(R) : R
    return eltype(x) === T ? x : convert(AbstractArray{T}, x)
end


# like_array(): Put `x` into an array of the same kind as `arr` -- same device,
# same array type -- keeping `x`'s own element type, which is typically real
# while `arr` is complex. Small precomputed factors are built on the host, where
# scalar assignment is allowed, and moved over with this in one go.
# No-ops for the cases where the caller already built `x` as the right kind of
# array: a host Array for a host destination, or a PencilArray made with
# similar(). Going through similar()+copyto! for a PencilArray would in fact be
# wrong, since its size() is only the process-local part.
like_array(arr::Array, x::AbstractArray) = x
like_array(arr::PencilArray, x::PencilArray) = x
function like_array(arr, x::AbstractArray)
    y = similar(arr, eltype(x), size(x))
    copyto!(y, x)
    return y
end


# to_host(): The dual of `like_array()`, for the steps that have to run on the
# CPU. Dispatch is by exclusion, so that no GPU package needs to be named here:
# ordinary arrays, numbers (the velocity components are literal 0 when there are
# no redshift-space distortions) and PencilArrays pass through untouched, and
# anything else is assumed to live on a device and is brought over.
#
# PencilArrays must not be converted: the callers derive *global* indices from
# `range_local()`, which a plain Array would answer wrongly on every rank but 0.
to_host(x::Number) = x
to_host(x::Array) = x
to_host(x::PencilArray) = x
to_host(x::AbstractArray) = Array(x)


############### functions to extend PencilFFTs ####
# none!


############### iterate_kspace()

function calc_global_indices(ijk_local, localrange, nxyz, nxyz2; wrap)
    DIMS = length(ijk_local)

    ijk_global = MVector(ijk_local...)

    for d in 1:DIMS
        ig = localrange[d][ijk_local[d]] - 1  # global index of local index in direction d

        if wrap
            ig = (ig < nxyz2[d]) ? ig : (ig - nxyz[d])
        end

        ijk_global[d] = ig
    end

    return (ijk_global...,)
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
    nxyz = size_global(deltak)
    nx2 = first_half_dimension ? nxyz[1] : (nxyz[1] ÷ 2 + 1)
    nxyz2 = (nx2, (@. nxyz[2:end] ÷ 2 + 1)...,)
    localrange = range_local(deltak)

    @maybe_threads usethreads for ijk in CartesianIndices(deltak)
        ijk_local = Tuple(ijk)
        ijk_global = calc_global_indices(ijk_local, localrange, nxyz, nxyz2; wrap)
        func(ijk_local, ijk_global)
    end

    return deltak
end

# The index (1,1,1) maps to x⃑ = (0,0,0).
iterate_rspace(args...; kwargs...) = iterate_kspace(args...; first_half_dimension=false, wrap=false, kwargs...)



# vim: set sw=4 et sts=4 :
