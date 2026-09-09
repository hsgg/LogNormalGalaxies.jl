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

PencilFFTs.range_local(arr::AbstractArray) = Tuple(1:s for s in size_global(arr))


############### functions to extend FFTW ####

# PencilFFTs.jl needs 'allocate_input()', but FFTW doesn't provide it:
PencilFFTs.allocate_input(plan::FFTW.FFTWPlan{T}) where {T} = Array{T}(undef, size(plan))

# 'allocate_input()' is the one place where the array type of the whole pipeline
# is decided: 'draw_phases()' calls it once, and every other array is derived
# from that one by transforming, 'similar()', or 'copy()'. A backend therefore
# only needs a method here to be usable throughout -- or, if adding a method is
# undesirable (e.g. because the backend is a test-only dependency), the array
# can be handed to 'draw_phases(rfftplan; deltar)' directly.


############### element types and array types ####

# like_array(): Return `x` where `arr` lives -- same device, same array type --
# and at `arr`'s precision, keeping `x`'s own realness, so that a real window or
# power spectrum stays real against a complex field. Precision and device have
# to travel together: a Float64 value is as fatal to a Float32 GPU run as a host
# pointer is, and a user-supplied array can be wrong in both ways at once.
#
# Neither obvious one-liner does the job. `convert(AbstractArray{T}, x)` changes
# only the element type (it is the identity when that already matches, which is
# why it is the first step here), and `typeof(arr)(x)` is worse: `typeof` is
# fully parameterised, e.g. MtlArray{ComplexF32,3,Metal.PrivateStorage}, so it
# would force the real factors to complex. Hence convert, then similar() +
# copyto!, the usual GPUArrays idiom.

# the precision from `arr`, the realness from `x`
match_precision_type(arr, x) = (R = real(eltype(arr)); eltype(x) <: Complex ? complex(R) : R)

# host destination: only the element type can differ
like_array(arr::Array, x::AbstractArray) =
    convert(AbstractArray{match_precision_type(arr, x)}, x)

# a PencilArray keeps its decomposition -- rebuilding one from size() would be
# wrong, since that is only the process-local block -- but not its storage: the
# parent goes through the methods above, so a host gather from `similar_local()`
# lands back on `arr`'s device. The wrapper is built on `pencil(arr)`, not
# `pencil(x)`, because a Pencil carries its array type and rejects a parent that
# does not match; both are the same decomposition at every call site.
function like_array(arr::PencilArray, x::PencilArray)
    p = like_array(parent(arr), parent(x))
    p === parent(x) && return x  # nothing to convert and nowhere to move
    return PencilArray(pencil(arr), p)
end

# and its parent is where broadcasting and `@strided` actually put the factors
like_array(arr::PencilArray, x::AbstractArray) = like_array(parent(arr), x)

# device destination: narrow on the host, then move across in one go
function like_array(arr, x::AbstractArray)
    y = convert(AbstractArray{match_precision_type(arr, x)}, x)
    z = similar(arr, eltype(y), size(y))
    copyto!(z, y)
    return z
end


# similar_local(): uninitialized, with `arr`'s local shape and index semantics
# but always on the host, for the k-space gathers built by scalar assignment. A
# device array becomes a plain Array that `like_array()` moves back; a PencilArray
# has to stay one, since `iterate_kspace()` reads its `range_local()` for global
# indices. A distributed GPU array needs both at once, hence host storage inside a
# PencilArray wrapper: `similar(::Pencil, Array)` restates the decomposition over
# host storage (and returns the same pencil when it already is), because a Pencil
# carries its array type and rejects a mismatched parent. `size(parent(arr))` is
# the local block in *memory* order, which is what the constructor checks.
similar_local(arr, ::Type{T}) where {T} = Array{T}(undef, size(arr)...)
similar_local(arr::PencilArray, ::Type{T}) where {T} =
    PencilArray(similar(pencil(arr), Array), Array{T}(undef, size(parent(arr))))
similar_local(arr) = similar_local(arr, eltype(arr))


# memory_dim(): where logical dimension `d` of `arr` lives in memory order.
# PencilArrays broadcast in memory order (PencilArrays/src/broadcast.jl), as
# does `@strided`, which unwraps them to their parent. A permutation `p` puts
# logical dimension `p[i]` in slot `i`, so `d` sits at `inv(p)[d]`.
memory_dim(arr, d) = d
memory_dim(arr::PencilArray, d) = inv(permutation(arr))[d]


# broadcast_dim(): `v`, covering the local extent of dimension `d`, placed
# where `arr` lives and reshaped to broadcast along that dimension. A separable
# k-space operation is then one fused broadcast against three of these: no N^3
# temporary, no scalar indexing, and one code path for every backend.
function broadcast_dim(arr, v::AbstractVector, d)
    dmem = memory_dim(arr, d)
    shape = ntuple(i -> i == dmem ? length(v) : 1, ndims(arr))
    return reshape(like_array(arr, v), shape)
end


# to_host(): bring `x` to the CPU for the steps that have to run there. Not the
# inverse of `like_array()`, despite the resemblance: that one matches a
# prototype array -- precision included -- and on a host pipeline moves nothing
# at all, whereas this is the unconditional one-way trip, with no prototype and
# nothing to say about the element type. Dispatch is by exclusion, so that no
# GPU package needs to be named here:
# ordinary arrays and numbers (the velocity components are literal 0 when there
# are no redshift-space distortions) pass through untouched, and anything else is
# assumed to live on a device and is brought over.
#
# PencilArrays are not unwrapped: the callers derive *global* indices from
# `range_local()`, which a plain Array would answer wrongly on every rank but 0. A
# device-backed one is restated over host storage instead, as in `similar_local()`.
to_host(x::Number) = x
to_host(x::Array) = x
to_host(x::PencilArray) = parent(x) isa Array ? x :
    PencilArray(similar(pencil(x), Array), Array(parent(x)))
to_host(x::AbstractArray) = Array(x)


# local_data(): the process-local block, for reductions. A PencilArray reduces
# collectively -- `mean`/`sum`/`var`/`extrema` build a user-defined MPI.Op,
# which aarch64 cannot do at all (JuliaParallel/MPI.jl#404) and which would
# reduce a second time under the Allgather in `*_global()`. Everything else,
# device arrays included, reduces where it already lives.
local_data(x) = x
local_data(x::PencilArray) = parent(x)


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
