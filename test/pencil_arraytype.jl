# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# A distributed GPU run is a PencilArray over a device array, which nothing here
# can produce: MPI.jl has no Metal extension, so an MtlArray never reaches
# MPI.Isend. So the device is faked. TestArray is a host Array that is not an
# `Array`, which is the only property these functions dispatch on. Modelled on
# PencilArrays' own test/array_types.jl.

using Test
using MPI
using PencilFFTs
using LogNormalGalaxies


struct TestArray{T,N} <: AbstractArray{T,N}
    data :: Array{T,N}
end

TestArray{T}(::UndefInitializer, dims::Dims) where {T} = TestArray(Array{T}(undef, dims))
TestArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} = TestArray(Array{T}(undef, dims))
TestArray{T}(::UndefInitializer, dims::Integer...) where {T} = TestArray{T}(undef, dims)

Base.parent(u::TestArray) = u.data
Base.size(u::TestArray) = size(u.data)
Base.similar(u::TestArray, ::Type{T}, dims::Dims) where {T} = TestArray(similar(u.data, T, dims))
Base.getindex(u::TestArray, i...) = getindex(u.data, i...)
Base.setindex!(u::TestArray, v, i...) = setindex!(u.data, v, i...)


@testset "PencilArray array types" begin

    LogNormalGalaxies.start_mpi()
    comm = MPI.COMM_WORLD
    dims = (8, 8, 8)

    @testset "host pencil is untouched" begin
        u = PencilArray{ComplexF64}(undef, Pencil(dims, comm))

        # must be a no-op, or every host gather pays for new MPI buffers
        @test similar(pencil(u), Array) === pencil(u)

        s = LogNormalGalaxies.similar_local(u)
        @test s isa PencilArray
        @test parent(s) isa Array
        @test eltype(s) === ComplexF64
        @test size(s) == size(u)

        # and coming back moves nothing at all
        @test LogNormalGalaxies.like_array(u, s) === s
        @test LogNormalGalaxies.to_host(u) === u
    end

    @testset "device pencil round-trip" begin
        pen = Pencil(TestArray, dims, comm)
        u = PencilArray{ComplexF64}(undef, pen)
        @test parent(u) isa TestArray

        s = LogNormalGalaxies.similar_local(u, Float64)

        # host storage, so the k-space gathers can assign element by element...
        @test parent(s) isa Array
        s[1,1,1] = 1.0
        @test s[1,1,1] == 1.0

        # ...while keeping the PencilArray semantics iterate_kspace() needs
        @test s isa PencilArray
        @test range_local(s) == range_local(u)
        @test size_global(s) == size_global(u)
        @test permutation(s) == permutation(u)

        # iterate_kspace() has to see the same global indices either way
        idx_u = Tuple[]
        idx_s = Tuple[]
        LogNormalGalaxies.iterate_kspace(u) do _, ijk_global; push!(idx_u, ijk_global) end
        LogNormalGalaxies.iterate_kspace(s) do _, ijk_global; push!(idx_s, ijk_global) end
        @test idx_u == idx_s

        # and like_array() puts the result back on the device, at the field's
        # precision but keeping its own realness
        d = LogNormalGalaxies.like_array(u, s)
        @test parent(d) isa TestArray
        @test eltype(d) === Float64
        @test d[1,1,1] == 1.0
        @test range_local(d) == range_local(u)

        # to_host() brings a device-backed PencilArray over without unwrapping it
        h = LogNormalGalaxies.to_host(u)
        @test h isa PencilArray
        @test parent(h) isa Array
        @test range_local(h) == range_local(u)
        @test size_global(h) == size_global(u)
    end
end


# vim: set sw=4 et sts=4 :
