# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# Helpers shared by more than one test file. runtests.jl includes it once; the
# individual files include it behind an `isdefined` guard so they still work on
# their own, and so a second include does not overwrite the method.

using LinearAlgebra: norm


# `≈` is itself norm-based. This exists only to *report* the error, which `≈`
# cannot do. No conversion is needed anywhere: mixed precisions promote on
# their own, and `norm` reduces pairwise, so even a pure Float32 norm of these
# arrays is good to ~3e-7 relative -- far below the tolerances in the callers.
reldiff(a, b) = norm(a .- b) / norm(a)


# vim: set sw=4 et sts=4 :
