# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


#!/usr/bin/env julia


dname = (@__DIR__)*"/../sim_configs"


config_head = raw"""
randomseed: "[0x659827fbbc75b78b, 0xe5aa8abfea05b963, 0x05caadbdc25751c9, 0xea636b29b0cfc8fb]"

pkfname: "./../../test/matterpower_zeff=0.38.dat"

bias: 1.455  # same as Agrawal+2017
D: 1.0       # already included in power spectrum
nbar: 3e-5   # non need to change between sims
L: 3e3       # let's keep it constant
nrlzs: 100  # can be the same for all

n_sim: 256
n_est: 256

f: 0.71
sigma_psi: 0.0

sim_vox: 1
sim_velo: 1
est_vox: 0
est_grid_assignment: 1
fxshift_sim: 0.0
fxshift_est: 0.0
"""


function maybewrite(fname, s)
    s_current = isfile(fname) ? read(fname, String) : nothing
    if s != s_current
        println("Writing '$fname'...")
        write(fname, s)
    else
        #println("Skipping '$fname'...")
    end
end


for nmesh in [256, 512]

    # realspace
    for sim_vox in [1,2], est_grid_assignment in [1,2]
        for fxshift_sim in [0.0, 0.5]
            s = config_head
            s = replace(s, r"n_sim: .*" => "n_sim: $nmesh")
            s = replace(s, r"n_est: .*" => "n_est: $nmesh")
            s = replace(s, r"f: .*" => "f: 0.0")
            s = replace(s, r"sim_vox: .*" => "sim_vox: $sim_vox")
            s = replace(s, r"est_grid_assignment: .*" => "est_grid_assignment: $est_grid_assignment")
            s = replace(s, r"fxshift_sim: .*" => "fxshift_sim: $fxshift_sim")

            sim_assign = sim_vox == 1 ? "NGP" : "CIC"
            est_assign = est_grid_assignment == 1 ? "NGP" : "CIC"
            fname = dname * "/pkest_realspace_nmesh$(nmesh)_sim$(sim_assign)_est$(est_assign)_fxshift$(fxshift_sim).yml"
            maybewrite(fname, s)

            # corrected
            s = replace(s, r"est_vox: .*" => "est_vox: 3")
            fname = replace(fname, r".yml" => "_p3.yml")
            maybewrite(fname, s)
        end
    end

    # redshift space
    for sim_velo in [1,2,3,4,5,6]
        for fxshift_sim in [0.0, 0.5]
            s = config_head
            s = replace(s, r"n_sim: .*" => "n_sim: $nmesh")
            s = replace(s, r"n_est: .*" => "n_est: $nmesh")
            s = replace(s, r"sim_vox: .*" => "sim_vox: 2")
            s = replace(s, r"sim_velo: .*" => "sim_velo: $sim_velo")
            s = replace(s, r"fxshift_sim: .*" => "fxshift_sim: $fxshift_sim")

            if sim_velo in [5,6]
                #s = replace(s, r"nrlzs: .*" => "nrlzs: 1000")
            end

            fname = dname * "/pkest_redshiftspace_nmesh$(nmesh)_simCIC_estNGP_velo$(sim_velo)_fxshift$(fxshift_sim).yml"
            maybewrite(fname, s)

            # corrected
            s = replace(s, r"est_vox: .*" => "est_vox: 3")
            fname = replace(fname, r".yml" => "_p3.yml")
            maybewrite(fname, s)
        end
    end

end
