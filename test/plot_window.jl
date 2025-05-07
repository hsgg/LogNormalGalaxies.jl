# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


using PythonPlot


k = 10.0 .^ (-3:0.1:0)
Δx = 1e3 / 64

Wmesh = @. sinc(k * Δx / (2 * π))

figure()
plot(k, Wmesh .^ 2)
