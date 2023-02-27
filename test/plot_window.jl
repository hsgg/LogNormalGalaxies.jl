using PyPlot


k = 10.0 .^ (-3:0.1:0)
Δx = 1e3 / 64

Wmesh = @. sinc(k * Δx / (2 * π))

figure()
plot(k, Wmesh .^ 2)
