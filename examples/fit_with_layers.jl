using GeneralizedSP2
using Plots

PLOT_DEFAULTS = Dict(
    :size => (450, 450),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1.5,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 10,
    :guidefontsize => 10,
    :tickfontsize => 8,
    :legendfontsize => 7,
    :left_margin => (0, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

β = 9.423
μ = 0.568
minlayers = 4
maxlayers = 4
lower_bound, upper_bound = 0, 1
𝛆 = 0:0.01:1
𝝷 = hcat(
    [3.4199, -0.916353, 0.638295],
    [-0.877837, 4.54196, 1.50423],
    [0.111267, 0.40718, 0.644496],
    [-0.0703375, 2.35554, 0.981319],
)
𝐜 = [0.181909, 0.047729, -2.71051, 0.355542]'
𝝷 = vcat(𝝷, 𝐜)

target_fermi_dirac(ε) = 1 / (1 + exp(β * (ε - μ)))

plot()
xlims!(lower_bound, upper_bound)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
title!("Data from Kipton μ=$μ, β=$β")
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
plot!(𝛆, target_fermi_dirac.(𝛆); label="Reference Fermi function", PLOT_DEFAULTS...)
plot!(
    𝛆,
    iterate_fermi_dirac(𝛆, 𝝷);
    label="Approximated function with 4 layers",
    PLOT_DEFAULTS...,
)
savefig("Kipton_data.png")

plt = plot()
xlims!(lower_bound, upper_bound)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
title!("My fitted results μ=$μ, β=$β")
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
plot!(𝛆, target_fermi_dirac.(𝛆); label="Reference Fermi function", PLOT_DEFAULTS...)
branches = determine_branches(μ, maxlayers)
𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (lower_bound, upper_bound))
𝐲 = forward_pass(branches, 𝐱)
plot!(
    𝐱,
    oneunit.(𝐲) - 𝐲;
    label="SP2 best Approximated with $maxlayers layers",
    linestyle=:dash,
    PLOT_DEFAULTS...,
)
for n in minlayers:maxlayers
    𝝷FD, 𝝷ₛ = fit_model(𝐱, μ, β; nlayers=n)
    plot!(
        𝐱,
        iterate_fermi_dirac(𝐱, 𝝷FD);
        label="Approximated function with $n layers",
        linestyle=:dot,
        PLOT_DEFAULTS...,
    )
    # savefig("my_fits_beta=$β,n=$n.png")
end
for n in minlayers:maxlayers
    𝐱′ = chebyshevnodes_1st(length(𝐱), (lower_bound, upper_bound))
    𝝷FD, 𝝷ₛ = fit_model(𝐱′, μ, β; nlayers=n)
    plot!(
        𝐱′,
        iterate_fermi_dirac(𝐱′, 𝝷FD);
        label="Approximated function with $n layers using Chebyshev nodes",
        linestyle=:dashdot,
        PLOT_DEFAULTS...,
    )
    savefig("my_fits2_beta=$β,n=$n.png")
end

β = 20
plt = plot()
xlims!(lower_bound, upper_bound)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
title!("My fitted results μ=$μ, β=$β")
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
plot!(𝛆, target_fermi_dirac.(𝛆); label="Reference Fermi function", PLOT_DEFAULTS...)
branches = determine_branches(μ, maxlayers)
𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (lower_bound, upper_bound))
𝐲 = forward_pass(branches, 𝐱)
plot!(
    𝐱,
    oneunit.(𝐲) - 𝐲;
    label="SP2 best Approximated with $maxlayers layers",
    linestyle=:dash,
    PLOT_DEFAULTS...,
)
for n in minlayers:maxlayers
    𝝷FD, 𝝷ₛ = fit_model(𝐱, μ, β; nlayers=n)
    plot!(
        𝐱,
        iterate_fermi_dirac(𝐱, 𝝷FD);
        label="Approximated function with $n layers",
        linestyle=:dot,
        PLOT_DEFAULTS...,
    )
    # savefig("my_fits_beta=$β,n=$n.png")
end
for n in minlayers:maxlayers
    𝐱′ = chebyshevnodes_1st(length(𝐱), (lower_bound, upper_bound))
    𝝷FD, 𝝷ₛ = fit_model(𝐱′, μ, β; nlayers=n)
    plot!(
        𝐱′,
        iterate_fermi_dirac(𝐱′, 𝝷FD);
        label="Approximated function with $n layers using Chebyshev nodes",
        linestyle=:dashdot,
        PLOT_DEFAULTS...,
    )
    savefig("my_fits2_beta=$β,n=$n.png")
end
