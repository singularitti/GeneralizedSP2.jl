using GeneralizedSP2
using Documenter

DocMeta.setdocmeta!(GeneralizedSP2, :DocTestSetup, :(using GeneralizedSP2); recursive=true)

makedocs(;
    modules=[GeneralizedSP2],
    authors="singularitti <singularitti@outlook.com> and contributors",
    sitename="GeneralizedSP2.jl",
    format=Documenter.HTML(;
        canonical="https://singularitti.github.io/GeneralizedSP2.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/singularitti/GeneralizedSP2.jl",
    devbranch="main",
)
