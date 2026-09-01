using Distributed
using ProgressBars

workers = addprocs(3)

@everywhere using Meshless
using Meshless.Solver

include("advection.jl")
include("dissipation.jl")
