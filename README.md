# Meshless.jl

Framework for numerical solutions of PDEs using the "virtual finite volume" method (VFVM).

## Theory

If a point `xi` has neighbors in a graph given by `xj`, then an approximation can be conducted for the "virtual" finite volume cell around `xi` if:

* The normal of the virtual faces is given by `2akj`, if `ak` are the weighed least-squares differentiation weights for point `xi` given values at `xj`, along direction `k`;
    - The normal is scaled such that its norm is equal to the area-over-volume ratio of the virtual face; and
* The distance between nodes `i` and `j` normal to each virtual face is given by `1/√(2akj)`.

Face centers are located at graph edge midpoints.

## Graph construction

```julia
    function Graph(
        origin::AbstractVector, widths::AbstractVector,
        surfaces::Tuple...;
        growth_ratio::Real = 1.2f0,
        tolerance::Real = 1f-7,
        interior_reference::AbstractVector,
        cutoff_ratio::Real = 2.1f0,
        refinement_regions::AbstractVector = [],
        boundary_layers::AbstractVector = [],
        hypercube_families::AbstractVector = [],
        verbose::Bool = false,
    )
```

Generate a graph given hypercube origins and widths.

Surfaces should be specified as tuples with family names, `Stereolitography` structs and local refinement levels, respectively.

Refinement regions, meanwhile, may be specified as tuples between distance functions (see `Line, Box, Ball, DistanceField` in this module) and local refinement levels.

An approximate growth rate is accepted for the block octree/quadtree.

Example:

```julia
stl = Stereolitography("wall.dat") # or STL in 3D
stl2 = Stereolitography("wall2.dat")

features = feature_regions(stl) |> DistanceField
region2 = Stereolitography("region.stl") |> DistanceField

graph = Graph(
    [-1.0, -1.0], [3.0, 3.0], # origin, widths
    ("wall", stl, 1e-3),
    ("wall2", stl2, 2e-3);
    growth_ratio = 1.2f0, # default
    refinement_regions = [
        features => 5e-4,
        region2 => 1e-2,
        Ball([0.0, 0.0], 1.0) => 2e-2,
        Box([-1.0, -1.0], [1.0, 2.0]) => 1e-2
    ],
)

# export to VTK:
using WriteVTK

vtk_grid("graph", graph) |> vtk_save
```

Point `interior_reference` is used to define the inside of the domain.
Ratio `cutoff_ratio` is used to determine how far outside of the boundaries
an octree cell must be to be considered within the domain, as a function of its
circumradius.

Boundary layer thicknesses must be specified for walls with anisotropic refinement
as per kwarg `boundary_layers`:

```julia
boundary_layers = [
    "family1" => 1e-4, # first height
    "family2" => 2e-4
]
```

Hypercube boundary families may be specified as in:

```julia
hypercube_families = [
    "inlet" => [
        (1, false), # x-axis, front
        (2, false), # y-axis, left
        (2, true), # y-axis, right
        (3, false), # z-axis, bottom
        (3, true), # z-axis, top
    ],
    "outlet" => [(1, true)]
]
```

## Domain construction

To evaluate residuals, one must convert a graph to a domain:

```julia
dom = Domain(graph; verbose = true)

# first index always identifies graph node:
uvw = rand(length(dom), ndims(dom))

# gradient at graph nodes
grad_uvw = gradient(dom, uvw)
dx, dy, dz = grad_uvw # tuple!

# divergent
flux = at_faces(dom, uvw) .* dom.face_normals |> x -> sum(x; dims = 2) |> vec

F = (flux .+ abs.(flux)) ./ 2 .* at_owners(dom, uvw) .+ (flux .- abs.(flux)) ./ 2 .* at_neighbors(dom, uvw)

div = green_gauss(dom, F)

# laplacian
ν = 1.0f0
laplacian = green_gauss(
    dom, ν .* (
        at_owners(dom, uvw) .- at_neighbors(dom, uvw)
    ) ./ dom.face_distances
)

# or:
grad_uvw_face = face_gradient(dom, uvw, grad_uvw)
dx, dy, dz = grad_uvw_face # tuple!

nx, ny, nz = dom.face_normals |> eachcol
laplacian = green_gauss(
    dom, (@. nx * dx + ny * dy + nz * dz)
)

# also, check out dom.face_areas and sum_faces(dom, u)

# MUSCL for interfaces at virtual faces:
uL, uR = MUSCL(u, grad_uvw) 
# check docstring for other args and kwargs

# imposing boundary conditions:
bdry = dom.boundaries["wall"]

# Dirichlet 0
at_boundary(bdry, uvw) .= 0

# Neumann
at_boundary(bdry, uvw) .= at_pivots(bdry, uvw) .- bdry.distances .* du!dy

# No penetration
u_normal = sum(bdry.normals .* at_pivots(bdry, uvw); dims = 2) |> vec
at_boundary(bdry, uvw) .= at_pivots(bdry, uvw) .- u_normal .* bdry.normals
```

## Postprocessing

```julia
using WriteVTK

# export graph to vtk:
vtk_grid("graph", graph) |> vtk_save

# export slice at plane to vtk:
vtk_grid(
    "slice",
    graph,
    [0.0, 1.0, 2.0], # origin
    [0.0, 1.0, 0.0] # widths
) |> vtk_save

# export underlying octree:
vtk_grid("tree", graph.octree) |> vtk_save

# interpolating to octree:
interpolator = Interpolator(graph, graph.octree)
# check out docstrings for interpolations
# to point clouds, triangulated surfaces, etc.

grid = vtk_grid("tree", graph.octree)
grid["u"] = interpolator(u)
vtk_save(grid)
```

You may interpolate field properties to surfaces (boundaries as identified by stereolitographies provided at construction):

```julia
surf = dom.surfaces["wall"]

u = rand(length(dom))
usurf = surf(u) # interpolation to facet centers

U = surface_integral(
    surf, usurf
)

# check out:
@show size(surf.normals) # (nfaces, ndims)
@show size(surf.areas) # (nfaces,)

# exporting all surfaces to folder with .vtm core, each family as a block:
export_surfaces(
    "folder", dom;
    u = u # kwargs are interpolated and passed as cell data
)
```

## Parallelism and partitioning

To run a domain partitioned with max. partition size of 100k cells:

```julia
pdom = PartitionedDomain(
    graph; max_partition_size = 100_000, # default
    skirt_order = 2, # number of DFS interations for neighborhood points of each partition
    verbose = true
)

# for residual calculation:
return_values = pdom(u, v) do subdomain, usub, vsub
    # do some calculations are edit local parts of input arrays in place
    # (returned to parent process)

    return_value # returned to parent process
end
```

For parallel execution over MPI:

```julia
pdom = PartitionedDomain(
    graph; verbose = true,
    workers = addprocs(4)
)
```

For execution with a custom backend array (e.g. `CuArray`):

```julia
to_backend = x -> CuArray(x)
from_backend = x -> Array(x)

pdom = ParitionedDomain(
    graph;
    conv_to_backend = to_backend,
    conv_from_backend = from_backend,
)
```

To port arrays to GPU only upon residual evaluation (increases communication time but allows for operation with small devices):

```julia
pdom = ParitionedDomain(
    graph;
    conv_to_backend = to_backend,
    conv_from_backend = from_backend,
    lazy_backend = true
)
```

Other data types often support function `obj = Meshless.to_backend(obj, conv_to_backend)`.

## CFD utilities

Check out the docstrings for the following functions and structs:

```julia
using Meshless.CFD

Fluid
speed_of_sound
dynamic_viscosity
heat_conductivity
primitive2state
state2primitive
FlowBC
ISA_atmosphere
streamwise_direction
pressure_coefficient
inviscid_fluxes
viscous_fluxes
Reynolds_number
adjust_Reynolds
TimeAverage

using Meshless.Turbulence

wall_function
shear_rate
Smagorinsky_νSGS
WALE_νSGS
Wray_Agarwal
standard_kϵ
shock_sensor
JST_sensor

using Meshless.Solver

multigrid
coarseners_and_prolongators
FAS!
```