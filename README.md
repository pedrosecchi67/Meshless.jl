# Meshless.jl

Package for meshless PDE solvers using Julia.

## Background mesh generation

Meshless solver point clouds are built based on "backgroud" octree meshes. These meshes may be built with:

```julia
function FixedMesh(
        origin::Vector{Float64}, widths::Vector{Float64},
        surfaces::Tuple{String, Stereolitography, Float64}...;
        refinement_regions::AbstractVector = [],
        growth_ratio::Float64 = 1.2,
        max_length::Float64 = Inf,
        ghost_layer_ratio::Float64 = -2.2,
        interior_point = nothing,
        approximation_ratio::Float64 = 5.0,
        filter_triangles_every::Int64 = 0,
        verbose::Bool = false,
        farfield_boundaries = nothing,
)
```

Generate an octree/quadtree mesh described by:

* A hypercube origin;
* A vector of hypercube widths;
* A set of tuples in format `(name, surface, max_length)` describing
    stereolitography surfaces (`Mesher.Stereolitography`) and 
    the max. cell widths at these surfaces;
* A set of refinement regions described by distance functions and
    the local refinement at each region. Example:

```julia
refinement_regions = [
    Mesher.Ball([0.0, 0.0], 0.1) => 0.005,
    Mesher.Ball([1.0, 0.0], 0.1) => 0.005,
    Mesher.Box([-1.0, -1.0], [3.0, 2.0]) => 0.0025,
    Mesher.Line([1.0, 0.0], [2.0, 0.0]) => 0.005
]
```

* A cell growth ratio;
* A maximum cell size (optional);
* A ratio between the cell circumradius and the SDF threshold past which
    cells are considered to be out of the domain. `ghost_layer_ratio = -2.0` 
    guarantees that a layer of at least two ghost cell layers are included 
    within each solid;
* A point reference within the domain. If absent, external flow is assumed;
* An approximation ratio between wall distance and cell circumradius past which
    distance functions are approximated;
* A number of recursive refinement levels past which the triangles in the provided
    triangulations are filtered to lighter, local topologies.

Farfield boundaries may be defined with the following syntax:

```julia
farfield_boundaries = [
    "inlet" => [
        (1, false), # fwd face, first dimension (x)
        (2, false), # left face, second dimension (y)
        (2, true), # right face, second dimension (y)
        (3, false), # bottom face, third dimension (z)
        (3, true), # top face, third dimension (z)
    ],
    "outlet" => [(1, true)]
]
```

Meshes may be exported to .vtk format by overloading WriteVTK.jl methods:

```julia
using Meshless
using WriteVTK

msh = FixedMesh(
    # ...
)

vtk = vtk_grid("file", msh)
vtk_save(msh)
```

## Graph construction

Point graphs may be built from a background mesh with:

```julia
function Graph(
    msh::Mesh;
    boundary_layers = [],
    n_per_quadrant::Int = 1, # number of connected neighbors per quadrant/octant
    order::Int = 2,
    verbose::Bool = false,
)
```

Build a graph from a mesh.

Boundary layers must be specified as boundary name/`BoundaryLayer` pairs:

```julia
boundary_layers = [
    "wall" => BoundaryLayer(1e-4, 1.1),
    "engine-inlet" => BoundaryLayer(1e-4, 1.1), # first height, growth ratio
]
```

Graphs may also be exported to .vtk format by overloading WriteVTK.jl methods:

```julia
using Meshless
using WriteVTK

graph = Graph(msh)

vtk = vtk_grid("file", graph)
vtk_save(msh)
```

## Domain construction and PDE solution

A graph can be converted to a PDE domain using:

```julia
function Domain(
    graph::Graph{T, Ti};
    partition_size::Int = 1000_000,
    degree::Int = 2,
    verbose::Bool = false,
) where {T <: Real, Ti <: Integer}
```

The p-METIS algorithm is ran to partition the graph into chunks of at most size `partition_size`. Polynomials of `degree`-th degree, meanwhile, are used to define differentiation operators.

The basic function for PDE residual calculation is:

```julia
using Meshless

domain = Domain(graph)

u = rand(length(domain)) # instantiate array
A = rand(length(domain), 3) # multi-dimensional array.
# first dimension identifies cell index

return_values = domain(u, A) do partition, u, A
    # calculate residuals, edit u, A in place

    r # optional (scalar) return value
end # return_values allocates return for each partition
```

The function also accepts kwargs, which are passed as they are to the called function.

We may also convert each partition to a given array backend (e. g. CUDA.jl) to run GPU-accelerated solutions. This is done via:

```julia
using CUDA

domain(u;
    conv_to_backend = x -> CuArray(x),
    conv_from_backend = x -> Array(x)) do part, u
    # run calculations
end
```

Note that passed array values and partition info are converted to the backend one partition at a time so as to allow for large, GPU-accelerated simulations without tight memory boundaries.

### Differentiation operators

The differentiation of a field variable may be done using:

```julia
domain(u, ux, uyy) do part, u, ux, uyy
    ux .= part(u, 1, 0, 0) # first derivative along first axis
    uyy .= part(u, 0, 2, 0) # second derivative along second axis
end
```

### Boundary conditions

The imposition of boundary conditions may be done using function `impose_bc!`. It is based on **boundary points** and **pivot points**.

A boundary point is located directly on the boundary, and it is the projection of the pivot point on said boundary. To impose a BC, we must define the value of a variable at the boundary points, based on its value at the pivot points. 

An example for the non-penetration condition is:

```julia
impose_bc!(partition, "boundary_name", u, v) do bdry, u, v
    nx, ny = bdry.normals |> eachcol

    un = @. nx * u + ny * v

    (
        u .- un .* nx,
        v .- un .* ny
    )
end
```

Optionally, the BC function may also return fewer output than input arrays, case in which only the first input arrays are edited in-place to impose BCs.

Also, the called function may return only a single array if only one field variable is to be edited.

Kwargs are also supported, and passed as-is to the called function.

Fields which can be used to calculate variable boundary values include:

```julia
bdry.points # matrix, (npts, ndims)
bdry.distances
bdry.normals # matrix, (npts, ndims)
```

### CFD utilities

Some functions are provided as utilities for CFD residual calculations.

Check the docstrings for:

```julia
using Meshless: CFD

?CFD.Fluid
?CFD.speed_of_sound
?CFD.state2primitive
?CFD.primitive2state
?CFD.rms
?CFD.HLL
?CFD.AUSM
?CFD.JSTKE
?CFD.pressure_coefficient
?CFD.Freestream
?CFD.initial_guess
?CFD.rotate_and_rescale!
``` 

Other, more important functions include:

```julia
function artificial_dissipation(
    part::Partition{N}, u::AbstractArray, λ::AbstractVector,
    dim::Int = 0
) where {N}
```

Obtain JST-KE type artificial dissipation using a given spectral radius, along dimension `dim`. The dissipation is summed along all dimensions if `dim = 0`.

```julia
function timescale(part::Partition{N}, λ::AbstractVecOrMat) where {N}
```

Obtain timescale (local time-step for CFL = 1) for advective equations given one or more spectral radii for the flux Jacobian (vector or matrix, with each column as a component).

```julia
smoothing(part::Partition{N}, u::AbstractArray) where {N}
```

Perform Laplacian smoothing iteration at partition.
Eq. to `part.smoothing(u)`

## Postprocessing

### Surface data

One may create a `Surface` struct based on a stereolitography object:

```julia
function Surface(
    domain::Domain, 
    stl::Stereolitography; 
    max_length::Float64 = 0.0, # if non-zero, tris are split until they meet max. size
    from_surface::Bool = true, # interpolate property only from boundary points?
)
```

One may, to obtain the forces on the surface, interpolate properties (in this case, `Cp`) from field variables to the surface:

```julia
surf = Surface(domain, stl)

Cp_surf = surf(Cp)
```

One may, then, integrate the forces along the surface:

```julia
CX = surface_integral(
    surf, - Cp_surf .* surf.normals[:, 1]
)
CY = surface_integral(
    surf, - Cp_surf .* surf.normals[:, 2]
)
```

Field `surf.points` is also available with surface point positions.

Surface data may also be exported to .vtk format:

```julia
vtk = vtk_grid("file", surf; 
    u = u, v = v) # field var. data as kwargs
vtk_save(vtk)
```

### Background volume grid

To export volume data, one may interpolate it to the background, octree grid:

```julia
intp = Interpolator(domain, mesh)

vtk = vtk_grid("file", mesh;
    u = intp(u), v = intp(v)) # interpolating field variables
vtk_save(vtk)
```

### Other probe points

A matrix of probe points `X` may have the values of field variables interpolated via an `Interpolator` struct. An example is:

```julia
X = rand(100, 3)
intp = Interpolator(domain, X)

u_at_X = intp(u)
```