module Meshless

    include("graphgen.jl")
    using .GraphGen    

    export Stereolitography, refine_to_length, merge_points,
        Box, Ball, Line, DistanceField,
        feature_regions,
        vtk_grid, vtk_save,
        Graph, multigrid, partition,
        Domain, PartitionedDomain,
        at_boundary, at_pivots,
        to_backend,
        at_owners, at_neighbors, at_faces,
        reduce_faces, sum_faces, green_gauss, divergent,
        gradient, face_gradient,
        JST_sensor, MUSCL,
        Interpolator, BoundaryInterpolator, coarseners_and_prolongators,
        Surface, surface_integral,
        export_surfaces

    using .GraphGen.LinearAlgebra
    using .GraphGen.NearestNeighbors

    using .GraphGen.WriteVTK

    using .GraphGen.DocStringExtensions
    using .GraphGen.ProgressBars

    include("nninterp.jl")
    using .NNInterpolator
    using .NNInterpolator.ArrayAccumulator

    include("arraybends.jl")
    using .ArrayBackends

    include("stash.jl")
    using .Stash
    using .Stash.UUIDs
    using .Stash.Distributed

    include("cfd.jl")
    using .CFD
    include("turbulence.jl")
    using .Turbulence
    include("solver.jl")
    using .Solver

    @declare_converter NNInterpolator.ArrayAccumulator.Accumulator
    @declare_converter FlowBC

    """
    $TYPEDSIGNATURES

    Interpolate values from the nodes in a graph to any set of points
    (matrix `X`, one row per point).
    Uses Inverse Distance Weighing if `linear = false` (def. is `true`).

    Returns a callable object which, given values for scalar, vector or tensor
    field properties at all nodes in the graph (first index identifies node),
    returns the interpolated values at `X`.
    """
    NNInterpolator.Interpolator(
        graph::Graph{Tf, Ti}, X::AbstractMatrix;
        linear::Bool = true,
    ) where {Tf, Ti} = let intp = Interpolator(
        graph.points |> permutedims,
        X; linear = linear, first_index = true,
        k = size(X, 2) + 1,
    )
        ArrayAccumulator.change_data_types!(intp, Ti, Tf)
        intp
    end

    """
    $TYPEDSIGNATURES

    Interpolate values between graphs
    Uses Inverse Distance Weighing if `linear = false` (def. is `true`).

    Returns a callable object which, given values for scalar, vector or tensor
    field properties at all nodes in the graph (first index identifies node),
    returns the interpolated values at `X`.
    """
    NNInterpolator.Interpolator(
        src::Graph, dst::Graph;
        linear::Bool = true,
    ) = Interpolator(src, dst.points'; linear = linear)

    """
    $TYPEDSIGNATURES

    Interpolate values from the nodes in a graph to any set of points
    (matrix `X`, one row per point). Uses only boundary points for interpolation.
    Uses Inverse Distance Weighing if `linear = false` (def. is `true`).

    Returns a callable object which, given values for scalar, vector or tensor
    field properties at all nodes in the graph (first index identifies node),
    returns the interpolated values at `X`.
    """
    function BoundaryInterpolator(
        graph::Graph{Tf, Ti}, X::AbstractMatrix;
        linear::Bool = true,
    ) where {Tf, Ti}
        nd = size(X, 2)
        bpoints = reduce(vcat, values(graph.boundary_points))
        bpoint_map = Dict(
            [
                k => Int64(bp) for (k, bp) in enumerate(bpoints)
            ]
        )

        intp = Interpolator(
            graph.points[:, bpoints] |> permutedims,
            X; first_index = true, linear = linear,
            k = nd + 1,
        )

        NNInterpolator.re_index!(intp, bpoint_map)
        ArrayAccumulator.change_data_types!(intp, Ti, Tf)
        intp
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator from graph to underlying octree points
    """
    function NNInterpolator.Interpolator(
        graph::Graph, octree::GraphGen.BlockMesher.Mesh;
        linear::Bool = false,
    )
        centers, _, _ = GraphGen.BlockMesher.get_cells(octree)
        centers = permutedims(centers)

        Interpolator(graph, centers; linear = linear)
    end

    """
    $TYPEDSIGNATURES

    Obtain virtual facet information from a point cloud stencil
    (points in rows of `X`) around a pivot point (`x`).
    """
    function virtual_facets(
        X::AbstractMatrix{Tf}, x::AbstractVector{Tf};
        tolerance::Real = 1f-7,
    ) where {Tf <: AbstractFloat}
        dX = Float64.(X .- x')
        w = 1.0 ./ (
            sum(
                dX .^ 2; dims = 2
            ) .+ tolerance
        ) |> vec

        A = [dX ones(size(X, 1))]

        normals = Tf.((pinv(A .* w) .* w')[1:length(x), :]) |> permutedims
        areas = sqrt.(sum(normals .^ 2; dims = 2)) .* 2 |> vec
        normals ./= (areas ./ 2 .+ eps(Tf))
        distances = 1.0f0 ./ (areas .+ eps(Tf))

        (
            normals = normals,
            areas = areas,
            normal_distances = distances,
        )
    end

    """
    $TYPEDFIELDS

    Struct to define a boundary
    """
    struct Boundary{Tf <: AbstractFloat, Ti <: Integer}
        pivot_points::AbstractVector{Ti}
        boundary_points::AbstractVector{Ti}
        normals::AbstractMatrix{Tf}
        distances::AbstractVector{Tf}
    end

    @declare_converter Boundary

    """
    $TYPEDSIGNATURES

    Obtain view to boundary points for a boundary object. 
    Assumes first index of array identifies node number.
    
    Example for boundary conditions:

    ```
    # no penetration
    uv = zeros(length(dom), 2)

    bdry = dom.boundaries["wall"]

    uv_pivots = at_pivots(bdry, uv) |> copy
    uv_normal = sum(uv_pivots .* bdry.normals; dims = 2) |> vec

    at_boundary(bdry, uv) .= uv_normal .* bdry.normals

    # Neumann
    u = zeros(length(dom))
    ∂u!∂n = 1.0

    at_boundary(bdry, u) .= at_pivots(bdry, u) .- bdry.distances .* ∂u!∂n
    ```
    """
    at_boundary(bdry::Boundary{Tf, Ti}, u::AbstractArray) where {Tf, Ti} = selectdim(
        u, 1, bdry.boundary_points
    )

    """
    $TYPEDSIGNATURES

    Obtain view to pivot points for a boundary object. 
    Assumes first index of array identifies node number.
    
    Example for boundary conditions:

    ```
    # no penetration
    uv = zeros(length(dom), 2)

    bdry = dom.boundaries["wall"]

    uv_pivots = at_pivots(bdry, uv) |> copy
    uv_normal = sum(uv_pivots .* bdry.normals; dims = 2) |> vec

    at_boundary(bdry, uv) .= uv_normal .* bdry.normals

    # Neumann
    u = zeros(length(dom))
    ∂u!∂n = 1.0

    at_boundary(bdry, u) .= at_pivots(bdry, u) .- bdry.distances .* ∂u!∂n
    ```
    """
    at_pivots(bdry::Boundary{Tf, Ti}, u::AbstractArray) where {Tf, Ti} = selectdim(
        u, 1, bdry.pivot_points
    )

    """
    $TYPEDFIELDS

    Struct to define a surface for post-processing.
    `offsets` define the offset between property sampling points
    and the surface.
    """
    struct Surface{Tf <: AbstractFloat, Ti <: Integer}
        points::AbstractMatrix{Tf}
        normals::AbstractMatrix{Tf}
        areas::AbstractVector{Tf}
        interpolator::NNInterpolator.Accumulator
        stl::Stereolitography
    end

    @declare_converter Stereolitography
    @declare_converter Surface{Float32, Int32}
    @declare_converter Surface{Float64, Int32}
    @declare_converter Surface{Float32, Int64}
    @declare_converter Surface{Float64, Int64}

    """
    $TYPEDSIGNATURES

    Interpolate field data (firs index for graph node) to surface facet centers
    """
    (surf::Surface)(u::AbstractArray) = surf.interpolator(u)

    """
    $TYPEDSIGNATURES

    integrate a property throughout a surface
    """
    surface_integral(surf::Surface, u::AbstractVector) = (surf.areas .* u |> sum)

    """
    $TYPEDSIGNATURES

    integrate a property throughout a surface. The first dimension in the array
    is assumed to refer to point/cell indices
    """
    surface_integral(surf::Surface, u::AbstractArray) = (
        surf.areas .* u |> a -> sum(a; dims = 1) |> a -> dropdims(a; dims = 1)
    )

    """
    Abstract type to define a domain, either partitioned or not.
    """
    abstract type AbstractDomain{Tf <: AbstractFloat, Ti <: Integer}
    end

    """
    $TYPEDFIELDS

    Struct to define a domain.
    """
    struct Domain{Tf <: AbstractFloat, Ti <: Integer} <: AbstractDomain{Tf, Ti}
        partition_index::Int
        face_accumulator::Accumulator
        points::AbstractMatrix{Tf}
        face_owners::AbstractVector{Ti}
        face_neighbors::AbstractVector{Ti}
        face_normals::AbstractMatrix{Tf}
        face_areas::AbstractVector{Tf}
        face_distances::AbstractVector{Tf}
        boundaries::Dict{String, Boundary{Tf, Ti}}
        surfaces::Dict{String, Surface{Tf, Ti}}
    end

    """
    $TYPEDSIGNATURES

    Obtain number of points in domain
    """
    Base.length(dom::Domain) = size(dom.points, 1)

    """
    $TYPEDSIGNATURES

    Obtain dimensionality of a domain
    """
    Base.ndims(dom::Domain) = size(dom.points, 2)

    """
    $TYPEDSIGNATURES

    Get a domain from a graph
    """
    function Domain(
        graph::Graph{Tf, Ti};
        tolerance::Real = 1f-7,
        verbose::Bool = false,
        partition_index::Int = 0,
    ) where {Tf, Ti}
        points = graph.points |> permutedims |> copy
        verbose && println("Building domain from graph with $(size(points, 1)) points...")

        face_owners = map(
            i -> fill(Ti(i), length(graph.neighbors[i])),
            1:size(points, 1)
        ) |> x -> reduce(vcat, x)
        face_neighbors = reduce(vcat, graph.neighbors)

        n0 = Ti(0)
        acc = map(
            neighs -> let rng = (n0 + 1):(n0 + length(neighs))
                stencil = Ti.(rng)
                n0 += length(stencil)

                stencil
            end, graph.neighbors
        ) |> x -> Accumulator(x; first_index = true)

        ArrayAccumulator.change_data_types!(acc, Ti)

        face_normals, face_distances, face_areas = let tups = map(
            i -> let neighs = graph.neighbors[i]
                X = @view points[neighs, :]
                x = points[i, :]

                virtual_facets(X, x; tolerance = tolerance)
            end, 1:size(points, 1)
        )
            (
                map(t -> t.normals, tups) |> x -> reduce(vcat, x),
                map(t -> t.normal_distances, tups) |> x -> reduce(vcat, x),
                map(t -> t.areas, tups) |> x -> reduce(vcat, x),
            )
        end

        t0 = time()
        verbose && println("Detecting boundaries...")

        boundaries = Dict{String, Boundary{Tf, Ti}}()
        for k in keys(graph.boundary_points)
            bpoints = graph.boundary_points[k] |> copy
            ppoints = graph.pivot_points[k] |> copy

            normals = points[ppoints, :] .- points[bpoints, :]

            distances = sum(normals .^ 2; dims = 2) |> vec
            @. distances = sqrt(distances) + eps(Tf)
            normals ./= distances

            boundaries[k] = Boundary(
                ppoints, bpoints,
                normals, distances
            )
        end

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        t0 = time()
        verbose && println("Building surface information...")

        surfaces = Dict{String, Surface{Tf, Ti}}()
        for (sname, stl) in graph.surfaces
            centers, normals = GraphGen.BlockMesher.centers_and_normals(stl)
            centers = permutedims(centers)
            normals = permutedims(normals)

            areas = sum(normals .^ 2; dims = 2) |> vec
            @. areas = sqrt(areas) + eps(Tf)
            normals ./= areas

            surfaces[sname] = Surface{Tf, Ti}(
                centers, normals, areas,
                BoundaryInterpolator(graph, centers; linear = true,),
                stl
            )
        end

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")
        verbose && println("Done with domain construction.")

        Domain{Tf, Ti}(
            partition_index,
            acc, points,
            face_owners, face_neighbors,
            face_normals, face_areas, face_distances,
            boundaries, surfaces,
        )
    end

    """
    $TYPEDSIGNATURES

    Reduce virtual face values over a point's neighboring edges,
    returning reduced point properties.
    """
    reduce_faces(
        op, dom::Domain, uf::AbstractArray
    ) = dom.face_accumulator(uf; op = op)

    """
    $TYPEDSIGNATURES

    Sum face property over a point's neighboring edges.
    """
    sum_faces(
        dom::Domain, uf::AbstractArray
    ) = reduce_faces(+, dom, uf)

    """
    $TYPEDSIGNATURES

    Run Green-Gauss integration over virtual faces
    (`sum_faces` on `uf .* dom.face_areas`).
    """
    green_gauss(
        dom::Domain, uf::AbstractArray
    ) = sum_faces(dom, uf .* dom.face_areas)

    """
    $TYPEDSIGNATURES

    Obtain view to property at virtual face owners.
    The first array index is interpreted as the node index.
    """
    at_owners(dom::Domain, u::AbstractArray) = selectdim(
        u, 1, dom.face_owners
    )

    """
    $TYPEDSIGNATURES

    Obtain view to property at virtual face neighbors.
    The first array index is interpreted as the node index.
    """
    at_neighbors(dom::Domain, u::AbstractArray) = selectdim(
        u, 1, dom.face_neighbors
    )

    """
    $TYPEDSIGNATURES

    Obtain values of properties at virtual faces (edge midpoints)
    from array of field properties, the first array index identifying
    each node.
    """
    at_faces(
        dom::Domain, u::AbstractArray
    ) = (
        at_owners(dom, u) .+ at_neighbors(dom, u)
    ) ./ 2

    """
    $TYPEDSIGNATURES

    Obtain (virtual facet) Green-Gauss gradient, given nodal,
    field property values of `u` (first index identifies graph node).
    Returns a tuple with values along each dimension if `dim == 0`.
    """
    function gradient(
        dom::Domain, u::AbstractArray, dim::Int = 0
    )
        if dim == 0
            return tuple(
                map(
                    d -> gradient(dom, u, d), 1:ndims(dom)
                )...
            )
        end

        n = @view dom.face_normals[:, dim]
        green_gauss(dom, at_faces(dom, u) .* n)
    end

    """
    $TYPEDSIGNATURES

    Obtain divergent out of a vector field. Each arg should be the component
    along one of the Cartesian axes, specified at virtual faces.
    """
    function divergent(
        dom::Domain, uf::AbstractVector...
    )
        ϕ = first(uf) |> similar
        ϕ .= 0

        for (n, u) in zip(
            eachcol(dom.face_normals), uf
        )
            ϕ .+= n .* u
        end

        green_gauss(dom, ϕ)
    end

    """
    $TYPEDSIGNATURES

    From values at nodes and (optionally)
    gradients at nodes along each Cartesian axis,
    obtain gradients at virtual faces.
    """
    function face_gradient(
        dom::Domain, u::AbstractArray,
        grad_u::Union{Tuple, Nothing} = nothing,
    )
        if isnothing(grad_u)
            grad_u = gradient(dom, u)
        end

        grad_u = map(g -> at_faces(dom, g), grad_u)

        ns = dom.face_normals |> eachcol
        d = dom.face_distances

        normal_gradient = first(grad_u) |> similar
        normal_gradient .= 0
        for (n, g) in zip(ns, grad_u)
            @. normal_gradient += n * g
        end

        new_normal_gradient = (
            at_neighbors(dom, u) .- at_owners(dom, u)
        ) ./ d
        for (n, g) in zip(ns, grad_u)
            @. g += (new_normal_gradient - normal_gradient) * n
        end

        grad_u
    end

    """
    $TYPEDSIGNATURES

    JST sensor.
    """
    function JST_sensor(
        dom::Domain, u::AbstractArray{Tf}
    ) where {
        Tf <: AbstractFloat
    }
        δu = at_neighbors(dom, u) .- at_owners(dom, u)

        (
            abs.(green_gauss(dom, δu)) .+ eps(Tf)
        ) ./ (
            green_gauss(dom, abs.(δu)) .+ eps(Tf)
        )
    end

    """
    Minmod operator
    """
    @inline minmod(u1::Real, u2::Real) = min(abs(u1), abs(u2)) * (sign(u1) + sign(u2)) / 2
    """
    Minmod operator over face
    """
    @inline face_minmod(uL, δuL, uR, δuR) = let grad = @. minmod(
        2 * δuL - (uR - uL), 2 * δuR - (uR - uL)
    )
        (
            (uL .+ grad ./ 2),
            (uR .- grad ./ 2)
        )
    end

    """
    $TYPEDSIGNATURES

    MUSCL reconstruction with minmod limiter.
    `u` must be provided at cells. Re-calculates cell gradients if not provided.

    Ducros sensor `D` may be provided at each cell: Value `0` switches to a centered
    scheme, while value `1` switches to minmod.
    """
    function MUSCL(
        dom::Domain, u::AbstractArray,
        ∇u::Union{Nothing, Tuple} = nothing;
        D::Union{Real, AbstractArray} = 1.0f0,
    )
        D = (
            D isa Real ?
            D :
            max.(at_owners(dom, D), at_neighbors(dom, D))
        )

        uo = at_owners(dom, u)
        un = at_neighbors(dom, u)

        uL, uR = begin
            if isnothing(∇u)
                ∇u = gradient(dom, u)
            end

            δuo = similar(uo)
            δun = similar(un)
            δuo .= 0
            δun .= 0

            po = at_owners(dom, dom.points)
            pn = at_neighbors(dom, dom.points)

            for (dx, g) in zip(eachcol(pn .- po), ∇u)
                δuo .+= dx .* at_owners(dom, g)
                δun .+= dx .* at_neighbors(dom, g)
            end

            face_minmod(uo, δuo, un, δun)
        end

        @. uL = uL * D + (uo + un) / 2 * (1.0f0 - D)
        @. uR = uR * D + (uo + un) / 2 * (1.0f0 - D)

        (uL, uR)
    end

    """
    $TYPEDSIGNATURES

    Export surfaces to folder `fname` with paraview multiblock files.
    Erases folder if already existent.
    Args may be used to specify which surfaces are to be exported.

    Kwargs are exported as field data after interpolation to surfaces.
    """
    function export_surfaces(
        fname::String, dom::AbstractDomain, snames::String...;
        kwargs...
    )
        if length(snames) == 0
            snames = tuple(keys(dom.surfaces)...)
        end

        if isdir(fname)
            @warn "Overwriting surface output in folder $fname."
            rm(fname; recursive = true, force = true)
        end
        mkdir(fname)

        vtm = joinpath(fname, "SURFACES") |> vtk_multiblock
        for sname in snames
            surf = dom.surfaces[sname]

            grid = vtk_grid(
                joinpath(fname, sname),
                surf.stl, vtm
            )
            for (k, v) in kwargs
                grid[String(k)] = surf(v)
            end
        end

        vtk_save(vtm)
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolators to move properties
    from coarse to fine multigrid levels, and vice versa.

    Returns arrays of callable, `Interpolator` structs such that:

    ```
    coarseners, prolongators = coarseners_and_prolongators(graphs...)

    A = rand(npts, ndims) # fine grid field property array

    Ac = coarseners[1](A) # to first coarse level
    @assert A ≈ prolongators[1](Ac) # ...and back

    # coarseners[i] coarsens to level i + 1
    # prolongators[i + 1] prolongs back to level i
    ```
    """
    function coarseners_and_prolongators(
        graphs::Graph...
    )
        graph = graphs[1]
        coarseners = NNInterpolator.ArrayAccumulator.Accumulator[]
        prolongators = NNInterpolator.ArrayAccumulator.Accumulator[]

        for i = 2:length(graphs)
            coars = Interpolator(
                graph, graphs[i]; linear = false,
            )
            prolong = Interpolator(
                graphs[i], graph; linear = false,
            )

            graph = graphs[i]

            push!(coarseners, coars)
            push!(prolongators, prolong)

            Base.GC.gc()
        end

        (coarseners, prolongators)
    end

    """
    $TYPEDFIELDS

    Struct to define a partitioned domain
    """
    mutable struct PartitionedDomain{Tf <: AbstractFloat, Ti <: Integer} <: AbstractDomain{Tf, Ti}
        n_nodes::Int64
        worker_partitions::Dict{Int64, Vector{Tuple{Int64, UUID, Vector{Ti}, Int64}}}
        surfaces::Dict{String, Surface{Tf, Ti}}
        lazy_backend::Bool
        conv_to_backend
        conv_from_backend
    end

    function _unstash_partitions!(pdom::PartitionedDomain)
        for (pid, tups) in pdom.worker_partitions
            try
                for (_, key, _, _) in tups
                    clean_stash!(pid, key)
                end
            catch ProcessExitedException
                # pass
            end
        end
    end

    """
    $TYPEDSIGNATURES

    Construct a partitioned domain across MPI processes.
    If `workers` is not provided, all partitions are allocated to the
    current process.

    If `conv_to_backend` and `conv_from_backend` conversors are defined,
    conversion of data to custom array backends is carried out upon each process,
    or at residual evaluation if `lazy_backend = true`.
    """
    function PartitionedDomain(
        graph::Graph{Tf, Ti};
        max_partition_size::Int = 100_000,
        workers::Union{Nothing, AbstractVector} = nothing,
        skirt_order::Int = 1,
        lazy_backend::Bool = false,
        conv_to_backend = identity,
        conv_from_backend = identity,
        verbose::Bool = false,
    ) where {Tf, Ti}
        verbose && println("Building partitioned domain from graph with $(size(graph.points, 2)) nodes...")
        t0 = time()

        if isnothing(workers)
            workers = [myid()]
        end

        worker_partitions = Dict{Int64, Vector{Tuple{Int64, UUID, Vector{Ti}, Int64}}}()

        verbose && println("Running METIS...")
        t0 = time()
        let partitions = partition(graph, max_partition_size; skirt_order = skirt_order)
            verbose && println("[DONE] - $(time() - t0) seconds elapsed")

            # distribute partitions among workers
            _workers = repeat(1:length(workers); 
                inner = length(partitions) / length(workers) |> ceil |> Int64)
            iworker = ipart -> _workers[ipart]

            # list batch of subgraphs for each process
            batches = [Tuple{Int64, Graph{Tf, Ti}}[] for _ = 1:length(workers)]
            for ipart = 1:length(partitions)
                subgraph, _, _ = partitions[ipart]

                ipid = iworker(ipart)

                push!(
                    batches[ipid],
                    (ipart, subgraph)
                )
            end

            # launch and collect futures
            futures = []
            for (ipid, subgraphs) in enumerate(batches)
                pid = workers[ipid]
                future = @spawnat pid begin
                    domains = map(
                        sub -> Domain(sub[2]; partition_index = sub[1]), subgraphs
                    )
                    if !lazy_backend
                        domains = map(
                            dom -> to_backend(dom, conv_to_backend),
                            domains
                        )
                    end

                    [
                        stash!(myid(), dom) for dom in domains
                    ]
                end

                push!(futures, future)
            end

            keys = fetch.(futures)

            # store keys to dict
            for ipart = 1:length(partitions)
                ipid = iworker(ipart)
                pid = workers[ipid]

                _, domain, n_skirt = partitions[ipart]
                key = popfirst!(keys[ipid])

                if !haskey(worker_partitions, pid)
                    worker_partitions[pid] = []
                end
                push!(
                    worker_partitions[pid],
                    (ipart, key, domain, n_skirt)
                )
            end
        end

        t0 = time()
        verbose && println("Building surface information...")

        surfaces = Dict{String, Surface{Tf, Ti}}()
        for (sname, stl) in graph.surfaces
            centers, normals = GraphGen.BlockMesher.centers_and_normals(stl)
            centers = permutedims(centers)
            normals = permutedims(normals)

            areas = sum(normals .^ 2; dims = 2) |> vec
            @. areas = sqrt(areas) + eps(Tf)
            normals ./= areas

            surf = Surface{Tf, Ti}(
                centers, normals, areas,
                BoundaryInterpolator(graph, centers; linear = true,),
                stl
            )

            if !lazy_backend
                surf = to_backend(surf, conv_to_backend)
            end

            surfaces[sname] = surf
        end

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")
        verbose && println("Done with domain construction.")

        pdom = PartitionedDomain(
            size(graph.points, 2),
            worker_partitions, surfaces,
            lazy_backend, conv_to_backend, conv_from_backend,
        )
        finalizer(_unstash_partitions!, pdom)

        pdom
    end

    """
    $TYPEDSIGNATURES

    Get number of cells in partitioned domain
    """
    Base.length(dom::PartitionedDomain) = dom.n_nodes

    """
    $TYPEDSIGNATURES

    Run a function on individual subdomains.
    Results are collected and returned in a vector,
    one entry per partitioned.
    kwargs are passed as they are.

    Example:

    ```
    r = dom(a, b) do a, b # at local subpartition domain
        # do something editing a, b in place;
        # r calculated

        r # pass return value
    end
    ```
    """
    function (pdom::PartitionedDomain)(f, args::AbstractArray...; kwargs...)
        tasks = []

        nparts = sum(length, values(pdom.worker_partitions))
        return_values = Vector{Any}(undef, nparts)

        for (pid, partitions) in pdom.worker_partitions
            t = @task begin
                for (ipart, key, domain, n_skirt) in partitions
                    myargs = map(
                        a -> selectdim(a, 1, domain) |> copy,
                        args
                    )
                    lazy_backend = pdom.lazy_backend
                    tobend = pdom.conv_to_backend
                    frombend = pdom.conv_from_backend

                    future = @spawnat pid begin
                        dom = unstash(myid(), key)
                        if lazy_backend
                            dom = to_backend(dom, tobend)
                        end
                        myargs = map(ma -> to_backend(ma, tobend), myargs)

                        r = f(dom, myargs...; kwargs...)
                        myargs = map(ma -> to_backend(ma, frombend), myargs)

                        (r, myargs)
                    end
                    t = fetch(future)

                    if t isa Distributed.RemoteException
                        throw(t)
                    end
                    r, myargs = t

                    for (ma, a) in zip(myargs, args)
                        indom = @view domain[1:(length(domain) - n_skirt)]
                        selectdim(a, 1, indom) .= selectdim(ma, 1, 1:(length(domain) - n_skirt))
                    end

                    return_values[ipart] = r
                end
            end
            push!(tasks, t)
        end
        schedule.(tasks)
        wait.(tasks)

        return_values
    end

end # module Meshless
