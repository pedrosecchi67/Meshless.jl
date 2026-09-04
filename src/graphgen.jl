module GraphGen

    include("metis.jl")
    using .METIS

    include("mesher.jl")
    using .BlockMesher

    using .BlockMesher.DocStringExtensions

    using .BlockMesher.LinearAlgebra
    using .BlockMesher.NearestNeighbors

    using .BlockMesher.WriteVTK

    using ProgressBars

    using Base.Threads: @threads, ReentrantLock, lock

    export Stereolitography, refine_to_length, merge_points,
        Box, Ball, Line, DistanceField,
        feature_regions,
        vtk_grid, vtk_save,
        Graph, multigrid, partition

    """
    $TYPEDSIGNATURES

    Flood through a grid from a seed point and return mask of interior points.
    Also returns a second mask flagging points which should be projected onto the boundary.
    """
    function flood_fill(
        fields::Dict{String, DistanceField},
        centers::AbstractMatrix, widths::AbstractMatrix,
        interior_reference::AbstractVector;
        cutoff_ratio::Real = 2.0f0,
        verbose::Bool = false,
    )
        mask = falses(size(centers, 2))
        boundary = falses(size(centers, 2))

        tree = KDTree(centers)
        radii = sum(widths .^ 2; dims = 1) |> vec |> x -> sqrt.(x) ./ 2

        front = let i = argmin(
            sum((centers .- interior_reference) .^ 2; dims = 1) |> vec
        )
            Set([i])
        end

        dist = (field, pt, r) -> norm(
            pt .- BlockMesher.projection(field, pt, r)
        )

        verbose && println("Flooding $(size(centers, 2)) cells...")

        lck = ReentrantLock()
        while length(front) > 0
            visit! = (new_front, i) -> begin
                m = false
                lock(lck) do 
                    m = mask[i]
                end
                if m
                    return
                end

                pt = centers[:, i]
                R = radii[i]

                # calculate minimum distance to nearby surfaces
                dmin = minimum(
                    [dist(field, pt, R * cutoff_ratio * 2) for field in values(fields)]
                )

                # if below distance threshold, cut advancement
                if dmin <= R * cutoff_ratio
                    return 
                end

                # if near iteration cutoff point, flag it as a boundary projection point
                if dmin <= R * (cutoff_ratio + 2.5f0)
                    boundary[i] = true
                end

                lock(lck) do
                    mask[i] = true

                    for neigh in inrange(tree, pt, R * 3.1f0) # add non-visited nearby points to front
                        push!(new_front, neigh)
                    end
                end
            end

            new_front = Set{Int64}()

            front = collect(front)
            @threads for i in front
                visit!(new_front, i)
            end

            front = new_front
            verbose && println("Front with $(length(front)) cells...")
        end
        verbose && println("Done. $(sum(mask)) cells flagged as interior, $(sum(boundary)) boundary cells")

        (mask, boundary)
    end

    """
    $TYPEDSIGNATURES

    Ensure a graph is undirected
    """
    function ensure_undirected!(
        graph::AbstractVector{Vector{Ti}}
    ) where {Ti <: Integer} # just loop around making sure edges are always recyprocal
        for (i, neighs) in enumerate(graph)
            for n in neighs
                nneighs = graph[n]

                if !(i in nneighs)
                    push!(nneighs, i)
                end
            end
        end
    end

    """
    $TYPEDSIGNATURES

    Obtain boundary layer points given a point and its projection
    upon a surface
    """
    function BL_points(
        pt::AbstractVector{Tf},
        proj::AbstractVector{Tf},
        h0::Tf,
        growth_ratio::Tf = 1.1f0,
    ) where {Tf <: AbstractFloat}
        n = pt .- proj
        d = norm(n)
        n ./= d

        # start from surface
        pts = [copy(proj)]
        x = copy(proj)

        L = zero(Tf)
        h = h0

        while L + h < d
            L += h
            x .+= n .* h
            push!(pts, copy(x))

            h *= growth_ratio
        end

        reduce(hcat, pts)
    end

    """
    $TYPEDSIGNATURES

    Clean stencil via quadrant rule (max. number of neighbors in each quadrant).
    """
    function clean_stencil(
        X::AbstractMatrix{Tf}, 
        x::AbstractVector{Tf},
        indices::AbstractVector{Ti},
        n_per_quadrant::Int = 2,
    ) where {Tf <: AbstractFloat, Ti <: Integer}
        dX = (X[:, indices] .- x)
        let asrt = sum(dX .^ 2; dims = 1) |> vec |> sortperm
            dX = dX[:, asrt]
            indices = indices[asrt]
        end

        quadrants = map(
            dx -> tuple((dx .>= 0)...), eachcol(dX)
        )

        remains = falses(length(indices))
        for quad in unique(quadrants)
            n = n_per_quadrant

            for (i, q) in enumerate(quadrants)
                if n > 0 && q == quad
                    remains[i] = true
                    n -= 1
                end
            end
        end

        indices[remains]
    end

    """
    $TYPEDSIGNATURES

    Project boundary points to closest surfaces and create boundary layer.
    Returns two dicts, the first pointing strings with family names to vectors of
    boundary point indices, and the second, to vectors of pivot point indices.
    """
    function boundary_layer!(
        origin::Vector{Tf}, widths::Vector{Tf},
        graph::AbstractVector{Vector{Ti}},
        points::AbstractMatrix{Tf}, radii::AbstractVector{Tf},
        boundary::AbstractVector,
        distance_fields::Dict{String, DistanceField},
        boundary_layers::Dict{String, Tf},
        hypercube_families::AbstractVector;
        cutoff_ratio::Real = 2.1f0,
        growth_ratio::Real = 1.2f0,
        verbose::Bool = false,
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        bindices = findall(boundary)
        bpoints = points[:, bindices]
        # find a graph connecting only the points near a boundary
        in_boundary_graph = let newinds = Ti.(cumsum(boundary)) # new indexing
            bgraph = [
                filter(
                    n -> boundary[n], 
                    neighs
                ) |> x -> newinds[x] for neighs in graph[bindices]
            ]
            ensure_undirected!(bgraph)

            for i = 1:length(bgraph)
                bgraph[i] = clean_stencil(
                    bpoints, bpoints[:, i], bgraph[i];
                )

                if i % 10000 == 0
                    Base.GC.gc()
                end
            end
            ensure_undirected!(bgraph)

            bgraph
        end

        # store projections on boundaries, distances
        bprojs = similar(bpoints)
        bdists = similar(bpoints, (size(bpoints, 2),))
        bfams = Vector{String}(undef, length(bdists))

        bdists .= Inf32
        
        # search for nearest projection for each boundary point
        for (fname, dfield) in distance_fields
            verbose && println("Identifying family $fname boundaries...")
            itr = 1:size(bpoints, 2)
            if verbose
                itr = ProgressBar(itr)
            end

            for i in itr
                x = bpoints[:, i]
                p = BlockMesher.projection(dfield, x, radii[bindices[i]] * (cutoff_ratio + 2.5f0))
                d = norm(p .- x)

                if d < bdists[i]
                    bdists[i] = d
                    bprojs[:, i] .= p
                    bfams[i] = fname
                end
            end
        end

        # same for hypercube families (see structure in Graph())
        for (fname, faces) in hypercube_families
            verbose && println("Identifying hypercube family $fname boundaries...")

            for face in faces
                dim, front = face

                for (i, x) in eachcol(bpoints) |> enumerate
                    p = copy(x)
                    p[dim] = (
                        front ? origin[dim] + widths[dim] : origin[dim]
                    )
                    d = norm(x .- p)

                    if d < bdists[i]
                        bdists[i] = d
                        bprojs[:, i] .= p
                        bfams[i] = fname
                    end
                end
            end
        end

        new_points = Matrix{Tf}[]
        new_ranges = AbstractRange[] # store ranges of point indices in a BL 
        # column here, for each boundary point
        new_graph = Vector{Ti}[] # new links for boundary layer points here

        n = size(points, 2)
        for (i, (pt, proj, fname)) in zip(
            eachcol(bpoints), eachcol(bprojs), bfams
        ) |> enumerate
            h0 = bdists[i] / 3 # at least two points in the BL
            if haskey(boundary_layers, fname)
                h0 = min(boundary_layers[fname], h0)
            end

            pts = BL_points(pt, proj, h0, growth_ratio)
            rng = (n + 1):(n + size(pts, 2)) # range of indices
            n += size(pts, 2)

            push!(new_points, pts)
            push!(new_ranges, rng)
        end

        new_points = reduce(hcat, new_points)

        Tinew = Ti
        if size(new_points, 2) + size(points, 2) > 1e9
            Tinew = Int64
        end

        # make graph for layers

        # utility to transfer indices of point in the layers to global indices
        get_in_layers = (ipt, inormal) -> let rng = new_ranges[ipt]
            if inormal > length(rng)
                return Tinew(bindices[ipt])
            end

            return Tinew(rng[max(1, inormal)])
        end

        pivot_points = Dict{String, AbstractVector{Tinew}}()
        boundary_points = Dict{String, AbstractVector{Tinew}}()

        # register boundary points
        push_pair! = (sname, ib, ip) -> begin
            if !haskey(pivot_points, sname)
                pivot_points[sname] = Tinew[]
            end
            if !haskey(boundary_points, sname)
                boundary_points[sname] = Tinew[]
            end

            push!(boundary_points[sname], ib)
            push!(pivot_points[sname], ip)
        end

        verbose && println("Building graph for layer points...")
        itr = 1:length(new_ranges)
        if verbose
            itr = ProgressBar(itr)
        end

        for ipt in itr
            rng = new_ranges[ipt]
            push_pair!(
                bfams[ipt], rng[1], rng[2]
            )

            for inormal = 1:length(rng)
                neighs = [
                    get_in_layers(
                        n, inormal
                    ) for n in in_boundary_graph[ipt]
                ]

                push!(
                    neighs, get_in_layers(ipt, inormal + 1)
                )

                push!(new_graph, neighs)
                if length(new_graph) % 10000 == 0
                    Base.GC.gc()
                end
            end
        end

        (new_points, new_graph, boundary_points, pivot_points)
    end

    """
    $TYPEDSIGNATURES

    Build a set of stereolitography objects with facets limited to those reasonably close
    to a graph node
    """
    function filter_close_facets!(
        points::AbstractMatrix{Tf},
        graph::AbstractVector{Vector{Ti}},
        dfields::Dict{String, DistanceField},
        cutoff_ratio::Real = 2.0f0,
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        surfaces = Dict{String, Stereolitography}()

        # get the distance to the farthest node as a distance threshold
        distance_threshold = [
            (
                sum(
                    (points[:, neighs] .- points[:, i]) .^ 2; dims = 1
                ) |> maximum |> x -> sqrt(x) * cutoff_ratio
            ) for (i, neighs) in enumerate(graph)
        ]

        tree = KDTree(points)

        for (sname, dfield) in dfields
            stl = dfield.stl

            # filter points
            mask = map(
                pt -> let (i, d) = nn(tree, pt)
                    d <= distance_threshold[i]
                end, eachcol(stl.points), 
            )
            newinds = cumsum(mask)
            pts = stl.points[:, mask]

            # filter simplices
            mask_simps = map(
                simp -> all(s -> mask[s], simp), eachcol(stl.simplices)
            )
            simplices = newinds[stl.simplices[:, mask_simps]]

            surfaces[sname] = Stereolitography(
                pts, simplices
            )
        end

        surfaces
    end

    """
    $TYPEDFIELDS

    Struct describing a graph
    """
    struct Graph{Tf <: AbstractFloat, Ti <: Integer}
        octree::Mesh
        points::AbstractMatrix{Tf}
        neighbors::AbstractVector{Vector{Ti}}
        surfaces::Dict{String, Stereolitography}
        boundary_points::Dict{String, AbstractVector{Ti}}
        pivot_points::Dict{String, AbstractVector{Ti}}
    end

    """
    $TYPEDSIGNATURES

    Generate a graph given hypercube origins and widths.

    Surfaces should be specified as tuples with family names,
    `Stereolitography` structs and local refinement levels, respectively.

    Refinement regions, meanwhile, may be specified as tuples between distance
    functions (see `Line, Box, Ball, DistanceField` in this module) and local refinement levels.

    An approximate growth rate is accepted for the block octree/quadtree.

    Example:

    ```
    stl = Stereolitography("wall.dat") # or STL in 3D
    stl2 = Stereolitography("wall2.dat")

    features = feature_regions(stl) |> DistanceField
    region2 = Stereolitography("region.stl") |> DistanceField

    msh = Graph(
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
    ```

    Point `interior_reference` is used to define the inside of the domain.
    Ratio `cutoff_ratio` is used to determine how far outside of the boundaries
    an octree cell must be to be considered within the domain, as a function of its
    circumradius.

    Boundary layer thicknesses must be specified for walls with anisotropic refinement
    as per kwarg `boundary_layers`:

    ```
    boundary_layers = [
        "family1" => 1e-4, # first height
        "family2" => 2e-4
    ]
    ```

    Hypercube boundary families may be specified as in:

    ```
    hypercube_families = [
        "inlet" => [
            (1, false), # x-axis, front
            (2, false), # y-axis, left
            (2, true), # y-axis, right
            (3, false), # z-axis, bottom
            (3, true), # z-axis, top
        ],
        "outlet" => [(1, true)]
    ] # defaults to all faces in family "FARFIELD"
    ```
    """
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
        if length(hypercube_families) == 0
            bdry = Tuple{Int64, Bool}[]
            for dim = 1:length(origin)
                push!(bdry, (dim, false))
                push!(bdry, (dim, true))
            end

            hypercube_families = ["FARFIELD" => bdry]
        end

        block_size = log(2.0) / log(growth_ratio) |> floor |> Int64
        block_size = max(1, block_size)

        Tf = (tolerance < 1e-7 ? Float64 : Float32)
        growth_ratio = Tf(growth_ratio)

        boundary_layers = Dict{String, Tf}(boundary_layers...)

        verbose && println("Running octree construction...")
        octree = BlockMesher.Mesh(
            origin, widths, surfaces...;
            tolerance = tolerance, block_size = block_size, 
            refinement_regions = refinement_regions, verbose = verbose,
        )

        centers, cell_widths, _ = BlockMesher.get_cells(octree)

        centers = Tf.(centers)
        cell_widths = Tf.(cell_widths)
        origin = Tf.(origin)
        widths = Tf.(widths)

        verbose && println("Running flood-fill algorithm to find domain interior...")
        t0 = time()

        mask, boundary = flood_fill(
            octree.distance_fields, centers, cell_widths,
            interior_reference; cutoff_ratio = cutoff_ratio,
            verbose = verbose,
        )

        Ti = Int32 # check which integer type is necessary 
        # based on cell count
        if sum(mask) > 1e9
            Ti = Int64
        end

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        # set hypercube boundary cells as boundaries
        for dim = 1:size(centers, 1)
            for i = 1:length(mask)
                if mask[i]
                    if abs(origin[dim] - centers[dim, i]) < cell_widths[dim, i] * 1.1f0
                        boundary[i] = true
                    end

                    if abs(origin[dim] + widths[dim] - centers[dim, i]) < cell_widths[dim, i] * 1.1f0
                        boundary[i] = true
                    end
                end
            end
        end

        mask = findall(mask)

        points = centers[:, mask]
        radii = sum(
            cell_widths[:, mask] .^ 2; dims = 1
        ) |> vec |> x -> sqrt.(x) ./ 2
        boundary = boundary[mask]

        # deallocate
        centers = nothing
        cell_widths = nothing
        # mask = nothing

        verbose && println("Generating initial graph...")
        t0 = time()

        # generate initial graph
        neighbors = let tree = KDTree(points)
            ratio = Tf(2.5 / sqrt(size(points, 1)))
            bratio = Tf(
                (cutoff_ratio / 2 + 2) + 0.01f0
            )

            [
                let pt = points[:, i]
                    R = radii[i] * (
                        boundary[i] ?
                        bratio : ratio
                    )

                    Ti.(setdiff(inrange(tree, pt, R), [i]))
                end for i = 1:size(points, 2)
            ]
        end

        # ensure symmetric links
        ensure_undirected!(neighbors)

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        verbose && println("Adding boundary layer points...")

        (boundary_points, pivot_points) = begin
            new_points, new_graph, boundary_points, pivot_points = boundary_layer!(
                origin, widths,
                neighbors, points, radii, boundary,
                octree.distance_fields, boundary_layers,
                hypercube_families; verbose = verbose, growth_ratio = growth_ratio,
            )
            
            # check if it's necessary to change the graph data type based
            # on the new points at the BL
            Tiold = Ti
            Ti = (first(new_graph) |> eltype)

            if Ti != Tiold
                for i = 1:length(neighbors)
                    neighbors[i] = Ti.(neighbors[i])

                    if i % 10000 == 0
                        Base.GC.gc()
                    end
                end
            end

            points = [points new_points]
            neighbors = [neighbors; new_graph]

            (boundary_points, pivot_points)
        end

        # ensure symmetric links
        ensure_undirected!(neighbors)

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        # process surfaces
        println("Processing surfaces...")
        t0 = time()

        surfaces = filter_close_facets!(points, neighbors, octree.distance_fields, cutoff_ratio)

        # erase distance fields in octree data structure: no need to hold them anymore
        for sname in keys(octree.distance_fields)
            delete!(octree.distance_fields, sname)
        end

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        npoints = size(points, 2)
        nedges = sum(length, neighbors)
        verbose && println("$npoints points, $nedges edges")

        Base.GC.gc()
        verbose && println("==Done with graph generation!==")

        Graph{Tf, Ti}(
            octree, points, neighbors, surfaces, boundary_points, pivot_points
        )
    end

    """
    $TYPEDSIGNATURES

    Partition graph.

    Returns the following data structures:

    ```
    partitions = partition(graph)

    for (part, domain, n_skirt) in partitions
        # part: Graph instance for the current sub-graph
        # domain: indices of all points in the current sub-graph
        # n_skirt: number of points in "domain" (the last n_skirt in the array)
        #    which are in the skirt of the partition, with `skirt_order`
        #    layers of nodes around the partition being selected for 
        #    inter-partition communication
    end
    ```
    """
    function partition(
        graph::Graph{Tf, Ti},
        max_partition_size::Int = 500_000;
        skirt_order::Int = 1,
    ) where {Tf, Ti}
        @assert skirt_order >= 1 "Skirt order must be at least 1"

        n_rounds = log2(max_partition_size) |> ceil |> Int64
        parts = let weights = [
            (
                sum(
                    (graph.points[:, neighs] .- graph.points[:, i]) .^ 2;
                    dims = 1
                ) |> vec |> x -> 1.0f0 ./ (sqrt.(x) .+ eps(Tf))
            ) for (i, neighs) in enumerate(graph.neighbors)
        ]
            METIS.partition(graph.neighbors, n_rounds, weights)
        end

        partids = zeros(Ti, length(graph.neighbors))
        ind2doms = Dict{Ti, Ti}[] # domain index mapping dictionaries

        parts = [
            begin
                partids[domain] .= p

                sort!(domain)
                skirt = METIS.skirt(graph.neighbors, domain, skirt_order)

                n_skirt = length(skirt)
                domain = [domain; skirt]

                ind2dom = Dict(
                    [k => Ti(i) for (i, k) in enumerate(domain)]...
                )

                neighbors = [
                    [
                        ind2dom[n] for n in neighs if haskey(ind2dom, n)
                    ] for neighs in graph.neighbors[domain]
                ]

                boundary_points = Dict{String, AbstractVector{Ti}}()
                pivot_points = Dict{String, AbstractVector{Ti}}()

                for bname in keys(graph.boundary_points)
                    boundary_points[bname] = Ti[]
                    pivot_points[bname] = Ti[]
                end

                push!(ind2doms, ind2dom)

                (
                    Graph{Tf, Ti}(
                        graph.octree,
                        graph.points[:, domain],
                        neighbors, 
                        Dict{String, Stereolitography}(), # not storing this for each partition
                        boundary_points, pivot_points
                    ), domain, n_skirt
                )
            end for (p, domain) in enumerate(parts)
        ]

        push_pair! = (sname, i, p) -> begin # push pair of boundary and pivot points to bdry dicts
            pid = partids[i]

            inew = ind2doms[pid][i]
            if !haskey(ind2doms[pid], p)
                return
            end
            pnew = ind2doms[pid][p]

            g = parts[pid][1]

            push!(g.boundary_points[sname], inew)
            push!(g.pivot_points[sname], pnew)
            ;
        end

        for sname in keys(graph.boundary_points)
            for (i, p) in zip(
                graph.boundary_points[sname], graph.pivot_points[sname]
            )
                push_pair!(sname, i, p)
            end
        end

        parts
    end

    """
    $TYPEDSIGNATURES

    Write graph using WriteVTK. Uses kwargs as cell data.
    """
    function WriteVTK.vtk_grid(
        fname::String, graph::Graph{Tf, Ti};
        kwargs...
    ) where {Tf <: AbstractFloat, Ti <: Integer}
        cells = MeshCell[]
        for (i, neighs) in enumerate(graph.neighbors)
            for n in neighs
                push!(
                    cells, MeshCell(
                        VTKCellTypes.VTK_LINE, [i, n]
                    )
                )
            end
        end

        vtk = vtk_grid(fname, graph.points, cells)

        for (k, v) in kwargs
            vtk[String(k)] = v
        end

        vtk
    end

    """
    $TYPEDSIGNATURES

    Extract a slice for VTK debugging of 3D grids.
    """
    function WriteVTK.vtk_grid(
        fname::String, graph::Graph{Tf, Ti},
        origin::AbstractVector, normal::AbstractVector;
        kwargs...
    ) where {Tf <: AbstractFloat, Ti <: Integer}
        point_sdf = (
            graph.points .- origin
        )' * normal

        point_mask = falses(length(point_sdf))
        for i = 1:length(point_sdf)
            neighbors = graph.neighbors[i]
            
            for n in neighbors
                if point_sdf[i] * point_sdf[n] < eps(Tf)
                    point_mask[i] = true
                    point_mask[n] = true
                end
            end
        end
        point_indices = findall(point_mask)
        new_inds = cumsum(point_mask)

        graph = Graph{Tf, Ti}(
            graph.octree,
            graph.points[:, point_indices],
            map(
                i -> let neighs = graph.neighbors[i]
                    [
                        new_inds[n] for n in neighs if point_mask[n]
                    ]
                end, point_indices
            ),
            Dict{String, Stereolitography}(), 
            Dict{String, AbstractVector{Ti}}(),
            Dict{String, AbstractVector{Ti}}(),
        )

        vtk_grid(
            fname, graph;
            [
                k => v[point_indices] for (k, v) in kwargs
            ]...
        )
    end

    """
    $TYPEDSIGNATURES

    Returns fine graph and vector of coarse graphs produced under 
    similar meshing parameters, each more sparse than the last by 
    factor `factor` for grid spacing.

    Increases boundary layer first heights until isotropy is reached, then
    handles surface refinement.
    """
    function multigrid(
        n_levels::Int,
        origin::AbstractVector, widths::AbstractVector,
        surfaces::Tuple...;
        factor::Real = 2.0f0,
        refinement_regions::AbstractVector = [],
        boundary_layers::AbstractVector = [],
        verbose::Bool = false,
        kwargs...
    )
        verbose && println("Building fine refinement level...")
        graph = Graph(
            origin, widths, surfaces...;
            refinement_regions = refinement_regions, 
            boundary_layers = boundary_layers,
            verbose = verbose, kwargs...
        )

        blayer_factor = 2 ^ (length(origin) - 1)

        coarse_graphs = []
        _boundary_layers = Dict(boundary_layers...)
        for nlevel = 1:n_levels
            surfaces = [
                let (sname, stl, h) = surf
                    if haskey(_boundary_layers, sname)
                        if _boundary_layers[sname] < h / 2
                            _boundary_layers[sname] = _boundary_layers[sname] * blayer_factor
                        else
                            h = h * factor
                        end
                    else
                        h = h * factor
                    end

                    (sname, stl, h)
                end for surf in surfaces
            ]
            refinement_regions = [
                dfield => h * factor for (dfield, h) in refinement_regions
            ]

            verbose && println("Building coarse level $nlevel...")

            push!(
                coarse_graphs,
                Graph(
                    origin, widths, surfaces...;
                    refinement_regions = refinement_regions, 
                    boundary_layers = _boundary_layers |> collect,
                    verbose = verbose, kwargs...
                )
            )
        end

        graph, coarse_graphs
    end

end
