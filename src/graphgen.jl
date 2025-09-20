module GraphGen

    const max_layer_ratio = 5.0

    include("mesher.jl")
    using .Mesher
    using .Mesher: Ball, Box, Line, Triangulation,
        Stereolitography, STLTree, Mesh, FixedMesh,
        point_in_polygon, mesh2json, json2mesh,
        vtk_grid
    using .Mesher.STLHandler: stl2vtk, in_range,
        feature_edges, refine_to_length, centers_and_normals

    using .Mesher.WriteVTK
    using .Mesher.LinearAlgebra
    using .Mesher.DocStringExtensions

    include("nninterp.jl")
    using .NNInterpolator
    using .NNInterpolator.NearestNeighbors
    using .NNInterpolator: KDTree

    include("quad_nn.jl")
    using .QuadrantNN

    export Ball, Box, Line, Triangulation,
        Stereolitography, STLTree, Mesh, FixedMesh,
        point_in_polygon, mesh2json, json2mesh,
        stl2vtk, in_range,
        feature_edges, 
        vtk_grid, vtk_save,
        Graph, BoundaryLayer,
        feature_edges, refine_to_length, centers_and_normals

    """
    Ensure symmetry in graph
    """
    function impose_symmetry!(connectivity::AbstractVector{Vector{Ti}}) where {Ti <: Integer}
        for (i, conn) in enumerate(connectivity)
                for j in conn
                        if i < j
                                if !(i in connectivity[j])
                                    push!(connectivity[j], i)
                                end

                                if !(j in conn)
                                    push!(conn, j)
                                end
                        end
                end
        end
    end

    """
    $TYPEDFIELDS

    Struct defining a graph
    """
    struct Graph{T <: Real, Ti <: Integer}
        points::AbstractMatrix{T}
        connectivity::AbstractVector{Vector{Ti}}
        boundary_points::Dict{String, Vector{Ti}}
        boundary_pivots::Dict{String, Vector{Ti}}
    end

    """
    Turn conn. graph to edge list
    """
    function graph2edgelist(graph::AbstractVector{Vector{Ti}}) where {Ti <: Integer}
        edges = Set{Tuple{Ti, Ti}}()

        for (i, neighs) in enumerate(graph)
                for n in neighs
                        if i < n
                            push!(edges, (i, n))
                        end
                end
        end

        edges
    end

    """
    Obtain exponential spacing between a point and its projection
    """
    function BL_points(
        x::AbstractVector{T}, p::AbstractVector{T}, 
        circumradius::T, h0::T,
        growth_ratio::T = 1.1,
        layer_size_ratio::T = 5.0,
    ) where {T <: Real}
        n = x .- p
        L = norm(n) + eps(T)

        n ./= L
        L -= circumradius # remove some room from BL height

        h = h0
        pts = [copy(p)] # start with projection
        Σ = T(0.0)

        while Σ + h <= L
            push!(
                pts, pts[end] .+ n .* h
            )

            Σ += h
            h *= growth_ratio
        end

        # concatenate points into single array
        pts = mapreduce(transpose, vcat, pts[end:-1:1])

        pts
    end

    """
    $TYPEDFIELDS

    Struct to define a boundary layer.
    `layer_size_ratio` is the ratio between the local BL height and the cell circumradius
    """
    struct BoundaryLayer{T <: Real}
        first_height::T
        growth_ratio::T
        layer_size_ratio::T # ratio between layer height and cell circumradius
    end

    """
    $TYPEDSIGNATURES

    Constructor for a boundary layer definition
    """
    BoundaryLayer(
        first_height::T, growth_ratio::T = 1.1;
        layer_size_ratio::T = 5.0
    ) where {T <: Real} = BoundaryLayer{T}(first_height, growth_ratio, min(layer_size_ratio, max_layer_ratio))

    """
    Add boundary layer nodes and edges to point matrix and connectivity graph;
    return new arrays
    """
    function add_boundary_layers(
        msh::Mesh,
        points::AbstractMatrix{T},
        radii::AbstractVector{T},
        blayers::Dict{String, BoundaryLayer{T}};
        Ti::Type = Int64,
    ) where {T <: Real}
        npts = size(points, 1)
        nd = size(points, 2)

        BLpoints = Matrix{T}[]
        BLtails = Ti[] # boundary nodes
        BLheads = Ti[] # nodes from which the boundary nodes withdraw information
        BLsurfaces = String[] # strings on which each boundary node lies

        mask = trues(npts)

        npts = Ti(0)
        register_points! = pts -> let np = size(pts, 1)
            npts += np # add to point counter

            push!(BLpoints, pts) # register to array

            npts
        end

        sdf_dict = Dict{String, Vector{T}}()

        # first establish mask and calc. sdfs
        for sname in keys(msh.boundary_projections)
            blyr = blayers[sname]

            projs = msh.boundary_projections[sname]'
            isin = msh.boundary_in_domain[sname]

            sdfs = (
                sum(
                    (projs .- points) .^ 2; dims = 2
                ) |> vec |> x -> sqrt.(x)
            ) .* (2 .* isin .- 1)

            @. mask = mask && (sdfs >= blyr.layer_size_ratio * radii)
            sdf_dict[sname] = sdfs
        end

        # register points filtering per mask and calculate new point indices 
        register_points!(points[mask, :])
        newinds = cumsum(mask) |> x -> Ti.(x)

        # now go through limitrophe mask points and add BL points
        for sname in keys(sdf_dict)
            blyr = blayers[sname]
            sdfs = sdf_dict[sname]
            projs = msh.boundary_projections[sname]'

            # add BL point if between the BL edge and the BL height + sqrt(N) * R, 
            # R being the cell circumradius
            should_project = findall(
                (sdfs .<= radii .* (blyr.layer_size_ratio + sqrt(nd))) .&& mask
            )

            for ihead in should_project # "head" of boundary layer
                head = points[ihead, :]
                proj = projs[ihead, :]

                pts = BL_points(head, proj, radii[ihead], 
                    blyr.first_height, blyr.growth_ratio)

                itail = register_points!(pts)

                # if we have more than one BL point, use the second to last as head
                new_ihead = Ti(itail - 1)
                if size(pts, 1) == 1 # ...else use the original projected point
                    new_ihead = newinds[ihead]
                end

                push!(BLtails, itail)
                push!(BLheads, new_ihead) # register BL head with new index
                push!(BLsurfaces, sname)
            end
        end

        points = reduce(
            vcat, 
            BLpoints
        )

        (
            points, BLtails, BLheads, BLsurfaces
        )
    end

    """
    Fix surface-penetrating connectivity for pivot points
    """
    function filter_no_penetration!(
        points::AbstractMatrix{T}, connectivity::AbstractVector{Vector{Ti}},
        ihead::Ti, itail::Ti
    ) where {Ti <: Integer, T <: Real}
        head = points[ihead, :]
        tail = points[itail, :]

        conn = connectivity[ihead]

        v = tail .- head
        isval = ((points[conn, :] .- head') * v) .<= 0.0

        conn = conn[isval]
        if !(itail in conn)
            push!(conn, itail)
        end

        connectivity[ihead] = conn
    end

    """
    $TYPEDSIGNATURES

    Build a graph from a mesh.

    Boundary layers must be specified as boundary name/`BoundaryLayer` pairs:

    ```
    boundary_layers = [
        "wall" => BoundaryLayer(1e-4, 1.1),
        "engine-inlet" => BoundaryLayer(1e-4, 1.1), # first height, growth ratio
    ]
    ```
    """
    function Graph(
        msh::Mesh;
        boundary_layers = [],
        n_per_quadrant::Int = 1,
        order::Int = 2,
        verbose::Bool = false,
    )
        T = Float64
        Ti = (
                length(msh) > 1000_000_000 ?
                Int64 :
                Int32
        )

        points = msh.centers |> permutedims
        radii = (msh.widths' .^ 2) |> x -> sum(x; dims = 2) |> vec |> x -> sqrt.(x) ./ 2

        boundary_layers = let bdict = Dict{
            String, BoundaryLayer{T}
        }()
            for (fname, blyr) in boundary_layers
                bdict[fname] = blyr
            end

            for fname in keys(msh.boundary_projections)
                if !haskey(bdict, fname)
                    bdict[fname] = BoundaryLayer{T}(
                        typemax(T), typemax(T), 0.0
                    )
                end
            end
            
            bdict
        end

        points, BLtails, BLheads, BLsurfaces = add_boundary_layers(
            msh, points, radii, boundary_layers;
            Ti = Ti
        )
        verbose && println("$(size(points, 1)) points after BL insertion")

        verbose && println("Defining connectivity...")
        connectivity = quadrant_rule_graph(
            permutedims(points); n_per_quadrant = n_per_quadrant,
        )
        connectivity = map(
            conn -> Ti.(conn), connectivity
        )

        for (t, h) in zip(BLtails, BLheads)
            connectivity[t] = [h]
            filter_no_penetration!(points, connectivity, h, t)
        end

        for _ = 1:(order - 1)
            connectivity = map(
                i -> let conn = connectivity[i]
                    setdiff(
                        reduce(union, connectivity[conn]), i
                    )
                end, 1:length(connectivity)
            )
        end

        impose_symmetry!(connectivity)

        verbose && let ncon = sum(length, connectivity)
            println("$ncon virtual faces found")
        end

        boundary_points = Dict{String, Vector{Ti}}()
        boundary_pivots = Dict{String, Vector{Ti}}()

        for sname in unique(BLsurfaces)
            mask = findall(BLsurfaces .== sname)

            boundary_points[sname] = BLtails[mask]
            boundary_pivots[sname] = BLheads[mask]
        end

        Graph{T, Ti}(points, connectivity, 
            boundary_points, boundary_pivots)
    end

    """
    $TYPEDSIGNATURES

    Export graph to VTK format
    """
    function WriteVTK.vtk_grid(fname, graph::Graph{T, Ti};
        kwargs...) where {T <: Real, Ti <: Integer}
        points = graph.points |> permutedims
        cells = [
                MeshCell(VTKCellTypes.VTK_LINE, edge |> collect) for edge in graph2edgelist(graph.connectivity)
        ]

        grid = vtk_grid(fname, points, cells)
        for (k, v) in kwargs
                grid[String(k)] = v
        end

        pivots = zeros(size(graph.points, 1))
        bpoints = zeros(size(graph.points, 1))
        for sname in keys(graph.boundary_points)
            bpts = graph.boundary_points[sname]
            bpivs = graph.boundary_pivots[sname]

            bpoints[bpts] .= 1.0
            pivots[bpivs] .= 1.0
        end

        grid["PIVOT_POINT"] = pivots
        grid["BOUNDARY_POINT"] = bpoints

        grid
    end

end
