module Meshless

    include("graphgen.jl")
    using .GraphGen
    using .GraphGen.Mesher: @threads

    include("nninterp.jl")
    using .NNInterpolator
    using .NNInterpolator: KDTree

    using .GraphGen.DocStringExtensions
    using .GraphGen.LinearAlgebra
    using .GraphGen.WriteVTK

    include("accumulator.jl")
    using .ArrayAccumulator

    include("arraybends.jl")
    using .ArrayBackends

    include("metis.jl")
    using .METIS

    include("cfd.jl")
    using .CFD

    export Ball, Box, Line, Triangulation,
        Stereolitography, STLTree, Mesh, FixedMesh,
        point_in_polygon, mesh2json, json2mesh,
        stl2vtk, in_range,
        feature_edges, refine_to_length,
        vtk_grid, vtk_save,
        Graph, BoundaryLayer,
        Domain, Partition,
        Interpolator, Surface, surface_integral,
        impose_bc!,
        artificial_dissipation, JST_sensor,
        timescale, smoothing, 
        CFD

    include("poly_regressor.jl")
    using .PolyRegression

    """
    $TYPEDFIELDS

    Struct to define a boundary
    """
    struct Boundary
        boundary_points::AbstractVector{Int64}
        pivot_points::AbstractVector{Int64}
        points::AbstractMatrix{Float64}
        distances::AbstractVector{Float64}
        normals::AbstractMatrix{Float64}
    end

    """
    $TYPEDSIGNATURES

    Construct a boundary from a set of points and vectors of boundary and pivot
    indices.
    """
    function Boundary(
        points::AbstractMatrix{Float64},
        boundary_points::AbstractVector{Int64}, pivot_points::AbstractVector{Int64}
    )
        bpoints = points[boundary_points, :]
        n = points[pivot_points, :] .- bpoints

        distances = sum(n .^ 2; dims = 2) |> vec |> x -> sqrt.(x)
        n ./= (distances .+ eps(Float64))

        Boundary(
            boundary_points, pivot_points, bpoints,
            distances, n
        )
    end

    """
    $TYPEDFIELDS

    Struct to describe a domain partition
    """
    struct Partition{N}
        id::Int64
        domain::AbstractVector{Int64}
        image::AbstractVector{Int64}
        image_in_domain::AbstractVector{Int64}
        points::AbstractMatrix{Float64}
        boundaries::Dict{String, Boundary}
        numerical_laplacian::NTuple{N, Accumulator}
        smoothing::Accumulator
        derivatives::Dict{NTuple{N, Int64}, Accumulator}
    end

    @declare_converter Accumulator
    @declare_converter Partition

    """
    $TYPEDFIELDS

    Struct to define a domain
    """
    struct Domain{N}
        graph::Graph
        partitions::AbstractVector{Partition{N}}
        tree::KDTree
        boundary_points::AbstractVector{Int64}
        boundary_tree::KDTree
    end

    """
    $TYPEDSIGNATURES

    Obtain number of points in domain
    """
    Base.length(dom::Domain) = size(dom.graph.points, 1)

    """
    $TYPEDSIGNATURES

    Constructor for a domain from a graph.

    Uses p-METIS algorithm to ensure we are only reaching up to a max. partition
    size.
    """
    function Domain(
        graph::Graph{T, Ti};
        partition_size::Int = 1000_000,
        degree::Int = 2,
        verbose::Bool = false,
    ) where {T <: Real, Ti <: Integer}
        N = size(graph.points, 2)
        npts = size(graph.points, 1)

        points = Float64.(graph.points)

        p = max(
            0,
            log2(partition_size) |> floor |> Int64
        )

        partitions = nothing
        if p > 0
            verbose && println("Running p-METIS algorithm...")
            partitions = METIS.partition(graph.connectivity, p)
        else
            partitions = [
                collect(1:npts)
            ]
        end

        indomain = falses(npts)
        partitions = [
            let image = partition
                skrt = METIS.skirt(graph.connectivity, image)

                image_in_domain = collect(1:length(image))
                domain = [image; skrt]

                domain_indmap = Dict(
                    [d => i for (i, d) in enumerate(domain)]
                )

                indomain[domain] .= true
                bdries = Dict{String, Boundary}()
                for (bname, bpoints) in graph.boundary_points
                    pivots = graph.boundary_pivots[bname]

                    # make sure all pivot/boundary pairs are in the domain
                    for (p, b) in zip(pivots, bpoints)
                        if indomain[p] && !indomain[b]
                            push!(domain, b)
                            domain_indmap[b] = length(domain)
                            indomain[b] = true
                        elseif indomain[b] && !indomain[p]
                            push!(domain, p)
                            domain_indmap[p] = length(domain)
                            indomain[p] = true
                        end
                    end

                    mask = indomain[bpoints] |> findall

                    bdries[bname] = Boundary(
                        view(points, domain, :),
                        map(i -> domain_indmap[i], bpoints[mask]),
                        map(i -> domain_indmap[i], pivots[mask]),
                    )
                end

                stencils = map(
                    i -> [
                        domain_indmap[i];
                        (
                            map(
                                n -> (
                                    indomain[n] ?
                                    domain_indmap[n] :
                                    0
                                ), 
                                graph.connectivity[i]
                            ) |> neighs -> filter(n -> n != 0, neighs)
                        )
                    ], domain
                )

                indomain[domain] .= false

                pts = points[domain, :]
                derivatives, numerical_laplacian = let weights = Dict{
                    NTuple{N, Int64}, AbstractVector
                }()
                    exponents = polynomial_exponents(N, degree)

                    let weight_matrices = map(
                        neighs -> regression_weights(pts[neighs, :],
                            pts[neighs[1], :], exponents),
                        stencils
                    )
                        for (i, e) in enumerate(exponents) 
                            weights[e] = map(
                                M -> M[:, i], weight_matrices
                            )
                        end
                    end

                    derivatives = Dict(
                        [
                            e => Accumulator(stencils, w;
                                first_index = true) for (e, w) in weights
                        ]
                    )

                    numerical_laplacian = map(
                        i -> let e = zeros(Int64, N)
                            e[i] = 1
                            e = tuple(e...)
                            
                            Accumulator(stencils,
                                map(w -> abs.(w), weights[e]);
                                first_index = true)
                        end, 1:N
                    ) |> x -> tuple(x...)

                    (derivatives, numerical_laplacian)
                end

                smoothing = Accumulator(
                    stencils, [
                        fill(1.0 / length(v)) for v in stencils
                    ]; first_index = true
                )

                Partition{N}(
                    id, domain, image, image_in_domain,
                    pts, bdries,
                    numerical_laplacian, smoothing, derivatives
                )
            end for (id, partition) in enumerate(partitions)
        ]

        if verbose
            for (i, part) in enumerate(partitions)
                println("Partition $i: $(length(part.domain)) domain, $(length(part.image)) image")
            end
        end

        tree = KDTree(graph.points' |> x -> Float64.(x))

        bdry_tree, bdry_points = let isbdry = falses(npts)
            for bpoints in values(graph.boundary_points)
                isbdry[bpoints] .= true
            end

            isbdry = findall(isbdry)

            (
                KDTree(graph.points[isbdry, :]' |> x -> Float64.(x)),
                isbdry
            )
        end

        Domain{N}(
            graph, partitions, tree,
            bdry_points, bdry_tree,
        )
    end

    """
    $TYPEDSIGNATURES

    Impose boundary conditions at a given boundary.

    The provided function should receive the values of arrays in args at pivot points
    and return their values at boundary points.

    Example with the non-penetration condition:

    ```
    impose_bc!(part, "wall", u, v) do bdry, u, v
        ub = copy(u)
        vb = copy(v)

        nx, ny = bdry.normals |> eachcol
        # see also: bdry.points, bdry.distances...

        un = @. nx * u + ny * v
        ub .-= un .* nx
        vb .-= un .* ny

        (
            ub, vb
        )
    end
    ```

    Another example, but with a single array in which the first dimension indicates
    the node index:

    ```
    impose_bc!(
        part, "wall", uv
    ) do bdry, uv
        uvb = uv .- sum(
            uv .* bdry.normals; dims = 2
        ) .* bdry.normals

        uvb
    end
    ```

    Kwargs are passed as they are to the BC function.

    Note that more field variable args may be passed than output arrays. In this case,
    only the first input field variables will be edited in place as per return values. Example:

    ```
    impose_bc!(
        part, "wall", u, v, w
    ) do bdry, u, v, w
        # do something to find ub, vb

        (ub, vb)
    end
    ```
    """
    function impose_bc!(
        f, part::Partition, bname::String,
        args::AbstractArray...; kwargs...
    )
        bdry = part.boundaries[bname]

        if length(bdry.pivot_points) == 0
            return
        end

        pivot_args = map(
            a -> selectdim(a, 1, bdry.pivot_points) |> copy,
            args
        )
        
        r = f(bdry, pivot_args...; kwargs...)

        if r isa AbstractArray
            if eltype(r) <: AbstractArray
                r = tuple(r...)
            else
                r = (r,)
            end
        end

        for (rr, a) in zip(r, args)
            selectdim(a, 1, bdry.boundary_points) .= rr
        end

        ;
    end

    """
    $TYPEDSIGNATURES

    Run a loop over the partitions of a domain and
    execute operations.

    Example:

    ```
    domain(r, u) do partition, rdom, udom
        # udom includes the parts of vector `u`
        # which affect the residual at partition `partition`.

        # now do some Cartesian grid operations and
        # update rdom
    end

    # after the loop, the values of `rdom` are returned to
    # array `r`
    ```

    This allows for large operations on field data
    to be performed one partition at a time,
    saving on max. memory usage.

    Return values are also stored in a vector, which is
    then returned.
    Kwargs are passed to the called function.

    Backend "converters" can be used to convert arrays to certain
    array operation libraries (e.g. CUDA.jl) before operations are 
    conducted. Example:

    ```
    dom(
        u;
        conv_to_backend = CuArray, # may also be a custom function
        conv_from_backend = Array
    ) do part, udom
        @show typeof(udom) # CuArray
    end
    ```

    This ensures that, while operating with GPU parallelization, 
    information regarding a single partition at a time is ported to the GPU,
    thus satisfying far tighter memory requirements.
    """
    function (dom::Domain)(
        f, args::AbstractArray...; 
        conv_to_backend = nothing,
        conv_from_backend = nothing,
        kwargs...
    )
        ret = Vector{Any}(undef, length(dom.partitions))
        @threads for ip = 1:length(ret)
            part = dom.partitions[ip]

            ret[ip] = let pargs = map(
                a -> selectdim(a, 1, part.domain) |> copy, 
                args
            )
                mypart = part
                if !isnothing(conv_to_backend)
                    pargs = map(
                        a -> to_backend(a, conv_to_backend), pargs
                    )
                    mypart = to_backend(part, conv_to_backend)
                end

                r = f(mypart, pargs...; kwargs...)
            
                if !isnothing(conv_from_backend)
                    pargs = map(
                        a -> to_backend(a, conv_from_backend), pargs
                    )
                end

                for (a, pa) in zip(args, pargs)
                    selectdim(a, 1, part.image) .= selectdim(pa, 1, part.image_in_domain)
                end

                r
            end
        end

        ret
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator (callable struct) from a domain to a matrix of points (shape
        `(npts, ndims)`).

    Uses first array dimension as a point index.

    If `from_surface` is true, the interpolation is performed from points
    indexed in integer array `domain.boundary_points`.
    """
    function Interpolator(
        domain::Domain,
        points::AbstractMatrix{Float64};
        from_surface::Bool = false,
        linear::Bool = true,
    )
        tree = domain.tree
        graph_points = domain.graph.points

        if from_surface
            tree = domain.boundary_tree
            graph_points = graph_points[domain.boundary_points, :]
        end

        intp = NNInterpolator.Interpolator(
            graph_points, points, tree; linear = linear, first_index = true
        )

        if !from_surface
            return intp
        end

        let hmap = Dict(
            [
                i => bp for (i, bp) in enumerate(domain.boundary_points)
            ]...
        )
            NNInterpolator.filtered(intp, hmap)
        end
    end

    """
    $TYPEDFIELDS

    Struct representing a surface for property integration and postprocessing
    """
    struct Surface
        stereolitography::Stereolitography
        points::AbstractMatrix{Float64}
        normals::AbstractMatrix{Float64}
        areas::AbstractVector{Float64}
        interpolator::NNInterpolator.Interpolator
    end

    """
    $TYPEDSIGNATURES

    Obtain a surface from a domain and a stereolitography object.

    If `max_length` is provided, the STL surface is refined by tri splitting until no
    triangle side is larger than the provided value.

    Data is interpolated only from boundary nodes if `from_surface` is true.
    """
    function Surface(
        domain::Domain, 
        stl::Stereolitography; 
        max_length::Float64 = 0.0,
        from_surface::Bool = true,
    )

        if max_length > 0.0
            stl = refine_to_length(stl, max_length)
        end

        nd = size(stl.points, 1)
        points = permutedims(stl.points)

        interpolator = Interpolator(domain, points; from_surface = from_surface)

        _, normals = centers_and_normals(stl)

        normals = let point_normals = similar(stl.points)
            point_normals .= 0.0
            
            for ipts in eachrow(stl.simplices)
                for (n, ipt) in zip(eachcol(normals), ipts)
                    point_normals[:, ipt] .+= n ./ nd
                end
            end

            permutedims(point_normals)
        end

        ϵ = eltype(normals) |> eps
        areas = map(
                    n -> norm(n) + ϵ, eachrow(normals)
        )
        normals = normals ./ areas

        Surface(
            deepcopy(stl),
            points,
            normals,
            areas,
            interpolator
        )

    end

    """
    $TYPEDSIGNATURES

    Interpolate a field property to a surface. The first index should refer to 
    cell/surface point index.
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
    $TYPEDSIGNATURES

    Export VTK grid for surface. kwargs are passed as point data, and interpolated if necessary.
    """
    function WriteVTK.vtk_grid(fname::String, surf::Surface;
        vtm_file = nothing,
        kwargs...)
        at_surface = Dict()
        for (k, v) in kwargs
            at_surface[k] = (
                size(v, 1) > size(surf.stereolitography.points, 2) ?
                surf(v) : v
            )
        end

        stl2vtk(fname, surf.stereolitography, vtm_file;
            at_surface...)
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator from a meshless domain to an underlying octree mesh.
    """
    Interpolator(domain::Domain, msh::Mesh; linear::Bool = false,) = Interpolator(
        domain, permutedims(msh.centers); linear = linear, from_surface = false
    )

    """
    $TYPEDSIGNATURES

    Run partial derivative operator with a given order of differentiation per axis
    """
    (part::Partition{N})(u::AbstractArray{Float64}, inds::Int64...) where {N} = (
        part.derivatives[inds](u)
    )

    """
    $TYPEDSIGNATURES

    Obtain JST sensor at cloud nodes.
    """
    function JST_sensor(part::Partition{N}, u::AbstractArray) where {N}
        ϵ = eps(Float64) |> sqrt

        ν = similar(u)
        ν .= ϵ

        for dim = 1:N
            lap = part.numerical_laplacian[dim]

            Δu = lap(u; Δ = true)
            ν .= max.(
                ν,
                abs.(Δu) ./ (lap(u; Δ = true, f = x -> abs.(x)) .+ ϵ)
            )
        end

        ν
    end

    """
    $TYPEDSIGNATURES

    Obtain JST-KE type artificial dissipation using a given spectral radius,
    along dimension `dim`. The dissipation is summed along all dimensions
    if `dim = 0`.
    """
    function artificial_dissipation(
        part::Partition{N}, u::AbstractArray, λ::AbstractVector,
        dim::Int = 0
    ) where {N}
        if dim == 0
            return sum(
                d -> artificial_dissipation(part, u, λ, d), 1:N
            )
        end

        lap = part.numerical_laplacian[dim]
        ϵ = eps(Float64) |> sqrt

        Δu = lap(u; Δ = true)
        ν = max.(
            ϵ,
            abs.(Δu) ./ (lap(u; Δ = true, f = x -> abs.(x)) .+ ϵ)
        )

        Δu .* λ .* ν
    end

    """
    $TYPEDSIGNATURES

    Obtain timescale (local time-step for CFL = 1) 
    for advective equations given one or more spectral
    radii for the flux Jacobian (vector or matrix, with
        each column as a component).
    """
    function timescale(part::Partition{N}, λ::AbstractVecOrMat) where {N}
        dt = similar(λ, (size(λ, 1),))
        dt .= Inf64
        ϵ = eps(eltype(dt))

        for dim = 1:N
            λi = (
                λ isa AbstractMatrix ?
                view(λ, :, dim) :
                λ
            )
            lap = part.numerical_laplacian[dim]

            dt .= min.(
                dt, 1.0 ./ (lap(λi; f = x -> abs.(x)) .+ ϵ)
            )
        end

        dt
    end

    """
    $TYPEDSIGNATURES

    Perform Laplacian smoothing iteration at partition.
    Eq. to `part.smoothing(u)`
    """
    smoothing(part::Partition{N}, u::AbstractArray) where {N} = part.smoothing(u)

end # module Meshless
