pdom = begin
    @info "Testing linear dissipation..."

    if !isdir("dissipation")
        mkdir("dissipation")
    end

    lower = Stereolitography(
        [0.0 1.0; 0.0 0.0], [2; 1;;]
    )

    graph, coarse_graphs = multigrid(
        2,
        [0.0, 0.0], [1.0, 1.0],
        ("lower", lower, 2f-2);
        refinement_regions = [
            Ball([0.0, 0.0], 0.1) => 1e-2,
            Ball([1.0, 0.0], 0.1) => 1e-2,
            Ball([0.0, 1.0], 0.1) => 2e-2,
            Ball([1.0, 1.0], 0.1) => 2e-2,
        ],
        interior_reference = [0.5, 0.5],
        hypercube_families = [
            "others" => [(1, false), (1, true), (2, true)],
        ],
        growth_ratio = 1.1,
        verbose = true,
    )

    dom = PartitionedDomain(graph; max_partition_size = 1000,
        verbose = true, workers = workers,)

    u = zeros(Float32, length(dom))
    ν = ones(Float32, length(dom))

    ω = 0.5f0
    march! = u -> dom(u, ν) do dom, u, ν 
        D = similar(u)
        r = similar(u)
        D .= 0
        r .= 0

        νf = max.(
            at_owners(dom, ν), at_neighbors(dom, ν)
        )

        D .= - green_gauss(dom, νf ./ dom.face_distances)

        ∇uf = face_gradient(dom, u)        
        r .= divergent(
            dom, ∇uf...
        )

        @. u -= r / D * ω

        bdry = dom.boundaries["lower"]
        at_boundary(bdry, u) .= 1

        bdry = dom.boundaries["others"]
        at_boundary(bdry, u) .= 0

        ;
    end

    for _ = ProgressBar(1:5000)
        march!(u)
    end

    intp = Interpolator(graph, graph.octree)

    vtk_grid("dissipation", graph.octree; u = intp(u),) |> vtk_save
    vtk_grid("dissipation/graph", graph; u = u,) |> vtk_save
    export_surfaces("dissipation_surfaces", dom; u = u,)

    coarseners, prolongators = coarseners_and_prolongators(graph, coarse_graphs...)

    coarse = coarse_graphs[end]
    for coars in coarseners
        global u
        u = coars(u)
    end
    vtk_grid("dissipation/coarse", coarse; u = u) |> vtk_save

    pdom
end
Base.GC.gc()
