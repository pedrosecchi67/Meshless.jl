pdom = begin
    @info "Testing linear advection along diagonal..."

    if !isdir("advection")
        mkdir("advection")
    end

    upper = Stereolitography(
        [0.0 0.0; 0.0 1.0], [1; 2;;]
    )
    lower = Stereolitography(
        [0.0 1.0; 0.0 0.0], [2; 1;;]
    )

    graph, coarse_graphs = multigrid(
        5,
        [0.0, 0.0], [1.0, 1.0],
        ("upper", upper, 1f-2),
        ("lower", lower, 1f-2);
        refinement_regions = [
            Line([0.0, 0.0], [1.0, 1.0]) => 2f-2,
        ],
        interior_reference = [0.5, 0.5],
        hypercube_families = [
            "outlet" => [(1, true), (2, true)],
        ],
        boundary_layers = [
            "upper" => 2f-3,
            "lower" => 2f-3,
        ],
        verbose = true,
    )

    pdom = PartitionedDomain(graph; verbose = true, max_partition_size = 2000,
        workers = workers,)
    coarse_doms = [
        PartitionedDomain(
            g; verbose = true, max_partition_size = 2000,
            workers = workers,
        ) for g in coarse_graphs
    ]

    coarseners, prolongators = coarseners_and_prolongators(graph, coarse_graphs...)

    uv = zeros(Float32, length(pdom), 2)
    C = ones(Float32, length(pdom), 2)

    march! = uv -> begin
        ω = 0.5f0

        S = similar(uv)
        S .= 0
        Ss = [S]

        f = (l, uv; second_order::Bool = false,) -> begin
            dom = (
                l == 0 ?
                pdom : coarse_doms[l]
            )
        
            duv = similar(uv)
            duv .= 0

            s = Ss[l + 1]

            dom(uv, duv, s) do dom, uv, duv, s 
                D = similar(uv)
                r = similar(uv)
                D .= 0
                r .= 0

                Cf = at_faces(dom, C) .* dom.face_normals |> x -> sum(x; dims = 2) |> vec

                uL = at_owners(dom, uv)
                uR = at_neighbors(dom, uv)
                if second_order
                    uL, uR = MUSCL(dom, uv)
                end

                ϕ = @. uL * (Cf + abs(Cf)) / 2 + uR * (Cf - abs(Cf)) / 2

                r .= green_gauss(dom, ϕ)
                D .= green_gauss(dom, (Cf .+ abs.(Cf)) ./ 2)

                @. duv = - r / D + s
                uvnew = @. uv + duv

                bdry = dom.boundaries["upper"]
                at_boundary(bdry, uvnew) .= [1 0]

                bdry = dom.boundaries["lower"]
                at_boundary(bdry, uvnew) .= [0 1]

                bdry = dom.boundaries["outlet"]
                at_boundary(bdry, uvnew) .= at_pivots(bdry, uv)

                @. duv = (uvnew - uv)
            end

            (duv, ω)
        end

        S .= f(0, uv; second_order = true)[1] .- f(0, uv; second_order = false)[1]
        for coarsener in coarseners
            push!(Ss, coarsener(Ss[end]))
        end

        FAS!(f, uv; 
            coarseners = coarseners,
            prolongators = prolongators,)

        uv
    end

    for _ = ProgressBar(1:20)
        march!(uv)
    end

    vtk_grid("advection/graph", graph; uv = uv',) |> vtk_save

    pdom
end
Base.GC.gc()
