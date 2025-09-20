@info "Running RAE-2822 test case..."

path = "rae2822/"

stl = Stereolitography(path * "airfoil.dat")

msh = FixedMesh(
    [-20.0, -20.0], [40.0, 40.0],
    ("wall", stl, 1e-2);
    refinement_regions = [
        Triangulation(
            feature_edges(stl; radius = 0.05, angle = 45.0)
        ) => 1e-3
    ],
    farfield_boundaries = [
        "inlet" => [
            (1, false), # fwd face, first dimension (x)
            (2, false), # left face, second dimension (y)
            (2, true), # right face, second dimension (y)
        ],
        "outlet" => [(1, true)]
    ],
    verbose = true,
    growth_ratio = 1.1,
)

graph = Graph(
    msh;
    verbose = true,
    boundary_layers = [
        "wall" => BoundaryLayer(1e-5, 1.1),
    ]
)

domain = Domain(graph; partition_size = 10000,
    verbose = true)

partid = zeros(length(domain))
domain(partid) do part, partid
    partid .= Float64(part.id)
end

uv = zeros(length(domain), 2)
uv[:, 1] .= 1.0

domain(uv) do part, uv
    impose_bc!(
        part, "wall", uv
    ) do bdry, uv
        uv .* 0.0
    end
end
uvmag = sum(uv .^ 2; dims = 2) |> vec |> x -> sqrt.(x)

uvmag_smooth = copy(uvmag)
domain(uvmag_smooth) do part, uv
    uv .= smoothing(part, uv)
end

u = domain.graph.points[:, 2] .+ domain.graph.points[:, 1] .* 0.1
ux = similar(u)
domain(u, ux) do part, u, ux
    ux .= part(u, 1, 0)
end

dissipation = similar(uvmag)
domain(uvmag, dissipation) do part, u, dissipation
    dissipation .= JST_sensor(part, u)
end

λ = similar(dissipation)
λ .= 1.0
dt = similar(λ)
domain(λ, dt) do part, λ, dt
    dt .= timescale(part, λ)
end

vtk = vtk_grid(path * "graph", graph; id = partid, uv = uvmag, u = u, ux = ux, 
    dissipation = dissipation, dt = dt, uvmag_smooth = uvmag_smooth)
vtk_save(vtk)

surf = Surface(domain, stl)
vtk = vtk_grid(path * "wall", surf; id = partid, uv = uvmag,
    dissipation = dissipation, dt = dt)
vtk_save(vtk)

intp = Interpolator(domain, msh)
vtk = vtk_grid(path * "mesh", msh; id = intp(partid), uv = intp(uvmag), u = intp(u), ux = intp(ux),
    dissipation = intp(dissipation), dt = intp(dt), uvmag_smooth = intp(uvmag_smooth))
vtk_save(vtk)
