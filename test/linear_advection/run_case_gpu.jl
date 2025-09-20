using Meshless
using CUDA

@info "Running linear advection test case..."

path = ""

inlet_upper = Stereolitography(
    [
        0.0 0.0;
        0.0 1.0
    ]; closed = false
)

inlet_lower = Stereolitography(
    [
        0.0 1.0;
        0.0 0.0
    ]; closed = false
)

outlet = Stereolitography(
    [
        1.0 1.0 0.0;
        0.0 1.0 1.0
    ]; closed = false
)

msh = FixedMesh(
    [0.0, 0.0], [1.0, 1.0],
    ("inlet_upper", inlet_upper, 0.01),
    ("inlet_lower", inlet_lower, 0.01),
    ("outlet", outlet, 0.01),
    ;
    refinement_regions = [
        Line([0.0, 0.0], [1.0, 1.0]) => 0.01
    ],
    interior_point = [0.5, 0.5],
    verbose = true
)

graph = Graph(msh; verbose = true)

domain = Domain(graph; degree = 2,)

step! = u -> begin
    dt = domain(
        u;
        conv_to_backend = CuArray,
        conv_from_backend = x -> Array(x),
    ) do part, u
        λ = similar(u)
        λ .= 1.0

        timescale(part, λ) |> minimum |> x -> x * 0.5
    end |> minimum

    domain(
        u;
        conv_to_backend = CuArray,
        conv_from_backend = x -> Array(x),
    ) do part, u
        Cx = similar(u)
        Cx .= 1.0
        Cy = similar(u)
        Cy .= 1.0

        u .+= (
            artificial_dissipation(part, u, Cx, 1) .+
            artificial_dissipation(part, u, Cy, 2) .-
            (part(u, 1, 0) .* Cx .+ part(u, 0, 1) .* Cy)
        ) .* dt

        impose_bc!(part, "inlet_upper", u) do bdry, u
            ub = similar(u)
            ub .= 1.0

            ub
        end
        impose_bc!(part, "inlet_lower", u) do bdry, u
            ub = similar(u)
            ub .= 0.0

            ub
        end
        impose_bc!(part, "outlet", u) do bdry, u
            ub = copy(u)

            ub
        end
    end
end

u = zeros(length(domain))

for _ = 1:2000
    step!(u)
end

intp = Interpolator(domain, msh)

vtk = vtk_grid(path * "mesh", msh; u = intp(u))
vtk_save(vtk)

vtk = vtk_grid(path * "graph", graph; u = u)
vtk_save(vtk)
