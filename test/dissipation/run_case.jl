@info "Running dissipation test case..."

path = "dissipation/"

one_surf = Stereolitography(
    [
        0.0 0.0;
        0.0 1.0
    ]; closed = false
)

zero_surf = Stereolitography(
    [
        0.0 1.0 1.0 0.0;
        0.0 0.0 1.0 1.0
    ]; closed = false
)

msh = FixedMesh(
    [0.0, 0.0], [1.0, 1.0],
    ("one_surf", one_surf, 0.01),
    ("zero_surf", zero_surf, 0.01),
    ;
    refinement_regions = [
        Ball([0.0, 0.0], 0.1) => 0.01, 
        Ball([0.0, 1.0], 0.1) => 0.01, 
    ],
    interior_point = [0.5, 0.5],
    verbose = true
)

graph = Graph(msh; verbose = true)

domain = Domain(graph; degree = 2,)

dt = 1e-5

step! = u -> begin
    domain(u) do part, u
        u .+= dt .* (part(u, 2, 0) .+ part(u, 0, 2))

        impose_bc!(part, "zero_surf", u) do bdry, u
            ub = similar(u)
            ub .= 0.0

            ub
        end
        impose_bc!(part, "one_surf", u) do bdry, u
            ub = similar(u)
            ub .= 1.0

            ub
        end
    end
end

u = zeros(length(domain))

for _ = 1:20
    step!(u)
end

intp = Interpolator(domain, msh)

vtk = vtk_grid(path * "mesh", msh; u = intp(u))
vtk_save(vtk)

vtk = vtk_grid(path * "graph", graph; u = u)
vtk_save(vtk)
