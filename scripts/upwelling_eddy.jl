using Oceananigans
using Oceananigans.Units

Lx = 5000
Ly = 10000
Lz = 500

Nx = 64
Ny = 192
Nz = 12

underlying_grid = RectilinearGrid(size = (Nx, Ny, Nz),
                                  topology = (Bounded, Periodic, Bounded),
                                  x = (-Lx, 0),
                                  y = (-Ly/2, Ly/2),
                                  z = ExponentialDiscretization(Nz, -Lz, 0),
                                  halo = (5, 5, 5))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> -1000*(1 - 1/((abs(x)/1000)^2+1)/((abs(y)/1000)^2+1))))

v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.00025))
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
pressure_solver = ConjugateGradientPoissonSolver(grid)

model = NonhydrostaticModel(grid;
                            advection = WENO(order=7),
                            coriolis = FPlane(latitude = 45),
                            boundary_conditions = (; v = v_bcs),
                            tracers = (:T, :S),
                            pressure_solver,
                            buoyancy = SeawaterBuoyancy())

set!(model, T = (x, y, z) -> 10 + max(-2, 2 * z/50), u = (args...) -> 0.01*randn()) #v = V_along)

simulation = Simulation(model, Δt = 10minutes, stop_time = 1days)

conjure_time_step_wizard!(simulation)

prog(sim) = @info prettytime(sim) * " in " * prettytime(sim.run_wall_time) * " with Δt = " * prettytime(sim.Δt)

add_callback!(simulation, prog, IterationInterval(100))

ζ = ∂x(model.velocities.v) - ∂y(model.velocities.u)
∇uₕ = ∂x(model.velocities.u) + ∂y(model.velocities.v)

simulation.output_writers[:tracers] = 
    JLD2Writer(model, merge(model.velocities, model.tracers, (; ζ, ∇uₕ)),
               filename = "eddy_$(Nx)_$(Ny).jld2",
               schedule = TimeInterval(1000),
               overwrite_existing = true)


