using Oceananigans
#using CUDA
using Oceananigans.Units

arch = CPU()

const Lx = 10000
const Ly = 10000
const Lz = 1000

Nx = arch isa GPU ? 128 : 64#64
Ny = arch isa GPU ? 128 : 64#192
Nz = arch isa GPU ? 16 : 12

underlying_grid = RectilinearGrid(arch,
                                  size = (Nx, Ny, Nz),
                                  topology = (Bounded, Periodic, Bounded),
                                  x = (-Lx, 0),
                                  y = (-Ly/2, Ly/2),
                                  z = ExponentialDiscretization(Nz, -Lz, 0),
                                  halo = (5, 5, 5))

hill(x, y) = 100-1100*(1 - 1/((abs(x)/1000)^2+1)/((abs(y)/1000)^2+1))
shelf(x, y) = -200 + 800*(tanh((x+2000)/500)-1)/2

grid = ImmersedBoundaryGrid(underlying_grid, 
                            GridFittedBottom((x, y) -> max(hill(x, y), shelf(x, y))))

linear_drag(x, y, t, u, C) = - C * u
linear_drag(x, y, z, t, u, C) = - C * u

bottom_friction(component) = FluxBoundaryCondition(linear_drag,
                                                   field_dependencies = component,
                                                   parameters = 0.001)

τ = 0.00025
coriolis = FPlane(latitude = 45)

ν = 1e-2#1e-4

D = π * √(2 * ν/abs(coriolis.f))

V₀ = √(2)*π*τ/D/coriolis.f

ekman_transport_field = Field{Nothing, Face, Center}(grid)
set!(ekman_transport_field, (y, z) -> -V₀ * cos(π / 4 + π * z / D) * exp(π * z / D))

u_bcs = FieldBoundaryConditions(bottom = bottom_friction(:u),
                                immersed = bottom_friction(:u),
                                west = OpenBoundaryCondition(ekman_transport_field; 
                                                             scheme = PerturbationAdvection(inflow_timescale = 10minutes,
                                                                                            outflow_timescale = 1day)))

v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τ),
                                bottom = bottom_friction(:v),
                                immersed = bottom_friction(:v))

T_nudging = Relaxation(; rate = 1/2hours, 
                         target = (x, y, z, t) -> 10 + 4*max(-1, z/100), 
                         mask = (x, y, z) -> exp(-(x+Lx)/(Lx/20)))

using Oceananigans.Solvers: ConjugateGradientPoissonSolver
pressure_solver = ConjugateGradientPoissonSolver(grid; maxiter = 10)

model = NonhydrostaticModel(grid;
                            advection = WENO(order=7),
                            coriolis,
                            forcing = (; T = T_nudging),
                            boundary_conditions = (; u = u_bcs, v = v_bcs),
                            tracers = (:T, :S),
                            pressure_solver,
                            buoyancy = SeawaterBuoyancy())

set!(model, T = (x, y, z) -> 10 + 4*max(-1, z/100), u = (args...) -> 0.01*randn()) #v = V_along)

simulation = Simulation(model, Δt = 1, stop_time = 1days)

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


