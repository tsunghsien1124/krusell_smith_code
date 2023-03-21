
using Interpolations            # to use interpolation 
using Random, LinearAlgebra
using QuantEcon                 # to use `gridmake`, `<:AbstractUtility`
using Optim                     # to use minimization routine to maximize RHS of bellman equation
using GLM                       # to regress
using JLD2                      # to save the result
using ProgressMeter             # to show progress of iterations
using Parameters                # to use type with keyword arguments
using Plots

struct TransitionMatrix
    P::Matrix{Float64}          # 4x4 
    Pz::Matrix{Float64}         # 2x2 aggregate shock
    Peps_gg::Matrix{Float64}    # 2x2 idiosyncratic shock conditional on good to good
    Peps_bb::Matrix{Float64}    # 2x2 idiosyncratic shock conditional on bad to bad
    Peps_gb::Matrix{Float64}    # 2x2 idiosyncratic shock conditional on good to bad
    Peps_bg::Matrix{Float64}    # 2x2 idiosyncratic shock conditional on bad to good
end

abstract type UMPSolutionMethod end

@with_kw struct EulerMethod <: UMPSolutionMethod
    update_k::Float64 = 0.7
end

@with_kw struct VFI <: UMPSolutionMethod
    Howard_on::Bool = false
    Howard_n_iter::Int = 20
end

function create_transition_matrix(
    ug::Real, # unemployment rate in good state
    ub::Real, # unemployment rate in bad state
    zg_ave_dur::Real, # average duration of good state
    zb_ave_dur::Real, # average duration of bad state
    ug_ave_dur::Real, # average duration of unemployment in good state
    ub_ave_dur::Real, # average duration of unemployment in bad state
    puu_rel_gb2bb::Real, # prob. of u to u cond. on g to b relative to that of b to b
    puu_rel_bg2gg::Real) # prob. of u to u cond. on b to g relative to that of g to g

    # probability of remaining in good state
    pgg = 1 - 1 / zg_ave_dur
    # probability of remaining in bad state
    pbb = 1 - 1 / zb_ave_dur
    # probability of changing from g to b
    pgb = 1 - pgg
    # probability of changing from b to g
    pbg = 1 - pbb

    # prob. of 0 to 0 cond. on g to g (0 means being umemployed and 1 employed)
    p00_gg = 1 - 1 / ug_ave_dur
    # prob. of 0 to 0 cond. on b to b
    p00_bb = 1 - 1 / ub_ave_dur
    # prob. of 0 to 1 cond. on g to g
    p01_gg = 1 - p00_gg
    # prob. of 0 to 1 cond. on b to b
    p01_bb = 1 - p00_bb

    # prob. of 0 to 0 cond. on g to b
    p00_gb = puu_rel_gb2bb * p00_bb
    # prob. of 0 to 0 cond. on b to g
    p00_bg = puu_rel_bg2gg * p00_gg
    # prob. of 0 to 1 cond. on g to b
    p01_gb = 1 - p00_gb
    # prob. of 0 to 1 cond. on b to g
    p01_bg = 1 - p00_bg

    # prob. of 1 to 0 cond. on g to g
    p10_gg = (ug - ug * p00_gg) / (1 - ug)
    # prob. of 1 to 0 cond. on b to b
    p10_bb = (ub - ub * p00_bb) / (1 - ub)
    # prob. of 1 to 0 cond. on g to b
    p10_gb = (ub - ug * p00_gb) / (1 - ug)
    # prob. of 1 to 0 cond on b to g
    p10_bg = (ug - ub * p00_bg) / (1 - ub)
    # prob. of 1 to 1 cond. on g to g
    p11_gg = 1 - p10_gg
    # prob. of 1 to 1 cond. on b to b
    p11_bb = 1 - p10_bb
    # prob. of 1 to 1 cond. on g to b
    p11_gb = 1 - p10_gb
    # prob. of 1 to 1 cond on b to g
    p11_bg = 1 - p10_bg

    #   (g1)         (b1)        (g0)       (b0)
    P = [pgg*p11_gg pgb*p11_gb pgg*p10_gg pgb*p10_gb
        pbg*p11_bg pbb*p11_bb pbg*p10_bg pbb*p10_bb
        pgg*p01_gg pgb*p01_gb pgg*p00_gg pgb*p00_gb
        pbg*p01_bg pbb*p01_bb pbg*p00_bg pbb*p00_bb]
    Pz = [pgg pgb
        pbg pbb]
    Peps_gg = [p11_gg p10_gg
        p01_gg p00_gg]
    Peps_bb = [p11_bb p10_bb
        p01_bb p00_bb]
    Peps_gb = [p11_gb p10_gb
        p01_gb p00_gb]
    Peps_bg = [p11_bg p10_bg
        p01_bg p00_bg]
    transmat = TransitionMatrix(P, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat
end

function KSParameter(;
    beta::AbstractFloat=0.99,
    alpha::AbstractFloat=0.36,
    delta::Real=0.025,
    theta::Real=1,
    k_min::Real=0,
    k_max::Real=1000,
    k_size::Integer=100,
    K_min::Real=30,
    K_max::Real=50,
    K_size::Integer=4,
    z_min::Real=0.99,
    z_max::Real=1.01,
    z_size::Integer=2,
    eps_min::Real=0.0,
    eps_max::Real=1.0,
    eps_size::Integer=2,
    ug::AbstractFloat=0.04,
    ub::AbstractFloat=0.1,
    zg_ave_dur::Real=8,
    zb_ave_dur::Real=8,
    ug_ave_dur::Real=1.5,
    ub_ave_dur::Real=2.5,
    puu_rel_gb2bb::Real=1.25,
    puu_rel_bg2gg::Real=0.75,
    mu::Real=0,
    degree::Real=7)

    if theta == 1
        u = LogUtility()
    else
        u = CRRAUtility(theta)
    end
    l_bar = 1 / (1 - ub)
    # individual capital grid
    k_grid = (range(0, stop=k_size - 1, length=k_size) / (k_size - 1)) .^ degree * (k_max - k_min) .+ k_min
    k_grid[1] = k_min
    k_grid[end] = k_max # adjust numerical error
    # aggregate capital grid
    K_grid = range(K_min, stop=K_max, length=K_size)
    # aggregate technology shock
    z_grid = range(z_max, stop=z_min, length=z_size)
    # idiosyncratic employment shock grid
    eps_grid = range(eps_max, stop=eps_min, length=eps_size)
    s_grid = gridmake(z_grid, eps_grid)               # shock grid

    # collection of transition matrices
    transmat = create_transition_matrix(
        ug, ub,
        zg_ave_dur,
        zb_ave_dur,
        ug_ave_dur,
        ub_ave_dur,
        puu_rel_gb2bb,
        puu_rel_bg2gg)

    ksp = (u=u, beta=beta, alpha=alpha, delta=delta, theta=theta,
        l_bar=l_bar, k_min=k_min, k_max=k_max, k_grid=k_grid,
        K_min=K_min, K_max=K_max, K_grid=K_grid, z_grid=z_grid,
        eps_grid=eps_grid, s_grid=s_grid, k_size=k_size, K_size=K_size,
        z_size=z_size, eps_size=eps_size, s_size=z_size * eps_size,
        ug=ug, ub=ub, transmat=transmat, mu=mu)

    return ksp
end

r(alpha::Real, z::Real, K::Real, L::Real) = alpha * z * K^(alpha - 1) * L^(1 - alpha)
w(alpha::Real, z::Real, K::Real, L::Real) = (1 - alpha) * z * K^(alpha) * L^(-alpha)

mutable struct KSSolution
    k_opt::Array{Float64,3}
    value::Array{Float64,3}
    B::Vector{Float64}
    R2::Vector{Float64}
end

function KSSolution(ksp::NamedTuple;
    load_value::Bool=false,
    load_B::Bool=false,
    filename::String="result.jld2")
    /
    if load_value || load_B
        result = load(filename)
        kss_temp = result["kss"]
    end
    if load_value
        k_opt = kss_temp.k_opt
        value = kss_temp.value
    else
        k_opt = ksp.beta * repeat(ksp.k_grid, outer=[1, ksp.K_size, ksp.s_size])
        k_opt = 0.9 * repeat(ksp.k_grid, outer=[1, ksp.K_size, ksp.s_size])
        k_opt .= clamp.(k_opt, ksp.k_min, ksp.k_max)
        value = ksp.u.(0.1 / 0.9 * k_opt) / (1 - ksp.beta)
    end
    if load_B
        B = kss_temp.B
    else
        B = [0.0, 1.0, 0.0, 1.0]
    end
    kss = KSSolution(k_opt, value, B, [0.0, 0.0])
    return kss
end


function generate_shocks(ksp::NamedTuple;
    z_shock_size::Integer=1100,
    population::Integer=10000)

    # unpack parameters
    Peps_gg = ksp.transmat.Peps_gg
    Peps_bg = ksp.transmat.Peps_bg
    Peps_gb = ksp.transmat.Peps_gb
    Peps_bb = ksp.transmat.Peps_bb

    # draw aggregate shock
    zi_shock = simulate(MarkovChain(ksp.transmat.Pz), z_shock_size)

    ### Let's draw individual shock ###
    epsi_shock = Array{Int}(undef, z_shock_size, population) # preallocation

    # first period
    rand_draw = rand(population)
    # recall: index 1 of eps is employed, index 2 of eps is unemployed
    if zi_shock[1] == 1 # if good
        epsi_shock[1, :] .= (rand_draw .< ksp.ug) .+ 1 # if draw is higher, become employed 
    elseif zi_shock[1] == 2 # if bad
        epsi_shock[1, :] .= (rand_draw .< ksp.ub) .+ 1 # if draw is higher, become employed
    else
        error("the value of z_shocks[1] (=$(z_shocks[1])) is strange")
    end

    # from second period ...   
    for t = 2:z_shock_size
        draw_eps_shock!(Val(zi_shock[t]), Val(zi_shock[t-1]), view(epsi_shock, t, :), epsi_shock[t-1, :], ksp.transmat)
    end

    # adjustment
    for t = 1:z_shock_size
        n_e = count(epsi_shock[t, :] .== 1) # count number of employed
        empl_rate_ideal = ifelse(zi_shock[t] == 1, 1.0 - ksp.ug, 1.0 - ksp.ub)
        gap = round(Int, empl_rate_ideal * population) - n_e
        if gap > 0
            become_employed_i = rand(findall(2 .== epsi_shock[t, :]), gap)
            epsi_shock[t, become_employed_i] .= 1
        elseif gap < 0
            become_unemployed_i = rand(findall(1 .== epsi_shock[t, :]), -gap)
            epsi_shock[t, become_unemployed_i] .= 2
        end
    end

    return zi_shock, epsi_shock
end

draw_eps_shock!(zi::Val{1}, zi_lag::Val{1}, epsi, epsi_lag::AbstractVector, transmat::TransitionMatrix) = draw_eps_shock!(epsi, epsi_lag, transmat.Peps_gg)
draw_eps_shock!(zi::Val{1}, zi_lag::Val{2}, epsi, epsi_lag::AbstractVector, transmat::TransitionMatrix) = draw_eps_shock!(epsi, epsi_lag, transmat.Peps_bg)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{1}, epsi, epsi_lag::AbstractVector, transmat::TransitionMatrix) = draw_eps_shock!(epsi, epsi_lag, transmat.Peps_gb)
draw_eps_shock!(zi::Val{2}, zi_lag::Val{2}, epsi, epsi_lag::AbstractVector, transmat::TransitionMatrix) = draw_eps_shock!(epsi, epsi_lag, transmat.Peps_bb)

function draw_eps_shock!(
    epsi_shocks,
    epsi_shock_before,
    Peps::AbstractMatrix)

    # loop over entire population
    for i = 1:length(epsi_shocks)
        rand_draw = rand()
        epsi_shocks[i] = ifelse(epsi_shock_before[i] == 1,
            (Peps[1, 1] < rand_draw) + 1,  # if employed before
            (Peps[2, 1] < rand_draw) + 1)  # if unemployed before
    end

    return nothing
end

function solve_ump!(umpsm::EulerMethod,
    ksp::NamedTuple, kss::KSSolution;
    max_iter::Integer=10000,
    tol::AbstractFloat=1e-8)
    alpha, beta, delta, theta, l_bar, mu =
        ksp.alpha, ksp.beta, ksp.delta, ksp.theta, ksp.l_bar, ksp.mu
    k_grid, k_size = ksp.k_grid, ksp.k_size
    K_grid, K_size = ksp.K_grid, ksp.K_size
    s_grid, s_size = ksp.s_grid, ksp.s_size
    k_min, k_max = ksp.k_min, ksp.k_max
    global counter = 0
    k_opt_n = similar(kss.k_opt)
    prog = ProgressThresh(tol, "Solving individual UMP by Euler method: ")
    while true
        global counter += 1
        for s_i = 1:s_size
            z, eps = s_grid[s_i, 1], s_grid[s_i, 2]
            for (K_i, K) = enumerate(K_grid)
                Kp, L = compute_Kp_L(K, s_i, kss.B, ksp)
                for (k_i, k) = enumerate(k_grid)
                    wealth = (r(alpha, z, K, L) + 1 - delta) * k +
                             w(alpha, z, K, L) * (eps * l_bar + mu * (1 - eps))
                    expec = compute_expectation_FOC(kss.k_opt[k_i, K_i, s_i], Kp, s_i, ksp)
                    cn = (beta * expec)^(-1.0 / theta)
                    k_opt_n[k_i, K_i, s_i] = wealth - cn
                end
            end
        end
        k_opt_n .= clamp.(k_opt_n, k_min, k_max)
        dif_k = maximum(abs, k_opt_n - kss.k_opt)
        ProgressMeter.update!(prog, dif_k)
        if dif_k < tol
            break
        end
        if counter >= max_iter
            @warn "Euler method failed to converge with 
            dif_k)"
            break
        end
        kss.k_opt .= umpsm.update_k * k_opt_n .+ (1 - umpsm.update_k) * kss.k_opt
    end
    return nothing
end

function compute_expectation_FOC(kp::Real,
    Kp::Real,
    s_i::Integer,
    ksp::NamedTuple)
    alpha, theta, delta, l_bar, mu, P =
        ksp.alpha, ksp.theta, ksp.delta, ksp.l_bar, ksp.mu, ksp.transmat.P
    global expec = 0.0
    for s_n_i = 1:ksp.s_size
        zp, epsp = ksp.s_grid[s_n_i, 1], ksp.s_grid[s_n_i, 2]
        Kpp, Lp = compute_Kp_L(Kp, s_n_i, kss.B, ksp)
        rn = r(alpha, zp, Kp, Lp)
        kpp = interpolate((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, s_n_i], Gridded(Linear()))
        cp = (rn + 1 - delta) * kp + w(alpha, zp, Kp, Lp) * (epsp * l_bar + mu * (1.0 - epsp)) - kpp(kp, Kp)
        global expec = expec + P[s_i, s_n_i] * (cp)^(-theta) * (1 - delta + rn)
    end
    return expec
end

function compute_Kp_L(K::Real, s_i::Integer,
    B::AbstractVector, ksp::NamedTuple)
    Kp, L = ifelse(s_i % ksp.eps_size == 1,
        (exp(B[1] + B[2] * log(K)), ksp.l_bar * (1 - ksp.ug)), # if good
        (exp(B[3] + B[4] * log(K)), ksp.l_bar * (1 - ksp.ub))) # if bad
    Kp = clamp(Kp, ksp.K_min, ksp.K_max)
    return Kp, L
end

function solve_ump!(umpsm::VFI,
    ksp::NamedTuple,
    kss::KSSolution;
    max_iter::Integer=100,
    tol::AbstractFloat=1e-8,
    print_skip::Integer=10)

    Howard, Howard_n_iter = umpsm.Howard_on, umpsm.Howard_n_iter
    global counter_VFI = 0  # counter
    prog = ProgressThresh(tol, "Solving individual UMP by VFI: ")
    while true
        global counter_VFI += 1
        value_old = copy(kss.value) # guessed value
        # maximize value for all state
        [maximize_rhs!(k_i, K_i, s_i, ksp, kss)
         for k_i in 1:ksp.k_size, K_i in 1:ksp.K_size, s_i in 1:ksp.s_size]
        # Howard's policy iteration
        !Howard || iterate_policy!(ksp, kss, n_iter=Howard_n_iter)
        # difference of guessed and new value
        dif = maximum(abs, value_old - kss.value)
        # progress meter of covergence process
        ProgressMeter.update!(prog, dif)
        # if difference is sufficiently small
        if dif == max_iter
            println("VFI reached its maximum iteration : $max_iter")
            break
        end
    end
end


function maximize_rhs!(
    k_i::Integer,
    K_i::Integer,
    s_i::Integer,
    ksp::NamedTuple,
    kss::KSSolution,
)
    # obtain minimum and maximum of grid
    k_min, k_max = ksp.k_grid[1], ksp.k_grid[end]

    # unpack parameters
    alpha, delta, l_bar, mu =
        ksp.alpha, ksp.delta, ksp.l_bar, ksp.mu

    # obtain state value
    k = ksp.k_grid[k_i]   # obtain individual capital value
    K = ksp.K_grid[K_i]   # obtain aggregate capital value
    z, eps = ksp.s_grid[s_i, 1], ksp.s_grid[s_i, 2]
    Kp, L = compute_Kp_L(K, s_i, kss.B, ksp) # next aggregate capital and current aggregate labor
    # if kp>k_c_pos, consumption is negative 
    k_c_pos = (r(alpha, z, K, L) + 1 - delta) * k +
              w(alpha, z, K, L) * (eps * l_bar + (1 - eps) * mu)
    obj(kp) = -rhs_bellman(ksp, kp, kss.value, k, K, s_i) # objective function
    res = optimize(obj, k_min, min(k_c_pos, k_max)) # maximize value
    # obtain result
    kss.k_opt[k_i, K_i, s_i] = Optim.minimizer(res)
    kss.value[k_i, K_i, s_i] = -Optim.minimum(res)
    return nothing
end


function rhs_bellman(ksp::NamedTuple,
    kp::Real, value::Array{Float64,3},
    k::Real, K::Real, s_i::Integer)
    u, s_grid, beta, alpha, l_bar, delta, mu =
        ksp.u, ksp.s_grid, ksp.beta, ksp.alpha, ksp.l_bar, ksp.delta, ksp.mu
    z, eps = s_grid[s_i, 1], s_grid[s_i, 2]
    Kp, L = compute_Kp_L(K, s_i, kss.B, ksp) # Next period aggregate capital and current aggregate labor
    c = (r(alpha, z, K, L) + 1 - delta) * k +
        w(alpha, z, K, L) * (eps * l_bar + (1.0 - eps) * mu) - kp # current consumption 
    expec = compute_expectation(kp, Kp, value, s_i, ksp)
    return u(c) + beta * expec
end

function compute_expectation(
    kp::Real,  # next period indicidual capital
    Kp::Real,  # next period aggragte capital
    value::Array{Float64,3}, # next period value
    s_i::Integer, # index of current state,
    ksp::NamedTuple)
    k_grid, K_grid = ksp.k_grid, ksp.K_grid # unpack grid
    beta, P = ksp.beta, ksp.transmat.P      # unpack parameters

    # compute expectations by summing up
    global expec = 0.0
    for s_n_i = 1:ksp.s_size
        value_itp = interpolate((k_grid, K_grid), value[:, :, s_n_i], Gridded(Linear()))
        global expec += P[s_i, s_n_i] * value_itp(kp, Kp)
    end
    return expec
end

function iterate_policy!(ksp::NamedTuple,
    kss::KSSolution; n_iter::Integer=20)
    value = similar(kss.value)
    for i = 1:n_iter
        # update value using policy
        value .=
            [rhs_bellman(ksp,
                kss.k_opt[k_i, K_i, s_i], kss.value,
                ksp.k_grid[k_i], ksp.K_grid[K_i], s_i)
             for k_i in 1:ksp.k_size,
             K_i in 1:ksp.K_size,
             s_i in 1:ksp.s_size]
        kss.value .= copy(value)
    end
    return nothing
end

abstract type SimulationMethod end

struct Stochastic <: SimulationMethod
    epsi_shocks::Matrix{Int}
    k_population::Vector{Float64}
end

Stochastic(epsi_shocks::Matrix{Int}) = Stochastic(epsi_shocks, fill(40, size(epsi_shocks, 2)))

struct NonStochastic <: SimulationMethod
    k_dens::Matrix{Float64}
end

function NonStochastic(ksp, zi)
    k_dens = zeros(ksp.k_size, ksp.eps_size)
    id = findlast(ksp.k_grid .< mean(ksp.K_grid))
    if zi == 1
        k_dens[id, 1] = ksp.ug
        k_dens[id, 2] = 1 - ksp.ug
    elseif zi == 2
        k_dens[id, 1] = ksp.ub
        k_dens[id, 2] = 1 - ksp.ub
    end
    return NonStochastic(k_dens)
end

function simulate_aggregate_path!(ksp::NamedTuple, kss::KSSolution,
    zi_shocks::AbstractVector, K_ts::Vector, sm::Stochastic)
    epsi_shocks, k_population = sm.epsi_shocks, sm.k_population

    T = length(zi_shocks)   # simulated duration
    N = size(epsi_shocks, 2) # number of agents

    # loop over T periods
    @showprogress 0.5 "simulating aggregate path ..." for (t, z_i) = enumerate(zi_shocks)
        K_ts[t] = mean(k_population) # current aggrgate capital

        # loop over individuals
        for (i, k) in enumerate(k_population)
            eps_i = epsi_shocks[t, i]   # idiosyncratic shock
            s_i = epsi_zi_to_si(eps_i, z_i, ksp.z_size) # transform (z_i, eps_i) to s_i
            # obtain next capital holding by interpolation
            itp_pol = interpolate((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, s_i], Gridded(Linear()))
            k_population[i] = itp_pol(k, K_ts[t])
        end
    end
    return nothing
end
epsi_zi_to_si(eps_i::Integer, z_i::Integer, z_size::Integer) = z_i + ksp.z_size * (eps_i - 1)

function simulate_aggregate_path!(ksp::NamedTuple, kss::KSSolution,
    zi_shocks::AbstractVector, K_ts::Vector, sm::NonStochastic)
    k_dens = sm.k_dens
    @showprogress 0.5 "simulating aggregate path ..." for (t, z_i) = enumerate(zi_shocks[1:end-1])
        k_dens_n = zeros(size(k_dens))
        K_ts[t] = dot(ksp.k_grid, k_dens[:, 1]) + dot(ksp.k_grid, k_dens[:, 2])
        Peps = get_Peps(Val(zi_shocks[t+1]), Val(z_i), ksp.transmat)
        for eps_i in 1:ksp.eps_size
            for (k_i, k) in enumerate(ksp.k_grid)
                s_i = epsi_zi_to_si(eps_i, z_i, ksp.z_size)
                itp = interpolate((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, s_i], Gridded(Linear()))
                kp = itp(k, K_ts[t])
                kpi_u = findfirst(kp .<= ksp.k_grid)
                kpi_l = findlast(kp .>= ksp.k_grid)
                weight_l = ifelse(kpi_u == kpi_l, 1,
                    (ksp.k_grid[kpi_u] - kp) / (ksp.k_grid[kpi_u] - ksp.k_grid[kpi_l]))
                for epsp_i in 1:ksp.eps_size
                    k_dens_n[kpi_u, epsp_i] += (1 - weight_l) * Peps[eps_i, epsp_i] * k_dens[k_i, eps_i]
                    k_dens_n[kpi_l, epsp_i] += weight_l * Peps[eps_i, epsp_i] * k_dens[k_i, eps_i]
                end
            end
        end
        k_dens_n .= k_dens_n / sum(k_dens_n)
        k_dens .= k_dens_n
    end
    K_ts[end] = dot(ksp.k_grid, k_dens[:, 1]) + dot(ksp.k_grid, k_dens[:, 2])
    return nothing
end
get_Peps(zpi::Val{1}, zi::Val{1}, transmat::TransitionMatrix) = transmat.Peps_gg
get_Peps(zpi::Val{1}, zi::Val{2}, transmat::TransitionMatrix) = transmat.Peps_bg
get_Peps(zpi::Val{2}, zi::Val{1}, transmat::TransitionMatrix) = transmat.Peps_gb
get_Peps(zpi::Val{2}, zi::Val{2}, transmat::TransitionMatrix) = transmat.Peps_bb

function regress_ALM!(ksp::NamedTuple, kss::KSSolution,
    zi_shocks::Vector, K_ts::Vector;
    T_discard::Integer=100)
    n_g = count(zi_shocks[T_discard+1:end-1] .== 1)
    n_b = count(zi_shocks[T_discard+1:end-1] .== 2)
    B_n = Vector{Float64}(undef, 4)
    x_g = Vector{Float64}(undef, n_g)
    y_g = Vector{Float64}(undef, n_g)
    x_b = Vector{Float64}(undef, n_b)
    y_b = Vector{Float64}(undef, n_b)
    global i_g = 0
    global i_b = 0
    for t = T_discard+1:length(zi_shocks)-1
        if zi_shocks[t] == 1
            global i_g = i_g + 1
            x_g[i_g] = log(K_ts[t])
            y_g[i_g] = log(K_ts[t+1])
        else
            global i_b = i_b + 1
            x_b[i_b] = log(K_ts[t])
            y_b[i_b] = log(K_ts[t+1])
        end
    end
    resg = lm([ones(n_g) x_g], y_g)
    resb = lm([ones(n_b) x_b], y_b)
    kss.R2 = [r2(resg), r2(resb)]
    B_n[1], B_n[2] = coef(resg)
    B_n[3], B_n[4] = coef(resb)
    dif_B = maximum(abs, B_n - kss.B)
    println("difference of ALM coefficient is B_n")
    return B_n, dif_B
end


function find_ALM_coef!(umpsm::UMPSolutionMethod, sm::SimulationMethod,
    ksp::NamedTuple, kss::KSSolution,
    zi_shocks::Vector{Int};
    tol_ump::AbstractFloat=1e-8,
    max_iter_ump::Integer=100,
    tol_B::AbstractFloat=1e-8,
    max_iter_B::Integer=20,
    update_B::AbstractFloat=0.3,
    T_discard::Integer=100)

    K_ts = Vector{Float64}(undef, length(zi_shocks))
    global counter_B = 0
    while true
        global counter_B = counter_B + 1
        println(" --- Iteration over ALM coefficient: $counter_B ---")

        # solve individual problem
        solve_ump!(umpsm, ksp, kss, max_iter=max_iter_ump, tol=tol_ump)

        # compute aggregate path of capital
        simulate_aggregate_path!(ksp, kss, zi_shocks, K_ts, sm)

        # obtain new ALM coefficient by regression
        B_n, dif_B = regress_ALM!(ksp, kss, zi_shocks, K_ts, T_discard=T_discard)

        # check convergence
        if dif_B < tol_B
            println("-----------------------------------------------------")
            println("ALM coefficient successfully converged : dif = $dif_B")
            println("-----------------------------------------------------")
            break
        elseif counter_B == max_iter_B
            println("----------------------------------------------------------------")
            println("Iteration over ALM coefficient reached its maximum ($max_iter_B)")
            println("----------------------------------------------------------------")
            break
        end

        # Update B
        kss.B .= update_B .* B_n .+ (1 - update_B) .* kss.B
    end
    return K_ts
end


function plot_ALM(z_grid::AbstractVector, zi_shocks::Vector,
    B::Vector, K_ts::Vector;
    T_discard::Integer=100)

    compute_approxKprime(K, z::Val{1}, B) = exp(B[1] + B[2] * log(K))
    compute_approxKprime(K, z::Val{2}, B) = exp(B[3] + B[4] * log(K))
    K_ts_approx = similar(K_ts) # preallocation

    # compute approximate ALM for capital
    K_ts_approx[T_discard] = K_ts[T_discard]

    for t = T_discard:length(zi_shocks)-1
        K_ts_approx[t+1] =
            compute_approxKprime(K_ts_approx[t], Val(zi_shocks[t]), B)
    end

    p = plot(T_discard+1:length(K_ts), K_ts[T_discard+1:end], lab="true", color=:red, line=:solid)
    plot!(p, T_discard+1:length(K_ts), K_ts_approx[T_discard+1:end], lab="approximation", color=:blue, line=:dash)
    title!(p, "aggregate law of motion for capital")
    return p
end

function plot_Fig1(ksp, kss, K_ts)
    K_min, K_max = minimum(K_ts), maximum(K_ts)
    K_lim = range(K_min, stop=K_max, length=100)
    Kp_g = exp.(kss.B[1] .+ kss.B[2] * log.(K_lim))
    Kp_b = exp.(kss.B[3] .+ kss.B[4] * log.(K_lim))

    p = plot(K_lim, Kp_g, linestyle=:solid, lab="Good")
    plot!(p, K_lim, Kp_b, linestyle=:solid, lab="Bad")
    plot!(p, K_lim, K_lim, color=:black, linestyle=:dash, lab="45 degree", width=0.5)
    title!(p, "FIG1: Tomorrow's vs. today's aggregate capital")
    return p
end

function plot_Fig2(ksp, kss, K_eval_point)
    k_lim = range(0, stop=80, length=1000)
    itp_e = interpolate((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, 1], Gridded(Linear()))
    itp_u = interpolate((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, 3], Gridded(Linear()))

    kp_e(k) = itp_e(k, K_eval_point)
    kp_u(k) = itp_u(k, K_eval_point)

    p = plot(k_lim, kp_e.(k_lim), linestyle=:solid, lab="employed")
    plot!(p, k_lim, kp_u.(k_lim), linestyle=:solid, lab="unemployed")
    plot!(p, k_lim, k_lim, color=:black, linestyle=:dash, lab="45 degree", width=0.5)
    title!(p, "FIG2: Individual policy function \n at K=$K_eval_point when good state")
    return p
end

# instance of KSParameter
ksp = KSParameter()
# instance of KSSolution
kss = KSSolution(ksp, load_value=false, load_B=false)
if size(kss.k_opt, 1) != length(ksp.k_grid)
    error("loaded data is inconsistent with k_size")
end
if size(kss.k_opt, 2) != length(ksp.K_grid)
    error("loaded data is inconsistent with K_size")
end

# generate shocks
Random.seed!(0) # for reproducability
@time zi_shocks, epsi_shocks = generate_shocks(ksp;
    z_shock_size=1100, population=10000);

#======================#
# find ALM coefficient #
#======================#
sm = Stochastic(epsi_shocks)
T_discard = 100
@time K_ts = find_ALM_coef!(EulerMethod(),
    sm, ksp, kss, zi_shocks,
    tol_ump=1e-8, max_iter_ump=10000,
    tol_B=1e-8, max_iter_B=500, update_B=0.3,
    T_discard=T_discard);

plot_ALM(ksp.z_grid, zi_shocks, kss.B, K_ts, T_discard=T_discard)

#kss.B  # Regression coefficient
println("Approximated aggregate capital law of motion")
println("log(K_{t+1})=$(kss.B[1])+$(kss.B[2])log(K_{t}) in good time (R2 = $(kss.R2[1]))")
println("log(K_{t+1})=$(kss.B[3])+$(kss.B[4])log(K_{t}) in bad time (R2 = $(kss.R2[2]))")

@save "result_Euler.jld2" ksp kss

# Compute mean of capital implied by regression
mc = MarkovChain(ksp.transmat.Pz)
sd = stationary_distributions(mc)[1]
logKg = kss.B[1] / (1 - kss.B[2])
logKb = kss.B[3] / (1 - kss.B[4])
meanK_reg = exp(sd[1] * logKg + sd[2] * logKb)
meanK_sim = mean(K_ts[T_discard+1:end])
println("mean of capital implied by regression is $meanK_reg")
println("mean of capital implied by simulation is $meanK_sim")

plot_Fig1(ksp, kss, K_ts)
plot_Fig2(ksp, kss, 40)

#=====================================#
# Solution with Young (2008)'s method #
#=====================================#
kss = KSSolution(ksp, load_value=false, load_B=false)
ns = NonStochastic(ksp, zi_shocks[1])
@time K_ts = find_ALM_coef!(EulerMethod(),
    ns, ksp, kss, zi_shocks,
    tol_ump=1e-8, max_iter_ump=10000,
    tol_B=1e-8, max_iter_B=500, update_B=0.3,
    T_discard=T_discard);

plot_ALM(ksp.z_grid, zi_shocks, kss.B, K_ts, T_discard=T_discard)

#kss.B  # Regression coefficient
println("Approximated aggregate capital law of motion")
println("log(K_{t+1})=$(kss.B[1])+$(kss.B[2])log(K_{t}) in good time (R2 = $(kss.R2[1]))")
println("log(K_{t+1})=$(kss.B[3])+$(kss.B[4])log(K_{t}) in bad time (R2 = $(kss.R2[2]))")

@save "result_Young.jld2" ksp kss

# Compute mean of capital implied by regression
mc = MarkovChain(ksp.transmat.Pz)
sd = stationary_distributions(mc)[1]
logKg = kss.B[1] / (1 - kss.B[2])
logKb = kss.B[3] / (1 - kss.B[4])
meanK_reg = exp(sd[1] * logKg + sd[2] * logKb)
meanK_sim = mean(K_ts[T_discard+1:end])
println("mean of capital implied by regression is $meanK_reg")
println("mean of capital implied by simulation is $meanK_sim")

plot_Fig1(ksp, kss, K_ts)
plot_Fig2(ksp, kss, 40)

#========================================#
# Solution with value function iteration #
#========================================#
kss = KSSolution(ksp, load_value=false, load_B=false)
ns = NonStochastic(ksp, zi_shocks[1])
@time K_ts = find_ALM_coef!(VFI(Howard_on=false, Howard_n_iter=20),
    ns, ksp, kss, zi_shocks,
    tol_ump=1e-8, max_iter_ump=10000,
    tol_B=1e-8, max_iter_B=500, update_B=0.3,
    T_discard=T_discard);

plot_ALM(ksp.z_grid, zi_shocks, kss.B, K_ts, T_discard=T_discard)

#kss.B  # Regression coefficient
println("Approximated aggregate capital law of motion")
println("log(K_{t+1})=$(kss.B[1])+$(kss.B[2])log(K_{t}) in good time (R2 = $(kss.R2[1]))")
println("log(K_{t+1})=$(kss.B[3])+$(kss.B[4])log(K_{t}) in bad time (R2 = $(kss.R2[2]))")

@save "result_VFI.jld2" ksp kss 


# Compute mean of capital implied by regression
mc = MarkovChain(ksp.transmat.Pz)
sd = stationary_distributions(mc)[1]
logKg = kss.B[1]/(1-kss.B[2])
logKb = kss.B[3]/(1-kss.B[4])
meanK_reg = exp(sd[1]*logKg + sd[2]*logKb)
meanK_sim = mean(K_ts[T_discard+1:end])
println("mean of capital implied by regression is $meanK_reg")
println("mean of capital implied by simulation is $meanK_sim")

plot_Fig1(ksp, kss, K_ts)
plot_Fig2(ksp, kss, 40)
