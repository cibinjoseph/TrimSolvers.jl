"""
    Trim solvers using various methods. All solvers expect the system to be of the form 
    `res = system(x, u)`. `x` is the state vector, `u` is the control vector 
    and `res` is the residual vector.
"""

module TrimSolvers
export trim_newton

using LinearAlgebra

"""
    Implements Gauss-Newton fixed-point iteration method
    for a system that could be over-determined or under-determined.
    This means the no. of residuals need not be same as controls.
"""
function trim_newton(system, x0, u0, nres::Int;
        u_min=[0.0], u_max=[0.0], relx=0.0, perturb_scale=0.005, max_iter=500,
        res_tol=1e-12, u_tol=1e-12)

    u_history = []
    res_history = []

    u_current = u0
    n = length(u0)
    Jac = zeros(nres, n)

    # Perturbation magnitude as a percentage of the range
    range_u = u_max-u_min
    perturb_vec = zeros(Float64, n)
    valid_range = false
    if norm(range_u) > eps(Float64)
        valid_range = true
        perturb_vec = range_u * perturb_scale
    end

    for i = 1:max_iter
        push!(u_history, u_current)
        res_current = system(x0, u_current)
        push!(res_history, res_current)

        if norm(res_current) < res_tol
            break
        end

        if i == max_iter
            @warn "Failed to converge in max_iter iterations."
            break
        end

        # Compute Jacobian using secant method
        for iu = 1:n
            u_perturb = deepcopy(u_current)
            if valid_range
                h = perturb_vec[iu]
            else
                h = u_perturb[iu] * perturb_scale
            end
            u_perturb[iu] += h
            res = system(x0, u_perturb)
            Jac[:, iu] = (res-res_current)./h
        end

        u_next = u_current - Jac \ res_current

        if norm(u_next-u_current) < u_tol
            break
        end

        u_current = u_next*(1.0-relx) + u_current*relx
    end

    return u_history, res_history
end
end
