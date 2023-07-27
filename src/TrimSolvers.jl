"""
    Trim solvers using various methods. All solvers expect the system to be of the form 
    `res = system(x, u)`. `x` is the state vector, `u` is the control vector 
    and `res` is the residual vector.
"""

module TrimSolvers
export trim_newton

using LinearAlgebra

"""
    Implements Gauss-Newton fixed-point iteration method for a system that could be under-determined, determined or over-determined.
    For the over-determined case, provide a weighing matrix.
"""
function trim_newton(system, x0, u0, nres::Int;
        u_min=[0.0], u_max=[0.0], w=[0],
        relx=0.0, perturb_scale=0.005, max_iter=500,
        res_tol=1e-6, u_tol=1e-6, boundary_control=false,
        log_file=false)

    u_history = []
    res_history = []

    u_current = u0
    n = length(u0)
    Jac = zeros(nres, n)

    if log_file
        fh = open("trim.log", "w")
    end

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

        if log_file
            @info "Writing to trim.log ..."
            println(fh, "i = $i")
            println(fh, "u = $(u_current)")
            println(fh, "res = $(res_current)")
            println(fh, "")
            flush(fh)
        end

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

        u_next = u_current - pinv(Jac; w=w) * res_current

        if norm(u_next-u_current) < u_tol
            break
        end

        # Arbitrarily set back value 5% from boundary if boundary is violated
        if boundary_control
            for iu = 1:n
                if u_next[iu] > u_max[iu]
                    u_next[iu] -= range_u[iu]*0.05
                end
                if u_next[iu] < u_min[iu]
                    u_next[iu] += range_u[iu]*0.05
                end
            end
        end

        u_current = u_next*(1.0-relx) + u_current*relx
    end

    if log_file
        close(fh)
    end
    return u_history, res_history
end

"""
    Computes the pseudo-inverse for matrix A. Also known as the Moore-Penrose inverse.
"""
function pinv(A; w=[0.0])
    m, n = size(A)

    if m < n
        # Under-determined system (Right inverse)
        if norm(w) < 1e-6
            inv_A = A' * inv(A * A')
        else
            inv_W = Diagonal(w.^(-1))
            inv_A = inv_W * A' * inv(A * inv_W * A')
        end

    elseif m > n
        # Over-determined system (Left inverse)
        if norm(w) < 1e-6
            inv_A = (A' * A) \ A'
        else
            W = Diagonal(w)
            inv_A = (A' * W * A) \ A' * W
        end

    else
        # Determined system
        inv_A = inv(A)

    end
    return inv_A
end

end
