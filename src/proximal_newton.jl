#### Functions for quick minimization of quadratics and general (twice-differentiable)
# convex functions with bound constraints.

using LinearAlgebra, SparseArrays
using OSQP

""" x -> 0.5 x^T Q x - b^T x. """
struct Quadratic{M,V}
    Q :: M
    b :: V
end

(q :: Quadratic)(x) = 0.5*dot(x,q.Q*x) - dot(q.b,x)
grad(q :: Quadratic, x) = q.Q*x - q.b

""" Solve min_{l ≤ x ≤ u} 0.5x^TQx - b^Tx using OSQP."""
function min_bound_constrained_quadratic_osqp(Q :: Quadratic, l, u)
    n = length(l)
    if minimum(eigvals(Q.Q)) < 0.0 # nonconvex? bail?
        return zeros(n), false
    else
        model = OSQP.Model()

        OSQP.setup!(model; P = sparse(Q.Q), q = collect(-Q.b), A = sparse(I, n,n), l = collect(l), u = collect(u), polish = true, verbose = false)
        results = OSQP.solve!(model)
        results.x, true
    end
end

"""
    Solve the QP
    0.5*x'*Q*x + b'*x s.t. x_i = v_i for i in mask.
"""
function _solve_eq_qp(Q :: Quadratic, mask, values)
    Q_new = _fix_values(Q, mask, values)
    try
        x_free = Q_new.Q\Q_new.b
        x = deepcopy(values)
        ind_free = 1
        for i in eachindex(x)
            if !mask[i]
                x[i] = x_free[ind_free]
                ind_free+=1
            end
        end
        return x, true
    catch e
        return deepcopy(values), false
    end
end


function _fix_values(Q_st :: Quadratic, mask, values)
     n = length(Q_st.b)
     Q = Q_st.Q
     not_mask = (!).(mask)
     Q_new = Q[not_mask, not_mask]
     b_new = Q_st.b[not_mask]- Q[not_mask,:]*values #?????
     Quadratic(Q_new, b_new)
end

"""Attempts to solve min 0.5 x^TQx +b^Tx s.t. l ≤ x ≤ u by (repeatedly) guessing the active set.
Returns (x, f) where f is true if the KKT conditions are satisfied. """
function active_set_min_bound_constrained_quadratic(Q_st :: Quadratic, l, u, max_iters = 10, tol=1E-10; mask_l = falses(length(l)), mask_u = falses(length(u)))
    Q,b = Q_st.Q, -Q_st.b
    n = length(b)
    #if mask_l[i], x_i is constrained to be l_i
    #if mask_u[i], x_i is constrained to be u_i

    for iter in 1:max_iters
        # set up and solve equality-constrained QP

        eq_mask = falses(n)
        values = zeros(n)
        for i in 1:n
            if mask_u[i]
                eq_mask[i] = true
                values[i] = u[i]
            elseif mask_l[i]
                eq_mask[i] = true
                values[i] = l[i]
            end
        end

        x, eq_flag = _solve_eq_qp(Q_st, eq_mask, values)
        if !eq_flag
            return (zeros(n), false)
        end

        #compute gradient
        g = grad(Q_st, x)
        #check kkt conditions
        if all(((x,l,u),) -> l-tol < x < u+tol, zip(x, l, u)) # primal feasibility
            ### compute dual variables by enforcing stationarity...
            # and check complementary slackness
            comp_slack = true
            for i in eachindex(x)
                if g[i] < -tol
                    if abs(x[i] - u[i]) > tol
                        comp_slack = false
                        mask_u[i] = false
                        mask_l[i] = false
                    end
                elseif g[i] > tol
                    if abs(x[i] - l[i]) > tol
                        comp_slack = false
                        mask_u[i] = false
                        mask_l[i] = false
                    end
                end
            end
            if comp_slack
                return (x, true)
            end
        else # Not primal feasible, add constraints...
            for i in 1:n
                if x[i] < l[i] - tol
                    mask_l[i] = true
                    mask_u[i] = false
                end
                if x[i] > u[i] + tol
                    mask_u[i] = true
                    mask_l[i] = false
                end
            end
        end
    end
    return (zeros(n), false)
end

function min_bound_constrained_quadratic(Q_st :: Quadratic, l, u)
    # try netwon-type method.
    @assert all(l .< u)
    (r, flag) = active_set_min_bound_constrained_quadratic(Q_st :: Quadratic, l, u)
    if !flag
        r, flag  = min_bound_constrained_quadratic_osqp(Q_st, l, u)
    end
    r, flag
end

function form_quadratic_from_least_squares(A,b)
    Quadratic(A'*A, A'*b)
end

""" Simple proximal-newton method to solve

    min_x f(x)
    s.t. l ≤ x ≤ u

    returns (x,f) where f is true if the method succeeded.

    fgh(x) should return (f(x), g, H)
    where g and H are the gradient and hessian of f.
"""
function bounded_proximal_newton(fgh, x, l, u, iters, ftol_rel, gtol_abs)
    v_old = Inf
    @assert all(l .< u)
    @assert !any(isnan.(x))
    for i in 1:iters
        v,g,H = fgh(x)

        if minimum(eigvals(H)) ≤ 0.0 # Problem is not locally convex. 
            return x, false
        end

        if sum(abs2, g) ≤ gtol_abs # clipped/projected gradient?
            return x, true
        end

        if v > v_old + 1E-10 || any(isnan.(x))
            return x, false
        end

        if (v_old-v)/v < ftol_rel
            return x, true
        end

        v_old = v
        f_hat = Quadratic(H, -g)
        delta_x, flag = min_bound_constrained_quadratic(f_hat, l-x, u-x)
        if !flag
            return x, false
        end
        x = x + delta_x
    end
    x, false
end
