"""
    Used to approximately compute
    argmin_(x,y) ⟨ ψ(x,y), img ⟩
    using a grid + newton.

    ψ is the psf.

    Exploits the fact that the PSF is separable, but is NOT implemented as a convolution.
    Much more efficient for smaller image patches.
"""
struct LMO{P_1, P_2, T1, T2, N1, N2}
    s :: ForwardModel{P_1, P_2}
    grid_1 :: T1
    positions_1 :: SVector{N1, Float64}
    grid_2 :: T2
    positions_2 :: SVector{N2, Float64}
end


LMO(s :: ForwardModel{P_1, P_2}) where {P_1, P_2} = LMO(P_1, P_2,s)

LMO(n_grid:: Int64, s :: ForwardModel) = LMO(n_grid, n_grid, s)

function LMO(n_grid_1:: Int64, n_grid_2 :: Int64, s :: ForwardModel{P_1, P_2}) where {P_1, P_2}
    grid_p1 = collect(range(0.0, stop = s.widths[1], length=n_grid_1))
    grid_p2 = collect(range(0.0, stop = s.widths[2], length=n_grid_2))

    LMO(grid_p1, grid_p2, s)
end

function LMO(grid_p1, grid_p2, s :: ForwardModel{P_1, P_2}) where {P_1, P_2}
    grid_1 = reshape(reinterpret(Float64, [s.psf_1(x) for x in grid_p1]), P_1-1, :) |> collect
    grid_2 = reshape(reinterpret(Float64, [s.psf_2(x) for x in grid_p2]), P_2-1, :) |> collect

    LMO(s, SMatrix{size(grid_1,1), size(grid_1,2)}(grid_1), SVector{length(grid_p1)}(grid_p1),
        SMatrix{size(grid_2,1),size(grid_2,2)}(grid_2),  SVector{length(grid_p2)}(grid_p2))
end

"""
    Compute
    argmin_(x,y) ⟨ ψ(x,y), target ⟩
    using a grid + newton.
"""
function (l::LMO)(target)
    # PSF is separable.
    grid_scores = l.grid_1'*target*l.grid_2

    # for some reason argmin is super slow.
    score = Inf
    m_i = -1
    @inbounds for i in eachindex(grid_scores)
        c = grid_scores[i] < score
        score = ifelse(c, grid_scores[i], score)
        m_i = ifelse(c, i, m_i)
    end
    I = Tuple(CartesianIndices(grid_scores)[m_i])
    c_1, c_2 = l.positions_1[I[1]], l.positions_2[I[2]]
    # try newton's method to improve the estimate
    p, v, flag = _newton_LMO(SVector(c_1, c_2), target, l.s.psf_1, l.s.psf_2, score, 2.0, 20)
    if flag
        v, PointSource(1.0, p[1], p[2])
    else
        score, PointSource(1.0, c_1, c_2)
    end
end

# TODO: Change radius to l_∞ instead of l_2 ?
""" Newton's method to find a local minimum of (x,y) ↦ ⟨ ψ(x,y), window ⟩. """
function _newton_LMO(p :: SVector{2, Float64}, window, s_x :: GaussPSF, s_y :: GaussPSF, min_v, radius, max_iters=10)
    v = Inf
    p_zero = p
    r_sq = radius*radius
    for i in 1:max_iters
        # compute function value, gradient, and hessian
        r_x, d_x, dd_x = derivatives(s_x,p[1])
        r_y, d_y, dd_y = derivatives(s_y,p[2])

        w_r_y = window*r_y
        w_d_y = window*d_y

        v = dot(r_x, w_r_y)

        # check if we've strayed too far..
        d = p-p_zero
        if i > 1 && (v > min_v || dot(d, d) > r_sq)
          return p, v, false
        end

        g_x = dot(d_x, w_r_y)
        g_y = dot(r_x, w_d_y)
        g = SVector(g_x, g_y)

        H_xx = dot(dd_x, w_r_y)
        H_yy = dot(r_x, window*dd_y)
        H_xy = dot(d_x, w_d_y)
        H_asym = SMatrix{2,2}(H_xx, H_xy, H_xy, H_yy)
        H = Hermitian(H_asym)

        # Check hessian positive semi-definite
        lam = eigvals(H)
        if lam[1] < 0.0 || lam[2] < 0.0
          return p, v, false
        end
        if norm(g) < 1E-5
            break
        end
        delta = H_asym\g
        p = p - delta
    end
     p, v, true
end
