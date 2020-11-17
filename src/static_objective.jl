### The next ~200 lines of code compute the objective value, gradient, and hessian.
# They are awful, but pretty fast.
struct StaticNCVXObjective{K, S, T}
    m :: S
    l :: T
end

_vectorize(s :: SVector{K, PointSource}) where {K} = 
	vcat(getproperty.(s, :intensity), getproperty.(s, :x), getproperty.(s, :y))

_interleave_gradient(gw, gx, gy) = vcat(gw,gx,gy)

# Eval a bunch of utiltiy functions for various numbers of sources (up to MAX_STATIC_K)
for k in 1:MAX_STATIC_K
    let k = k
        exs = vec([quote dot(v[$i],v[$j]) end  for i in 1:k, j in 1:k])
        quote
            @inline function _k(v :: NTuple{$k})
                    SMatrix{$k,$k, Float64}($(exs...))
            end
        end |> eval

        exs = vec([:(dot(v_1[$i],v_2[$j]))  for i in 1:k, j in 1:k])
        quote
            @inline function _k(v_1 :: NTuple{$k}, v_2 :: NTuple{$k})
                    SMatrix{$k,$k}($(exs...))
                end
        end |> eval

        exs = vec([:(dot(v_1[$i],M*v_2[$i]))  for i in 1:k])
        quote
            @inline function _m_dot(v_1 :: NTuple{$k}, M, v_2 :: NTuple{$k})
                    SVector{$k}($(exs...))
            end
        end|> eval

        quote
            @inline function _diagm(v :: SVector{$k, Float64})
                   v.*$(SMatrix{k,k,Float64}(I))
            end
        end |> eval


        linearizer = LinearIndices((k, 3))
        ex_w = [:(x[$(linearizer[i,1])]) for i in 1:k]
        ex_x = [:(x[$(linearizer[i,2])]) for i in 1:k]
        ex_y = [:(x[$(linearizer[i,3])]) for i in 1:k]
  		
        quote
            @inline function _devectorize(x :: SVector{$(3*k)})
                         SVector($(ex_w...)),SVector($(ex_x...)), SVector($(ex_y...))
                     end
        end |> eval

        ex1 = [:(x[$i][1]) for i in 1:k]
        ex2 = [:(x[$i][2]) for i in 1:k]
        ex3 = [:(x[$i][3]) for i in 1:k]
        
        quote
            @inline function _unpack_derivatives(x :: SVector{$k})
                (
                    tuple($(ex1...)),
                    tuple($(ex2...)),
                    tuple($(ex3...))
                )
            end
        end |> eval
    end
end

function _f(p :: SingleMoleculeLocalization.StaticNCVXObjective{k}, x_vec :: SVector) where {k}
    w, sx, sy = _devectorize(x_vec)

    f_x = Tuple((p.m.psf_1).(sx))
    f_y = Tuple((p.m.psf_2).(sy))

    k_x = _k(f_x)
    k_y = _k(f_y)

    # function value
    r = 0.5*sum(w'*(k_x.*k_y)*w)
    l = -_m_dot(f_x, p.l.y, f_y)
    r += LinearAlgebra.dot(l,w)

    r + 0.5*sum(abs2, p.l.y)
end



function _fg(p :: StaticNCVXObjective{k}, x_vec) where {k}
    w, sx, sy = _devectorize(x_vec)
    
    # broadcast fails to infer!
    d_x = map(x->derivatives(p.m.psf_1,x), sx)
    d_y = map(y->derivatives(p.m.psf_2,y), sy) 
    
    f_x, dx, ddx = _unpack_derivatives(d_x)
    f_y, dy, ddy = _unpack_derivatives(d_y)

    k_x = _k(f_x)
    k_y = _k(f_y)

    # function value
    r = 0.5*sum(w'*(k_x.*k_y)*w)
    l = -_m_dot(f_x, p.l.y, f_y)
    r += dot(l,w)

    # gradients...
    g_w = l + (k_x.*k_y)*w
    g_x = (-w).*_m_dot(dx, p.l.y, f_y)
    d_k_x  = _k(dx, f_x)
    g_x += w.*((d_k_x.*k_y)*w)
    g_y = -w.*(_m_dot(f_x, p.l.y, dy))
    d_k_y = _k(dy, f_y)
    g_y += w.*((d_k_y.*k_x)*w)
    r + 0.5*sum(abs2, p.l.y), _interleave_gradient(g_w, g_x, g_y)
end

function (p :: StaticNCVXObjective{k})(x_vec_dyn) where {k}
    x_vec = SVector{3k, Float64}(x_vec_dyn)
    w, sx, sy = _devectorize(x_vec)
    target = p.l.y

    # broadcast fails to infer!
    d_x = map(x->derivatives(p.m.psf_1,x), sx)
    d_y = map(y->derivatives(p.m.psf_2,y), sy)

    f_x, dx, ddx = _unpack_derivatives(d_x)
    f_y, dy, ddy = _unpack_derivatives(d_y)

    _ncvx_impl(target, w, f_x, dx, ddx, f_y, dy, ddy)
end

# TODO: Cache reused tensor contractions
function _ncvx_impl(target, w :: SVector{k}, f_x, dx, ddx, f_y, dy, ddy) where {k}
    k_x = _k(f_x)
    k_y = _k(f_y)

    # function value
    r = 0.5*sum(w'*(k_x.*k_y)*w)
    l = -_m_dot(f_x, target, f_y)
    r += dot(l,w)

    # gradients...
    g_w = l + (k_x.*k_y)*w
    g_x = (-w).*_m_dot(dx, target, f_y)
    d_k_x  = _k(dx, f_x)
    g_x += w.*((d_k_x.*k_y)*w)
    g_y = -w.*(_m_dot(f_x, target, dy))
    d_k_y = _k(dy, f_y)
    g_y += w.*((d_k_y.*k_x)*w)

    # hessian
    d_d_x = _k(dx, dx)
    dd_x = _k(ddx, f_x)
    H_xx = (w*w').*d_d_x.*k_y
    H_xx += _diagm(w.*((k_y.*dd_x)*w) - w.*(_m_dot(ddx, target, f_y)))
    d_d_y = _k(dy, dy)
    dd_y = _k(ddy, f_y)
    H_yy = (w*w').*d_d_y.*k_x
    H_yy += _diagm(w.*((k_x.*dd_y)*w) - w.*(_m_dot(f_x, target, ddy)))
    H_ww = k_x.*k_y
    # off diagonals
    H_xy = (w*w').*(d_k_x.*(d_k_y'))
    H_xy += _diagm(w.*((d_k_x.*d_k_y)*w) - w.*(_m_dot(dx, target, dy)))
    H_xw = w.*(d_k_x.*k_y)
    H_xw += _diagm((d_k_x.*k_y)*w - _m_dot(dx, target, f_y))
    H_yw = w.*(d_k_y.*k_x)
    H_yw += _diagm((d_k_y.*k_x)*w - _m_dot(f_x, target, dy))
    g = _interleave_gradient(g_w, g_x, g_y)
    H = vcat(hcat(H_ww, H_xw', H_yw'), hcat(H_xw, H_xx, H_xy), hcat(H_yw, H_xy', H_yy))
    r + 0.5*sum(abs2, target), SVector(g), Symmetric(SMatrix(H))
end
