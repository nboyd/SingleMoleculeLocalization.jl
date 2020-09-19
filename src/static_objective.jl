### The next ~200 lines of code compute the objective value, gradient, and hessian.
# They are awful, but pretty fast.

#TODO: Cache some of the repeated tensor contractions

struct StaticNCVXObjective{K, S, T}
    m :: S
    l :: T
end

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

        exs = [:(PointSource(x[$((i-1)*3 + 1)],x[$((i-1)*3 + 2)], x[$((i-1)*3 + 3)])) for i in 1:k]

        quote
            @inline function _devectorize(x :: SVector{$(3*k)})
                         $(Expr(:tuple, exs...))
                     end
        end |> eval


        ex1 = Expr(:tuple, [:(x[$i][1]) for i in 1:k]...)
        ex2 = Expr(:tuple, [:(x[$i][2]) for i in 1:k]...)
        ex3 = Expr(:tuple, [:(x[$i][3]) for i in 1:k]...)

        quote
            @inline function _unpack_derivatives(x :: NTuple{$k})
                (
                    $ex1,
                    $ex2,
                    $ex3
                )
            end
        end |> eval

        linearizer = LinearIndices((3, k))
        exs = vcat([[:(gw[$i]),:(gx[$i]),:(gy[$i])] for i in 1:k]...)

        quote
            @inline function _interleave_gradient(gw :: SVector{$k}, gx, gy)
                SVector{$(3*k)}($(exs...))
            end
        end |> eval

        linearizer = LinearIndices((3, k))
        pairs = []
        for i in 1:k
            for j in 1:k
                push!(pairs, ((linearizer[1,j],linearizer[1,i]),:(H_ww[$j,$i])))
                push!(pairs, ((linearizer[2,j],linearizer[2,i]),:(H_xx[$j,$i])))
                push!(pairs, ((linearizer[3,j],linearizer[3,i]),:(H_yy[$j,$i])))

                push!(pairs, ((linearizer[2,j],linearizer[1,i]),:(H_xw[$j,$i])))
                push!(pairs, ((linearizer[1,i],linearizer[2,j]),:(H_xw[$j,$i])))

                push!(pairs, ((linearizer[3,j],linearizer[1,i]),:(H_yw[$j,$i])))
                push!(pairs, ((linearizer[1,i],linearizer[3,j]),:(H_yw[$j,$i])))

                push!(pairs, ((linearizer[2,j],linearizer[3,i]),:(H_xy[$j,$i])))
                push!(pairs, ((linearizer[3,i],linearizer[2,j]),:(H_xy[$j,$i])))
            end
        end
        pairs = [(LinearIndices((3k, 3k))[i,j], v) for ((i,j), v) in pairs]
        @assert length(unique(first.(pairs))) == 9*k*k
        sorted_exs = last.(sort(pairs))


        quote
            @inline function _interleave_hessian(H_ww :: SMatrix{$k,$k},H_xx, H_yy, H_xw, H_yw, H_xy)
                SMatrix{$(3*k), $(3*k)}($(sorted_exs...))
            end
        end |> eval

    end
end

function _fg(p :: StaticNCVXObjective{k}, x_vec) where {k}
    sources = _devectorize(x_vec)

    w = SVector(getproperty.(sources, :intensity))
    target = p.l.y

    sx = getproperty.(sources,:x)
    sy = getproperty.(sources,:y)

    d_x = derivatives.((p.m.psf_1,),sx)
    d_y = derivatives.((p.m.psf_2,),sy)


    f_x, dx, ddx = _unpack_derivatives(d_x)
    f_y, dy, ddy = _unpack_derivatives(d_y)

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
    r + 0.5*sum(abs2, target), _interleave_gradient(g_w, g_x, g_y)
end

function (p :: StaticNCVXObjective{k})(x_vec) where {k}
    sources = _devectorize(x_vec)

    w = SVector(getproperty.(sources, :intensity))
    target = p.l.y

    sx = getproperty.(sources,:x)
    sy = getproperty.(sources,:y)

    d_x = derivatives.((p.m.psf_1,),sx)
    d_y = derivatives.((p.m.psf_2,),sy)


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
    H = _interleave_hessian(H_ww,H_xx, H_yy, H_xw, H_yw, H_xy)
    r + 0.5*sum(abs2, target), SVector(g), Symmetric(SMatrix(H))
end
