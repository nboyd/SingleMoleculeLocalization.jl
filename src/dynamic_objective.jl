struct NCVXObjective{P_1,P_2}
    m :: ForwardModel{P_1, P_2}
    l :: SquaredLoss
end

# TODO: Remove allocations?
function (p :: NCVXObjective)(x_vec)
    @inbounds begin
    sources = reinterpret(PointSource, x_vec) |> copy

    k = length(sources)

    w = getproperty.(sources, :intensity)

    target = p.l.y
    d_x = [derivatives(p.m.psf_1, s.x) for s in sources]
    d_y = [derivatives(p.m.psf_2, s.y) for s in sources]

    f_x, dx, ddx = getindex.(d_x, 1), getindex.(d_x, 2), getindex.(d_x, 3)
    f_y, dy, ddy = getindex.(d_y, 1), getindex.(d_y, 2), getindex.(d_y, 3)

    #### contract to form k_x and k_y (?)
    k_x = zeros(k,k)
    k_y = zeros(k,k)

    for i in 1:k
        for j in 1:k
            k_x[i,j] = dot(f_x[i], f_x[j])
        end
    end

    for i in 1:k
        for j in 1:k
            k_y[i,j] = dot(f_y[i], f_y[j])
        end
    end

    r = 0.0
    for i in 1:k, j in 1:k
        r += 0.5*w[i]*w[j]*k_x[i,j]*k_y[i,j]
    end

    l = zeros(k)

    ## Linear part.
    for i in 1:k
        l[i] = -f_x[i]'*target*f_y[i]
    end

    for i in 1:k
        r += l[i]*w[i]
    end

    ### Gradients...

    g_w = zero(w)
    copy!(g_w, l)
    for i in 1:k, j in 1:k
        g_w[i] += w[j]*k_x[i,j]*k_y[i,j]
    end

    g_x = zeros(k)
    #linear part
    for i in 1:k
        g_x[i] = -w[i]*dx[i]'*target*f_y[i]
    end
    # quadratic term...
    d_k_x  = zeros(k,k)
    for i in 1:k, j in 1:k
        d_k_x[i,j] = dx[i]'*f_x[j]
    end
    for i in 1:k, j in 1:k
        g_x[i] += w[i]*w[j]*d_k_x[i,j]*k_y[i,j]
    end

    g_y = zeros(k)
    # linear part
    for i in 1:k
        g_y[i] -= w[i]*f_x[i]'*target*dy[i]
    end
    # quadratic term...
    d_k_y = zeros(k,k)
    for i in 1:k
        for j in 1:k
            d_k_y[i,j] = dy[i]'*f_y[j]
        end
    end

    for i in 1:k, j in 1:k
        g_y[i] += w[i]*w[j]*d_k_y[i,j]*k_x[i,j]
    end



    g = zeros(3*k)

    linearizer = LinearIndices((3, k))
    for i in 1:k
        g[linearizer[1,i]] = g_w[i]
        g[linearizer[2,i]] = g_x[i]
        g[linearizer[3,i]] = g_y[i]
    end

    r + 0.5*sum(abs2, target), g
    end
end
