# If the number of sources is ≤ MAX_STATIC_K
# call static methods (to evaluate the function value and derivatives).
# If the number of sources is more than MAX_STATIC_K use dynamic methods.
const MAX_STATIC_K = 15
const MAX_NEWTON_K = 5


""" Wrapper around a small static array representing the loss function
x -> sum(abs2, x-y)."""
struct SquaredLoss{P_1, P_2, T}
    y :: SMatrix{P_1,P_2,Float64,T}
end

(l :: SquaredLoss)(lambda) :: Float64 = sum(abs2, lambda - l.y)

grad(l :: SquaredLoss, o) = 2.0*(o - l.y)

struct PatchLocalizer{T,S}
    lmo :: T
    model :: S
    function PatchLocalizer(lmo :: LMO, model :: ForwardModel)
        new{typeof(lmo), typeof(model)}(lmo, model)
    end
end

PatchLocalizer(model :: ForwardModel) = PatchLocalizer(LMO(model), model)

"""
    (p :: PatchLocalizer)(target :: SMatrix, max_iters :: Int64, min_gap :: Float64;
        sources :: Vector{PointSource} = PointSource[], max_score :: Float64 = Inf)

    Estimate the number and locations of point sources within the small image-patch `target`.
    `min_gap` is used to control the number of sources: if the drop in the squared loss
    after adding an additional source is less than `min_gap` the previous sources are returned.

    `lmo` is an instance of [`LMO`](#) and is used to estimate single sources.

    For larger images (bigger than, say, 30 by 30) see [`Localizer`](#).

"""
function (p :: PatchLocalizer)(target :: SMatrix, max_iters :: Int64, min_gap :: Float64; sources :: Vector{PointSource} = PointSource[], max_score :: Float64 = Inf)
    loss = SquaredLoss(target)
    lmo = p.lmo
    model = p.model
    for iter in 1:max_iters
        sources = PointSource[s for s in sources if s.intensity > 0.0]
        o = model(sources)
        old_loss = loss(o)

        old_loss < min_gap && return sources

        loss_gradient = grad(loss, o)
        score, source = lmo(loss_gradient)

        (score > max_score || score > 0.0) && return sources

        source = fit_single_weight(model, source, loss_gradient)
        old_sources = deepcopy(sources)
        push!(sources, source)

        sources = nonconvex(loss, model, sources)
        gap = old_loss - loss(model(sources))
        gap < min_gap && return old_sources
    end
    #@warn "Hit max iters in ADCG."
    return sources
end

function fit_single_weight(model, new_source, gradient)
    this_source = PointSource(1.0, new_source.x, new_source.y) |> model
    t = dot(this_source,-gradient)/(2*dot(this_source, this_source))
    PointSource(min(max(t, model.min_intensity), model.max_intensity), new_source.x, new_source.y)
end

function nonconvex(lossfn :: SquaredLoss, s :: ForwardModel{P_1, P_2}, initial_sources) where {P_1, P_2}
    k = length(initial_sources)
    if k == 0
        PointSource[]
    elseif k ≤ MAX_STATIC_K
        _nonconvex_impl(lossfn, s, SVector{k}(initial_sources)) :: Vector{PointSource}
    else
        _nonconvex_impl(lossfn, s, initial_sources) :: Vector{PointSource}
    end
end

function _nonconvex_impl(lossfn :: SquaredLoss, s :: ForwardModel{P_1, P_2}, initial_sources :: SVector{k}) where {k, P_1, P_2}
        initial_x = _vectorize(initial_sources) #reinterpret(Float64, Vector(initial_sources)) |> collect
        bounds = SVector{3*k}(vcat(fill((s.min_intensity,s.max_intensity), k),fill((0.0,s.widths[1]),k), fill((0.0,s.widths[2]),k) )) #repeat(SVector((s.min_intensity,s.max_intensity),(0.0,s.widths[1]), (0.0,s.widths[2])), k)

        initial_x = clamp.(initial_x, first.(bounds), last.(bounds))

        fgh = StaticNCVXObjective{k, typeof(s), typeof(lossfn)}(s, lossfn)

        flag = false
        if k ≤ MAX_NEWTON_K
            s_init = SVector{3*k}(initial_x)

            x_prox, flag, msg = bounded_proximal_newton(fgh, s_init, first.(bounds), last.(bounds), 30, 1E-7, 1E-7, x->_f(fgh,x))#1E-10, 1E-12, x->_f(fgh,x))
        end
        if !flag #true
            opt = Opt(:LD_SLSQP, length(initial_x))
            opt.lower_bounds = getindex.(bounds,1)
            opt.upper_bounds = getindex.(bounds,2)

            function fg!(x, g_storage)
                v, g = _fg(fgh, SVector{3*k}(x))
                if length(g_storage) != 0
                    g_storage .= g
                end
                v
            end

            opt.min_objective = fg!
            opt.ftol_rel = 1E-8
            (minf,x,ret) = NLopt.optimize(opt, initial_x)

            w, x, y = _devectorize(SVector{3*k}(x))
            r = [PointSource(w,x,y) for (w,x,y) in zip(w,x,y)]
            Vector(r)
        else
            w, x, y = _devectorize(x_prox)
            r = [PointSource(w,x,y) for (w,x,y) in zip(w,x,y)]
            Vector(r)
        end
end

function _nonconvex_impl(lossfn :: SquaredLoss, s :: ForwardModel{P_1, P_2}, initial_sources) where {P_1, P_2}
        k = length(initial_sources)
        initial_x = reinterpret(Float64, Vector(initial_sources)) |> collect
        bounds = repeat([(s.min_intensity,s.max_intensity),(0.0,s.widths[1]), (0.0,s.widths[2])], k)
        initial_x = clamp.(initial_x, first.(bounds), last.(bounds))

        fg = NCVXObjective(s,lossfn)

        opt = Opt(:LD_SLSQP, length(initial_x))
        opt.lower_bounds = getindex.(bounds,1)
        opt.upper_bounds = getindex.(bounds,2)

        function fg!(x, g_storage)
            v, g = fg(x)
            if length(g_storage) != 0
                g_storage .= g
            end
            v
        end

        opt.min_objective = fg!
        opt.ftol_rel = 1E-8
                   
        (minf,x,ret) = NLopt.optimize(opt, initial_x)
        reinterpret(PointSource, x) |> copy
end

function weights_only(lossfn :: SquaredLoss, s, initial_sources)
    k = length(initial_sources)
    if k == 0
        return PointSource[]
    end

    f_x = s.psf_1.(getproperty.(initial_sources, :x))
    f_y = s.psf_2.(getproperty.(initial_sources, :y))

    K_x = [dot(a,b) for a in f_x, b in f_x]
    K_y = [dot(a,b) for a in f_y, b in f_y]

    b = [dot(f_x, lossfn.y*f_y) for (f_x, f_y) in zip(f_x, f_y)]

    w, flag = min_bound_constrained_quadratic(Quadratic(K_x .* K_y, b), fill(s.min_intensity,k), fill(s.max_intensity,k))
    @assert flag

    [PointSource(w,p.x, p.y) for (w,p) in zip(w, initial_sources)]
end

include("static_objective.jl")
include("dynamic_objective.jl")