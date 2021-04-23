######
# This file deals with localizing point sources in large images.
# It exports the ImageLocalizer struct. ImageLocalizer is a bit complicated, but in short
# it processes a large image by first making a cheap, rough estimate of all point sources in the image
#   (techically it looks for local maxima of (x,y) ↦ ⟨ψ(x,y), img⟩ using a coarse grid + local newton).
# It then (approximately) solves a greedy set cover problem to place small boxes to cover all of the rough points.
# Each of those boxes is then processed with ADCG; the resulting points are returned.

struct ImageLocalizer{P1, P2, LMO_T}
    patch_localizer :: PatchLocalizer{LMO_T, ForwardModel{P1, P2}}
    prop_min_intensity :: Float64
    adcg_min_gap :: Float64
    adcg_max_iters :: Int64
end

"""
    ImageLocalizer(σ, adcg_min_gap, adcg_max_iters = 20, P1 = 16, P2 = 16)

   Used to estimate the number and locations of point sources within large images.

   `σ` is the width of the PSF (in pixels).

    `adcg_min_gap` is used to control the number of sources: if the drop in the squared loss
    after adding an additional source is less than `adcg_min_gap` the previous sources are returned.

    Uses a [`PatchLocalizer`](@ref) to localize within small `P1` x `P2` pixel patches.

"""
function ImageLocalizer(σ, adcg_min_gap, adcg_max_iters = 20, P1 = 16, P2 = 16)
    @assert P1 == P2
    @assert iseven(P1)
    model = ForwardModel(σ, (float(P1-1), float(P2-1)), (P1-1, P2-1), 0.0, Inf)
    ImageLocalizer(PatchLocalizer(model), adcg_min_gap, adcg_max_iters)
end


function ImageLocalizer(patch_localizer :: PatchLocalizer{LMO_T, ForwardModel{P1, P2}}, adcg_min_gap, adcg_max_iters = 20) where {P1, P2, LMO_T}
    @assert P1 == P2
    @assert iseven(P1)
    t_sq = sum(abs2, patch_localizer.model(PointSource(1.0, (P1 -1)/2, (P2 -1)/2)))
    # t_sq /= 2
    ImageLocalizer(patch_localizer, sqrt(adcg_min_gap*t_sq), adcg_min_gap, adcg_max_iters)
end

(l :: ImageLocalizer)(img) = first(boxes_and_locs(l, img))

function boxes_and_locs(l :: ImageLocalizer{P1, P2}, img) where {P1, P2}
    boxes, rough_points = preprocess_img(l.patch_localizer.model, img, l.prop_min_intensity)
    r = PointSource[]
    r_weights = Float64[]
    r_boxes = Tuple{Box, Vector{PointSource}}[]

    for box in boxes
        image_coords, box_coords = process_box(img, box, rough_points, l)
        append!(r, image_coords)
        append!(r_weights, [dist_to_center(p, box) for p in image_coords])
        push!(r_boxes, (box, box_coords))
    end

    # cluster duplicate localizations and pick localization closest to center of box.
    clusters = _cluster_points_idxs(loc_to_point.(r), 0.5)
    [r[c[argmin(r_weights[c])]] for c in clusters], img, r_boxes, rough_points, boxes
end

#TODO: Figure out when, if ever, we should warmstart here
function process_box(img, box, initial_points, localizer :: ImageLocalizer{P1, P2}, (l_y, u_y, l_x, u_x) = (1.0, 1.0, 1.0, 1.0)) where {P1, P2}
    patch_localizer = localizer.patch_localizer
    model = patch_localizer.model
    # collect points in box. there is a better way to do this.
    filtered = PointSource[]
    for p in initial_points
        x,y = image_coords_to_box_coords(p[1], p[2], box)
        if 0.0 <= x <= P1-1 && 0.0 <= y <= P2-1
            push!(filtered, PointSource(1.0, x,y))
        end
    end

    l = SquaredLoss(SMatrix{P1-1, P2-1}(@views img[box.lowerleft[1]:box.upperright[1],box.lowerleft[2]:box.upperright[2]]))

    # locs = weights_only(l,model, filtered)
    # locs = nonconvex(l,model, locs)
    # locs = patch_localizer(l.y, localizer.adcg_max_iters, localizer.adcg_min_gap; sources = locs, max_score = -localizer.prop_min_intensity)
    locs = patch_localizer(l.y, localizer.adcg_max_iters, localizer.adcg_min_gap; max_score = -localizer.prop_min_intensity)

    # filter for sources away from the edge...
    locs = [p for p in locs if (l_y < p.x < (P1-1)-u_y && l_x < p.y < (P2-1)-u_x)]

    PointSource[PointSource(p.intensity, box_coords_to_image_coords(p.x, p.y, box)...) for p in locs], locs
end

### TODO: The offset (1.5,-0.5) is mysterious.
function rough_points(model :: ForwardModel{P_1, P_2}, img, values, min_brightness = 125.0) where {P_1, P_2}
    r_1 = (P_1 -1) >> 1
    r_2 = (P_2 -1) >> 1
    proposals = SVector{2,Float64}[]
    ind = CartesianIndices(img)
    for (linearI, F) in enumerate(values)
        #TODO: Min brightness should probably be min brighness for an offset of 0.5 px?
        if F > min_brightness # This is conservative. Could check the actual newton extrapolation.
            I = Tuple(ind[linearI])
            if r_1 < I[1] < size(img, 1) - r_1 && r_2 < I[2] < size(img, 2) - r_2
                target = SMatrix{P_1-1, P_2-1}(@view img[I[1]-r_1:I[1]+r_1,I[2]-r_2:I[2]+r_2])
                p_local = float.(SVector(r_1, r_2)) .+ (1.5, 0.5)
                p, v, flag = _newton_LMO(p_local, -target, model.psf_1, model.psf_2, -F, 1.0, 20)
                if flag
                    push!(proposals, Float64.(Tuple(I)) .+ p-p_local .+ (0.5,-0.5))
                end
            end
        end
    end
    proposals
end

struct Box
    lowerleft :: NTuple{2, Int64}
    upperright :: NTuple{2, Int64}
end

Box(a :: SVector, b) = Box(Tuple(a), Tuple(b))

@inline center(b :: Box) = (b.lowerleft .- 1.0) .+ (b.upperright .- b.lowerleft .+ 1.0)./2

@inline function _px_to_box(p_ij, radius)
    bottom_left = p_ij .- radius
    Box(bottom_left, bottom_left.+2 .*radius)
end

@inline function dist_to_center(p :: PointSource, b :: Box)
    c = center(b)
    (p.x - c[1])^2 + (p.y - c[2])^2
end

@inline _point_to_px(p) = ceil.(Int64, p)

function preprocess_img(model :: ForwardModel{P1, P2}, img, min_intensity = 0.1, min_point_radius = 1.0) where {P1, P2}
    r = floor((P1 - 1) >> 1 - model.psf_1.sigma)
    br = ((P1 - 1) >> 1, (P2 - 1) >> 1)
    points, cps = _points_only(model, img, min_intensity, min_point_radius)
    box_centers = setcover_boxes(points, br, size(img) .- br, r)
    boxes = _px_to_box.(box_centers, (br,))
    boxes, cps
end

function _points_only(model, img, min_intensity = 0.1, min_point_radius = 1.0)
    grid_values = coarsegridvalues(img, model)
    points = rough_points(model, img, grid_values, min_intensity)
    points, mean_of_clusters(points, min_point_radius)
end

function _cluster_points_idxs(points, radius)
    tree = KDTree(points, Euclidean())
    neighbs = inrange(tree, points, radius)
    edges = LightGraphs.SimpleGraphs.SimpleEdge{Int64}[]
    if !isempty(neighbs)
        sizehint!(edges, sum(length, neighbs))
    end
    for (point_idx, neighbors) in enumerate(neighbs)
        for neighbor_idx in neighbors
            push!(edges, Edge(point_idx, neighbor_idx))
        end
    end
    g = SimpleGraph(edges)
    connected_components(g)
end

cluster_points(points, radius) =
    [points[c] for c in _cluster_points_idxs(points, radius)]

mean_of_clusters(points, min_r) = map(mean, cluster_points(points, min_r))

@inline function box_coords_to_image_coords(x,y, box)
    x_off = box.lowerleft[1]-1.0 #?
    y_off = box.lowerleft[2]-1.0 #?
    x + x_off, y + y_off
end

@inline function image_coords_to_box_coords(x,y, box)
    x_off = box.lowerleft[1]-1.0 #?
    y_off = box.lowerleft[2]-1.0 #?
    x - x_off, y - y_off
end

function coarsegridvalues(img, model:: ForwardModel{P1, P2}) where {P1, P2}
    f_1 = centered(model.psf_1((P1-1)/2) |> collect)
    f_2 = centered(model.psf_2((P2-1)/2) |> collect)
    ImageFiltering.imfilter(img, (f_1, f_2'))
end


#### set cover

@inline function _size_of_relative_complement(s, exclude)
    r = 0
    for y in s
        r += ifelse(y ∉ exclude, 1, 0)
    end
    r
end

""" Given a vector of subsets of {1, …, n} return an array r such that
r[i] = [j s.t. i ∈ sets[j]]. """
function invert_map(sets, n)
    r = [Int64[] for _ in 1:n]
    for (s_i, s) in enumerate(sets)
        for atom in s
            push!(r[atom], s_i)
        end
    end
    r
end

""" Given a collection of subsets of {1, …, n} compute a set cover.
Returns a partial cover if the union doesn't cover {1, …, n}.
    Optional third argument offsets the score of each set."""
function greedy_set_cover(sets, n, score_offsets = zeros(length(sets)))
    pq = PriorityQueue{Int,Float64}(Base.Order.Reverse)
    for (set_i, set) in enumerate(sets)
        pq[set_i] = length(set) + score_offsets[set_i]
    end

    set_inds = Int64[]
    covered = Set{Int64}()
    atoms_to_sets = invert_map(sets, n)
    while true
        isempty(pq) && break
        length(covered) == n && break
        s_index, score = dequeue_pair!(pq)
        score ≤ 0 && break
        # Add set index to result.
        push!(set_inds,s_index)

        old_covered = deepcopy(covered)
        # Update covered atoms.
        for i in sets[s_index]
            push!(covered,i)
        end

        # Update the size of set complements -
        #   but only for sets that contain an atom of sets[s_index] that was not previously covered.
        for i in sets[s_index]
            if i ∉ old_covered
                for set_index in atoms_to_sets[i]
                    pq[set_index] = _size_of_relative_complement(sets[set_index], covered) + score_offsets[set_index]
                end
            end
        end
    end
    set_inds
end

#TODO: Un-hardcode the offsets here... Should be an option in ImageLocalizer
function setcover_boxes(proposed_points, l,u, dist_from_center = 5.0, offsets = ((-3,0.0), (3,0.0), (0, 0.25)))
    t = KDTree(proposed_points, Chebyshev());
    box_centers = Tuple{SVector{2,Int64}, Float64}[]
    sizehint!(box_centers, (length(offsets)^2)*length(proposed_points))
    for p in proposed_points
        for (x,s_x) in offsets, (y,s_y) in offsets
            center = _point_to_px(p + SVector(x,y))
            if all(l .< center .< u )
                push!(box_centers, (center, s_x+s_y))
            end
        end
    end
    ir = inrange(t, first.(box_centers), dist_from_center)
    # the lengths here should be at least 1 by construction, no?
    filtered = [(f,s,x) for ((f,s),x) in zip(box_centers,ir) if length(x) > 0]
    boxes, subsets, scores = first.(filtered), getindex.(filtered,3), getindex.(filtered,2)
    boxes[greedy_set_cover(subsets, length(proposed_points),scores)]
end
