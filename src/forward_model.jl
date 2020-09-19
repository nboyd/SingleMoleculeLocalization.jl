"""
    The forward model for a single molecule localization microscopy experiment.
"""
struct ForwardModel{P_1, P_2}
    psf_1 :: GaussPSF{P_1}
    psf_2 :: GaussPSF{P_2}
    widths :: Tuple{Float64,Float64}
    min_intensity :: Float64
    max_intensity :: Float64
end

""" A single point source (localized fluorophore). """
struct PointSource
    intensity :: Float64
    x :: Float64
    y :: Float64
end

loc_to_point(l) = SVector{2,Float64}(l.x, l.y)

_pixel_bounds(w, np) = range(0.0, stop = w, length = np+1)


ForwardModel(s, np :: Int64) =
    ForwardModel(s, float(np), np)

ForwardModel(s, w :: Float64, np :: Int64) =
    ForwardModel(s, w, np, 0.0, Inf)

ForwardModel(s, w :: Float64, np :: Int64, mini, maxi) =
    ForwardModel(s, (w,w), (np,np), mini, maxi)

ForwardModel(sigma, (w1,w2), (n1, n2), mini, maxi) =
    ForwardModel(GaussPSF{n1+1}(sigma, SVector{n1+1}(_pixel_bounds(w1, n1))),
                        GaussPSF{n2+1}(sigma, SVector{n2+1}(_pixel_bounds(w2, n2))),
                        (w1, w2), mini, maxi)

""" Render a single point source."""
@inline (p :: ForwardModel)(s :: PointSource) =
    s.intensity*p.psf_1(s.x)*p.psf_2(s.y)'

""" Render multiple point sources."""
function (p :: ForwardModel{P_1, P_2})(sources :: Vector{PointSource}) where {P_1, P_2}
    r = @SMatrix zeros(P_1-1,P_2-1)
    for s in sources
        r += p(s)
    end
    r
end
