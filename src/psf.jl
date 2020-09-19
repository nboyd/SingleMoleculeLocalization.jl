""" Statically-sized 1D Gaussian PSF approximation. """
struct GaussPSF{P}
    sigma :: Float64
    pixel_bounds :: SVector{P, Float64}
end

""" Evaluate the PSF ψ(x). """
@inline function (s :: GaussPSF)(x :: Float64)
    sigma = s.sigma
    pixel_bounds = s.pixel_bounds

    dist_from_x = (s.pixel_bounds .- x)./sigma
    ints = int_psf.(dist_from_x)

    popfirst(ints)-pop(ints)
end

@inline function derivatives(s :: GaussPSF, x)
    sigma = s.sigma
    pixel_bounds = s.pixel_bounds

    dist_from_x = (s.pixel_bounds .- x)./sigma
    ints = int_psf.(dist_from_x)

    r = popfirst(ints)-pop(ints)

    #### first derivative
    d_ints = psf.(dist_from_x)
    d_x = -(popfirst(d_ints)-pop(d_ints))/sigma

    #### Second derivative
    dd_ints = d_psf.(dist_from_x)
    dd_x = (popfirst(dd_ints)-pop(dd_ints))/(sigma*sigma)

    r, d_x, dd_x
end

### Fast approximations to the gaussian psf and derivatives...
# These come from cubic hermite spline approximations to the PDF.
# Could optimize these, but they seem fine.

""" (Approximate) CDF of a standard gaussian. """
@inline function int_psf(x)
    s = !signbit(x)
    x = ifelse(s, x, -x)
    r = ifelse(x ≤ 0.5, @evalpoly(x, 0.0,0.3989422804014327,0.0,-0.07015270562709953,0.011475151166382952),
    ifelse(x ≤ 1.0, @evalpoly(x, 0.0003252383067254283,0.39562836682196156,0.012077762115416488,-0.08910441695032567,0.022375021079331425),
    ifelse(x ≤ 1.5, @evalpoly(x, -0.010421430955656785,0.43667799829341414,-0.04659111672464811,-0.05192887663502551,0.013565397395025491),
    ifelse(x ≤ 3.5, @evalpoly(x, -0.10942747638184469,0.6941333642718885,-0.2974857275450951,0.056663948103827654,-0.004047424864559117),
    0.497939217356058))))
    ifelse(s, r, -r) + 0.5
end

""" (Approximate) derivative of the PDF of a standard gaussian. """
@inline function d_psf(x)
    s = !signbit(x)
    x = ifelse(s, x, -x)
    r = ifelse(x ≤ 0.5, @evalpoly(x, 0.0,-0.42091623376259724,0.13770181399659542),
    ifelse(x ≤ 1.0, @evalpoly(x, 0.024155524230832975,-0.534626501701954,0.2685002529519771),
    ifelse(x ≤ 1.5, @evalpoly(x, -0.09318223344929621,-0.31157325981015305,0.1627847687403059),
    ifelse(x ≤ 3.5, @evalpoly(x, -0.5949714550901902,0.3399836886229659,-0.04856909837470941),
    0.0))))
    ifelse(s, r, -r) + 0.5
end

""" (Approximate) PDF of a standard gaussian. """
@inline function psf(x)
    x = abs(x)
    ifelse(x ≤ 0.5, @evalpoly(x, 0.3989422804014327,0.0,-0.21045811688129862,0.04590060466553181),
    ifelse(x ≤ 1.0, @evalpoly(x, 0.39562836682196156,0.024155524230832975,-0.267313250850977,0.0895000843173257),
    ifelse(x ≤ 1.5, @evalpoly(x, 0.43667799829341414,-0.09318223344929621,-0.15578662990507652,0.054261589580101965),
    ifelse(x ≤ 3.5, @evalpoly(x, 0.6941333642718885,-0.5949714550901902,0.16999184431148295,-0.016189699458236468),
    0.0))))
end
