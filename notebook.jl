### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 9bfbceea-fc5c-11ea-1f11-637728378b2f
begin
	using Pkg
	Pkg.activate(".")
	using SingleMoleculeLocalization
	using Plots
	using PlutoUI
end

# ╔═╡ 46d6eaba-fc5e-11ea-118e-adf416de2193
md"""
!!! warning

    This package uses the squared loss! This means it is only appropriate for 		  relatively high SNR images with *zero-mean* backgrounds,
    i.e. those generated by [TIRF](https://en.wikipedia.org/wiki/Total_internal_reflection_fluorescence_microscope) microscopy.

    You may need to apply a variance-stabilizing transform in addition to removing any background.
"""

# ╔═╡ 4113113e-fc5f-11ea-0286-ff3a622036c3
md"""
## Simulation
"""

# ╔═╡ 5055983e-fc5e-11ea-04fb-eb892dca899d
md"""
A [`ForwardModel`](#) describes the parameters of single molecule localization microscopy (SMLM) experiment.
For square images and an isotropic Gaussian point-spread function a model can be constructed as follows:
"""

# ╔═╡ 23d3ff0e-fc5d-11ea-0e58-f3a229b2f929
model = ForwardModel(1.5, 16)

# ╔═╡ 6b91c7ac-fc5f-11ea-3d07-1f44b24914b6
md"""This models a 16 by 16 image patch with an integrated Gaussian PSF with standard deviation of 1.5 pixels."""

# ╔═╡ 775d8e0e-fc5f-11ea-0f8d-4f48f4df07a8
md"""
!!! warning

    Because `ForwardModel` uses statically-sized arrays, performance degrades *rapidly* with increasing patch size.

A `PointSource` has three
properties: `x` and `y` are the coordinates of the source within an image, while
`intensity` is the brightness.
"""

# ╔═╡ 9091b40e-fc5f-11ea-20fa-f34d6755a862
p = PointSource(1.0, 7.5, 4.7)

# ╔═╡ 970990fe-fc5f-11ea-24ab-43fed51855fc
md"""constructs a unit-intensity point source at the spatial location ($(p.x), $(p.y)).

We can apply the forward model to a point source to generate a noiseless image:
"""

# ╔═╡ b366bc2c-fc5f-11ea-3380-ddc471d0a0ee
img = model(p);

# ╔═╡ b8181284-fc5f-11ea-2409-8136c0cffc50
heatmap(img, colormap=:grays, aspect_ratio = 1, c=:grays)

# ╔═╡ 43dc010e-fc60-11ea-2ee5-abb161902a8a
md"""It's also easy to generate a noisy image of a collection of sources:"""

# ╔═╡ 3d6a0c62-fc60-11ea-17bb-d9692bc9de4a
true_sources = [PointSource(3.23, 2.12, 12.12), PointSource(3.0, 2.5, 2.8), PointSource(4.0, 8.0, 7.0)];

# ╔═╡ 55e8d46e-fc63-11ea-243c-952e8898bef0
noiselevel = 0.01

# ╔═╡ 5447ce86-fc60-11ea-264d-8f17a03236de
noisy_img = model(true_sources) + noiselevel*randn(16,16);

# ╔═╡ 61c41b54-fc5e-11ea-0861-05d329ce1e10
heatmap(noisy_img,  aspect_ratio = 1, c=:grays);

# ╔═╡ 8263544a-fc60-11ea-3628-531a245e9580
md"""## Localizing small image patches

For small patches (less than about 20 by 20) we provide an efficient estimator,
`PatchLocalizer`."""

# ╔═╡ 8fcc6cf2-fc60-11ea-3246-7d5cd3c53ab4
patchlocalizer = PatchLocalizer(model);

# ╔═╡ 96d336b6-fc60-11ea-0340-770f05551bf3
est_sources = patchlocalizer(noisy_img, 5, 1E-1)

# ╔═╡ 609b3994-fc61-11ea-3074-1dd755edfd54
begin
	heatmap(noisy_img, c=:grays,  aspect_ratio = 1)
	scatter!(getproperty.(true_sources, :y).+0.5, getproperty.(true_sources, :x).+0.5, markercolor=:blue, label="true sources")
	scatter!(getproperty.(est_sources, :y).+0.5, getproperty.(est_sources, :x).+0.5, marker=:x, markercolor=:red, label="estimated sources")
end

# ╔═╡ af644d32-fc60-11ea-00b2-f17d495f79fe
md"""The two arguments to `patchlocalizer` are the maximum number of sources the algorithm will estimate and the minimum drop in the
loss function the algorithm will accept when adding a new source.
When the drop in the squared loss function from adding a new source falls below `1E-1`,
 the algorithm will return the previously estimated sources."""

# ╔═╡ c7ccc2e8-fc60-11ea-2162-d9e447ed422e
md"""## Localizing in large images

For larger images we provide the `ImageLocalizer` type:
"""

# ╔═╡ e9a706ec-fc60-11ea-3468-cddc5c9cbcd9
localizer = ImageLocalizer(1.5, 1E-1);

# ╔═╡ f02921dc-fc60-11ea-35e6-81c159c9b767
md"""The first argument is again the standard deviation of the gaussian PSF,
while the second is the minimum drop parameter discussed above.

`ImageLocalizer`'s can comfortably handle large images:"""

# ╔═╡ 1a5e2f5e-fc61-11ea-215a-2348bff277e8
# Simulate a large image...
large_img = begin
	large_img = 0.01*randn(1024, 1024)
	for i in 1:1500
		source = PointSource(5.0+randn(), 7+rand(), 7+rand())
		x,y = rand(1:(1024-15)),rand(1:(1024-15))
		large_img[x:x+15, y:y+15] .+= model(source)
	end
	large_img
end;

# ╔═╡ 416b2c50-fc61-11ea-0ea9-f3927bc904e6
large_image_estimated_sources = localizer(large_img)

# ╔═╡ 7562d8a4-fc62-11ea-06b6-7dba6ecd83e2
begin
	heatmap(large_img, aspect_ratio=1, c=:grays)
	scatter!(getproperty.(large_image_estimated_sources, :y).+0.5, getproperty.(large_image_estimated_sources, :x).+0.5, marker=:x, markersize=0.1, markercolor=:red, label="estimated sources")
end

# ╔═╡ Cell order:
# ╟─9bfbceea-fc5c-11ea-1f11-637728378b2f
# ╟─46d6eaba-fc5e-11ea-118e-adf416de2193
# ╟─4113113e-fc5f-11ea-0286-ff3a622036c3
# ╟─5055983e-fc5e-11ea-04fb-eb892dca899d
# ╠═23d3ff0e-fc5d-11ea-0e58-f3a229b2f929
# ╟─6b91c7ac-fc5f-11ea-3d07-1f44b24914b6
# ╟─775d8e0e-fc5f-11ea-0f8d-4f48f4df07a8
# ╠═9091b40e-fc5f-11ea-20fa-f34d6755a862
# ╟─970990fe-fc5f-11ea-24ab-43fed51855fc
# ╠═b366bc2c-fc5f-11ea-3380-ddc471d0a0ee
# ╟─b8181284-fc5f-11ea-2409-8136c0cffc50
# ╟─43dc010e-fc60-11ea-2ee5-abb161902a8a
# ╠═3d6a0c62-fc60-11ea-17bb-d9692bc9de4a
# ╠═55e8d46e-fc63-11ea-243c-952e8898bef0
# ╠═5447ce86-fc60-11ea-264d-8f17a03236de
# ╟─61c41b54-fc5e-11ea-0861-05d329ce1e10
# ╟─8263544a-fc60-11ea-3628-531a245e9580
# ╠═8fcc6cf2-fc60-11ea-3246-7d5cd3c53ab4
# ╠═96d336b6-fc60-11ea-0340-770f05551bf3
# ╟─609b3994-fc61-11ea-3074-1dd755edfd54
# ╟─af644d32-fc60-11ea-00b2-f17d495f79fe
# ╟─c7ccc2e8-fc60-11ea-2162-d9e447ed422e
# ╠═e9a706ec-fc60-11ea-3468-cddc5c9cbcd9
# ╟─f02921dc-fc60-11ea-35e6-81c159c9b767
# ╠═1a5e2f5e-fc61-11ea-215a-2348bff277e8
# ╠═416b2c50-fc61-11ea-0ea9-f3927bc904e6
# ╟─7562d8a4-fc62-11ea-06b6-7dba6ecd83e2
