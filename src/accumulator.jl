module ArrayAccumulator

	using DocStringExtensions

  export Accumulator
	
	"""
	$TYPEDFIELDS
	
	Struct to accumulate values over variable-length stencils
	"""
	struct Accumulator
	    n_output::Int64
	    stencils::Dict{Int64, Tuple}
	    first_index::Bool
	end
	
	"""
	$TYPEDSIGNATURES
	
	Construct accumulator struct from stencils and weights.
	
	Example:
	
	```
	acc = Accumulator(
	    [[1, 2], [2, 3, 4]],
	    [[-1.0, 2.0], [3.0, 4.0, 5.0]]
	)
	
	v = [1, 2, 3, 4]
	@show acc(v)
	# [3.0, 38.0]
	```
	
	If `first_index` is true, the first array dimension is considered
	to be the summation axis.
	"""
	function Accumulator(
	    inds::AbstractVector,
	    weights::Union{AbstractVector, Nothing} = nothing;
	    first_index::Bool = false
	)
	    ls = length.(inds)
	
	    d = Dict{Int64, Tuple}()
	    for l in unique(ls)
	        isval = (ls .== l) |> findall
	
	        is = reduce(
	            hcat, inds[isval]
	        )
	        ws = nothing
	        if !isnothing(weights)
	            ws = reduce(
	                hcat, weights[isval]
	            )
	        end
	
	        d[l] = (isval, is, ws)
	    end
	
	    n = length(ls)
	    Accumulator(n, d, first_index)
	end
	
	"""
	$TYPEDSIGNATURES
	
	Run accumulator over vector.

	If `Δ` is true, then the sum occurs over differences between the
	fetched stencil values, and the current stencil point.
	If `f` is provided, it is applied on the values to sum before adding them.
	"""
	function (acc::Accumulator)(v::AbstractVector;
			Δ::Bool = false, f = identity)
	    vnew = similar(v, eltype(v), acc.n_output)
	
	    vnew .= 0
	    for (i, stencil, weights) in values(acc.stencils)
	        if isnothing(weights)
	            vnew[i] .= dropdims(
	                sum(
	                    v[stencil];
	                    dims = 1
	                );
	                dims = 1
	            )
	        else
	            vnew[i] .= dropdims(
	                sum(
	                    (
							Δ ?
							f(v[stencil] .- v[i]') :
							f(v[stencil])
						) .* weights;
	                    dims = 1
	                );
	                dims = 1
	            )
	        end
	    end
	
	    vnew
	end
	
	"""
	$TYPEDSIGNATURES
	
	Run accumulator over array.
	Summation occurs over last dimension if `first_index` is false,
	or the first if true.

	If `Δ` is true, then the sum occurs over differences between the
	fetched stencil values, and the current stencil point.
	If `f` is provided, it is applied on the values to sum before adding them.
	"""
	(acc::Accumulator)(v::AbstractArray;
			Δ::Bool = false, f = identity) = mapslices(
	    vv -> acc(vv; Δ = Δ, f = f), v; dims = (acc.first_index ? 1 : ndims(v))
	)
	
end