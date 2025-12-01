"""
Factory methods to construct a Classical Hopfield Network model.

We provide a `build` method that accepts a `NamedTuple` with a `memories`
field. `memories` is expected to be an (N x K) array where each column is a
memory vector with entries in {-1, 1} (typically `Int32`). The Hebbian rule
is used to compute the weights.
"""

# build(ModelType, (memories = linearimagecollection,)) usage in notebook passes a NamedTuple
function build(::Type{MyClassicalHopfieldNetworkModel}, kwargs::NamedTuple)
	@assert haskey(kwargs, :memories) "build requires a `memories` named argument"
	memories = kwargs.memories
	N, K = size(memories)

	# compute weights using Hebb's rule: average outer product over memories
	W = zeros(Float32, N, N)
	for k in 1:K
		s = Float32.(memories[:,k])
		W .+= s * s'
	end
	W ./= Float32(K)

	# remove self-connections
	for i in 1:N
		W[i,i] = 0.0f0
	end

	b = zeros(Float32, N) # classical Hopfield uses zero bias

	# precompute energy for each stored memory for reference
	energy = Dict{Int64,Float32}()
	for k in 1:K
		s = Float32.(memories[:,k])
		E = -0.5f0 * (s' * (W * s))[1]
		energy[k] = Float32(E)
	end

	return MyClassicalHopfieldNetworkModel(W, b, energy, Array{Int32,2}(memories))
end

