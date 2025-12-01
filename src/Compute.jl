"""
Compute / utility functions for the Classical Hopfield Network.

Provides:
- `recover(model, s0, true_image_energy; ...)` : perform asynchronous updates to
  attempt to recover a stored memory starting from `s0`.
- `compute_energy(model, s)` : compute network energy for a binary state vector s.
- `hamming(a,b)` : Hamming distance between two binary vectors.
- `decode(s)` : reshape a {-1,1} vector to a 2D image with values in {0.0,1.0}.
"""

# compute energy for a state vector s (entries ±1). E = -1/2 * s' W s - b' s
function compute_energy(model::MyClassicalHopfieldNetworkModel, s::AbstractVector)
	s_f = Float32.(s)
	E = -0.5f0 * (s_f' * (model.W * s_f))[1] - (model.b' * s_f)[1]
	return Float32(E)
end

# Hamming distance between two vectors (counts unequal entries)
function hamming(a::AbstractVector, b::AbstractVector)
	@assert length(a) == length(b)
	count = 0
	for i in 1:length(a)
		if a[i] != b[i]
			count += 1
		end
	end
	return count
end

# decode a {-1,1} vector into a square image matrix of Float32 values in {0.0,1.0}
function decode(s::AbstractVector)
	N = length(s)
	n = round(Int, sqrt(N))
	@assert n * n == N "decode: input length must be a perfect square"
	# map -1 -> 0.0, 1 -> 1.0 (Bool -> Float32 broadcasting)
	vals = Float32.(s .== 1)
	return reshape(vals, n, n)
end

# recover: asynchronous updates, store frames and energies at each step
function recover(model::MyClassicalHopfieldNetworkModel, s0::Array{Int32,1}, true_image_energy::Float32; maxiterations::Int64=1000, patience::Union{Int,Nothing}=5, miniterations_before_convergence::Union{Int,Nothing}=nothing)
	N = length(s0)
	s = copy(s0)

	frames = Dict{Int64, Array{Int32,1}}()
	energydictionary = Dict{Int64, Float32}()

	if miniterations_before_convergence === nothing
		miniterations_before_convergence = patience
	end

	# history queue for patience check
	history = Vector{Array{Int32,1}}()

	for t in 1:maxiterations
		# asynchronous: pick a random neuron and update it
		i = rand(1:N)
		# local field
		h = 0.0f0
		for j in 1:N
			h += model.W[i,j] * Float32(s[j])
		end
		h -= model.b[i]

		new_si = h >= 0.0f0 ? Int32(1) : Int32(-1)
		s[i] = new_si

		# record frame and energy
		E = compute_energy(model, s)
		frames[t] = copy(s)
		energydictionary[t] = E

		push!(history, copy(s))
		if length(history) > patience
			popfirst!(history)
		end

		converged = false
		if t >= miniterations_before_convergence
			if length(history) == patience
				same = true
				for k in 2:patience
					if history[k] != history[1]
						same = false; break
					end
				end
				if same
					converged = true
				end
			end

			# check if current state exactly matches any stored memory
			for k in 1:size(model.memories, 2)
				mem = model.memories[:,k]
				if all(mem .== s)
					converged = true
					break
				end
			end
		end

		if converged
			break
		end
	end

	return frames, energydictionary
end

