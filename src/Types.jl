# Types for Classical Hopfield Network

mutable struct MyClassicalHopfieldNetworkModel
	W::Array{Float32,2}        # weight matrix (N x N)
	b::Array{Float32,1}        # bias/threshold vector (N)
	energy::Dict{Int64,Float32}# energy per stored memory
	memories::Array{Int32,2}   # stored memories as columns (N x K)
end

# allow a simple default constructor if needed (zero-sized placeholder)
MyClassicalHopfieldNetworkModel() = MyClassicalHopfieldNetworkModel(Array{Float32,2}(undef,0,0), Float32[], Dict{Int64,Float32}(), Array{Int32,2}(undef,0,0))
