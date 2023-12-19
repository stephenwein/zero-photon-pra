### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a975361e-b7a4-11ed-0bf9-99ee96a66231
# Author: Stephen C. Wein, Quantum Information Research Scientist at Quandela 
# Email: stephen.wein@quandela.com 
# Date: September 15th, 2023
#
# Description:
# This notebook demonstrates the ZPG method applied to a specialized noisy photon model and solves the resulting threshold detection statistics for a given unitary. The implementation is optimized for piecewise time-independent evolution, which can be solved using Krylov subspace methods much faster than the Adams integration method used by QuTiP, resulting in up to 2 orders of magnitude decrease in simulation time.

begin
	using LinearAlgebra
	using SparseArrays
	using BlockArrays
	using Random, RandomMatrices
	using Expokit
	Random.seed!(8675309)

	# Setting sparse matrix type
	Sp = SparseMatrixCSC{ComplexF64, Int64}

	# Takes the Kronecker product for op at position n in space with dims
	function KronInject(op::Sp, n::Int, dims::Vector{Int})::Sp
		ops = [sparse(ComplexF64(1.0)I, d, d) for d in dims]
		ops[n] = op
		opout = ops[1]
    	for i in 2:size(ops, 1)
            opout = kron(opout, ops[i])
		end
		return opout
	end
	
	# Builds the left-acting superoperator
	function SuperL(op::Sp)::Sp
		return kron(op, sparse(1.0I, size(op)))
	end

	# Builds the right-acting superoperator
	function SuperR(op::Sp)::Sp
		return kron(sparse(1.0I, size(op)), transpose(op))
	end

	# Builds the jump superoperator
	function SuperJ(op::Sp)::Sp
		return SuperL(op)*SuperR(copy(op'))
	end

	# Builds the dissipator superoperator
	function SuperD(op::Sp)::Sp
		return SuperJ(op) .- (0.5)*SuperL(op'*op) .- (0.5)*SuperR(op'*op)
	end

	# Builds the detector jump operator
	function JumpD(U::Matrix{ComplexF64}, n::Int)::Vector{Sp}
		m = size(U, 1)
		jumps = [spzeros(ComplexF64, Int64, 4^n, 4^n) for i in 1:m]
		s = sparse(Int64[1], Int64[2], ComplexF64[1.], 2, 2)
		d = fill(2, n)
		for k in 1:m for i in 1:n for j in 1:n
			jumps[k] .+= SuperL(KronInject(U[k,i]*s,i,d))*SuperR(KronInject(U[k,j]'*s',j,d))
		end end end
		return jumps
	end

	# Builds a time-independent source generator with dephasing
	function Generator(n::Int, dephasing::Float64)::Sp
		s = sparse(Int64[1], Int64[2], ComplexF64[1.], 2, 2)
		dims = fill(2, n)
		gen = spzeros(ComplexF64, Int64, 4^n, 4^n)
		for i in 1:size(dims, 1)
			gen .+= SuperD(KronInject(s, i, dims)) .+ dephasing*SuperD(KronInject(copy(s'*s), i, dims))
		end
		return gen
	end
	
	# Builds the trace vector in the FL space for n qubits
	function TraceFL(n::Int)::Array{Bool}
		return Array(reshape(sparse(I, 2^n, 2^n), 1, 4^n))
	end

	# Builds an initial state
	function InitialState(n::Int, theta::Float64)::Vector{Float64}
		istate = Float64[cos(theta/2)^2 0.; 0. sin(theta/2)^2]
		ostate = istate
		for i in 2:n
			ostate = kron(ostate, istate)
		end
		return reshape(ostate, 4^n)
	end

	# Threshold detection inverse transformation
	function ThresholdInverse(m::Int)::Matrix{Int8}
		if m == 1
			sub = Matrix{Int8}([1;;])
		else
			sub = ThresholdInverse(m - 1)
		end
		return Matrix(mortar((zeros(Int8, size(sub)), sub), (sub, -sub)))
	end
	
	# Solves for all generating points corresponding to configurations in vconfig
	function ThresholdStatistics(n::Int, m::Int; min=0, max=nothing, circuit=nothing, beta::Float64=1.,g2::Float64=0.,M::Float64=1.,eta::Float64=1., matrixinverse::Bool=true)::Dict{String, Float64}

		# Setting unset inputs
		if max == nothing
			max = m
		end
		
		# Setting up the source model
		gen = Generator(n, 1/M - 1)
		if g2 == 0.
			mu = beta
			thetai = 2*atan(sqrt(mu/(1 - mu)))
		else
			mu = abs((1 - sqrt(1 - 2*g2*(beta*0.9999)))/g2)
			sq = sqrt(abs((1 - 2*g2)*mu^2))
			thetai = 2*atan(sqrt((mu + sq)/(2 - mu - sq)))
			gen0 = Generator(n, 0.) 
			
			# Build pulse
			thetap = 2*atan(sqrt((mu - sq)/(2 - mu + sq)))
			pulseset = [cos(thetap/2)^2 .*sparse(I, 4^n, 4^n) .+ sin(thetap/2)^2 .*jump' for jump in JumpD(Matrix{ComplexF64}(I, n, n), n)]
			pulse = pulseset[1]
			for i in 2:size(pulseset, 1)
				pulse *= pulseset[i] 
			end
		end
		trFL = TraceFL(n)
		istate = InitialState(n, thetai)

		# Setting up the circuit
		if circuit == nothing
			circuit = rand(Haar(2), Int64(m))
		end
		
		# Determining virtual detector configurations
		outcomes = map(collect, reverse.(Iterators.product(fill(Int8(0):Int8(1),m)...))[:])
		gpointlist = Int64[]
		pnlist = Int64[]
		for i in 1:2^m
			if sum(outcomes[i]) <= max
				push!(gpointlist, 2^m - i + 1)
				if sum(outcomes[i]) >= min
					push!(pnlist, i)
				end
			end
		end
		outcomesg = outcomes[gpointlist]
		jumplist = [filter!(x->x≠0, collect(Int8(1):Int8(m)).*v) for v in outcomesg]
		
		# Building detector jump superoperators
		jumps = JumpD(circuit, n)
		
		# Determining necessary FL subspace
		randzpg = copy(gen)
		for i in 1:size(jumps, 1)
			randzpg .-= rand(Float64).*jumps[i]
		end
		mask = [abs(x) > 0 for x in expmv(1, randzpg, istate)]
	
		# Masking the superoperators and regularizing the sparsity patterns
		randzpg = randzpg[mask, mask]
		pattern = findnz(randzpg)
		
		jumps = [eta.*((jump[mask, mask] .+ randzpg).nzval .- randzpg.nzval) for jump in jumps]

		gen = (gen[mask, mask] .+ randzpg).nzval .- randzpg.nzval
		if g2 != 0.
			gen0 = (gen0[mask, mask] .+ randzpg).nzval .- randzpg.nzval
			pulse = pulse[mask, mask]
		end
	
		istate = istate[mask]
		trFL = trFL[mask]
		
		# Builds the zero-photon generator corresponding to a specific configuration
		genpoint = function (v::Vector{Int8})
			zpg = copy(gen)
			for i in v
				zpg .-= jumps[i]
			end
			return abs(dot(trFL, expmv(100, sparse(pattern[1], pattern[2], zpg), istate)))
		end 

		genpointg2 = function (v::Vector{Int8})
			zpg = copy(gen)
			for i in v
				zpg .-= jumps[i]
			end
			zpg1 = sparse(pattern[1], pattern[2], zpg)
			zpg0 = sparse(pattern[1], pattern[2], zpg - gen + gen0)
			return abs(dot(trFL, expmv(100, zpg0, pulse*expmv(100, zpg1, istate))))
		end 

		# Evaluating all generating points corresponding to each virtual config
		if g2 == 0.
			genset = map(genpoint, jumplist)
		else
			genset = map(genpointg2, jumplist)
		end

		# Taking the inverse transform for threshold detection
		if matrixinverse && m <= 17 # Evaluate and store the matrix inverse
			pnset = ThresholdInverse(m)[pnlist, gpointlist]*genset
		else # Use the explicit solution to trade memory for time
			if matrixinverse
				print("Warning: automatically set matrixinverse = false due to memory constraints")
			end
			pnset = [sum(prod(fill(-1, m).^(outcomesg[i] .+ o).* outcomesg[i].^o)*genset[i] for i in 1:size(genset, 1)) for o in  outcomes[2^m .- pnlist .+ 1]]
		end

		# Formatting and building the dictionary
		pnset[pnset .< 10^-14] .= 0
		labels = [join(map(string, o)) for o in outcomes[pnlist]]
		return Dict(labels[i]=>pnset[i] for i in 1:size(pnset, 1))
	end
end

# ╔═╡ 9ff46b2b-ce14-483f-8d30-01d596187fdb
# 8 photons in 8 modes (Haar random matrix)
# Noisy simulation in ~20s compared to equivalent Python implementation in ~20mins
begin
	@time ThresholdStatistics(8, 8, min=0, max=8, M=0.95, g2=0.01, eta=0.1)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
Expokit = "a1e7a1ef-7a5d-5822-a38c-be74e1bb89f4"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RandomMatrices = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
BlockArrays = "~0.16.36"
Expokit = "~0.2.0"
RandomMatrices = "~0.5.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "06afa8f704e2a49bdc89e7f6b403885da0e83b75"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "6189f7819e6345bcc097331c7db571f2f211364f"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "174b4970af15a500a29e76151f5c53195784b9d4"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "0.16.36"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5ce999a19f4ca23ea484e92a1774a61b8ca4cf8e"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.8.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cf25ccb972fec4e4817764d01c82386ae94f77b4"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.14"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "e76a3281de2719d7c81ed62c6ea7057380c87b1d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.98"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expokit]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "b0313f5f1825aabb1adb186c837e0d339d02f351"
uuid = "a1e7a1ef-7a5d-5822-a38c-be74e1bb89f4"
version = "0.2.0"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "58d83dd5a78a36205bdfddb82b1bb67682e64487"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.9"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "f372472e8672b1d993e93dada09e23139b509f9e"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.5.0"

[[deps.GSL]]
deps = ["GSL_jll", "Libdl", "Markdown"]
git-tree-sha1 = "3ebd07d519f5ec318d5bc1b4971e2472e14bd1f0"
uuid = "92c85e6c-cbff-5e0c-80f7-495c94daaecd"
version = "1.0.1"

[[deps.GSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "56f1e2c9e083e0bb7cf9a7055c280beb08a924c0"
uuid = "1b77fbbe-d8ee-58f0-85f9-836ddc23a7a4"
version = "2.7.2+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "eabe3125edba5c9c10b60a160b1779a000dc8b29"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.11"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomMatrices]]
deps = ["Combinatorics", "Distributions", "FastGaussQuadrature", "GSL", "LinearAlgebra", "Random", "SpecialFunctions", "Test"]
git-tree-sha1 = "1ee746a1cfb05b8c87747433b44dc9007b1762dd"
uuid = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
version = "0.5.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6da46b16e6bca4abe1b6c6fa40b94beb0c87f4ac"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.8"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "9cabadf6e7cd2349b6cf49f1915ad2028d65e881"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═a975361e-b7a4-11ed-0bf9-99ee96a66231
# ╠═9ff46b2b-ce14-483f-8d30-01d596187fdb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
