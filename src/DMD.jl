using LinearAlgebra


function DMD(X::Array{Float64,2},Y::Array{Float64,2})
	"""
		An algorithm for approximating modes of the Koopman operator


		inputs:

			X: input time series
			Y: output time series(X shifted forward by 1 time step)
			r: rank of the truncation(assuming low-dimensional embedding)

		outputs:
			Phi: modes of the Koopman operator
	"""

	# 1. SVD of input matrix: 
	U_,S_,V_ = svd(X)

	## compute r, the rank of the truncation, using the 90% variance heuristic: 
	N = length(S_)

	var = cumsum(S_.^2)
	var_ = var./var[N]
	r = findmin((var_-0.9).^2)

	# rank-r truncation
	U = U_[1:end,1:r]
	S = diagm(S_)[1:r,1:r]
	V = V_[1:end,1:r]

	## compute S inverse: 
	S_inv = inv(S)

	# 2. build A tilde: 
	Ã = U'*Y*V*S_inv

	## 3. Compute eigenvalues and eigenvectors: 
	mu,W = eigen(Ã)

	# 4. build DMD modes: 
	Phi = Y*V*S_inv*W

	return Phi

end