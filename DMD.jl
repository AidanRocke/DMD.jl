using LinearAlgebra


## data matrix: 
X_ = rand(10,100)

X,Y = X_[1:end,1:99], X_[1:end,2:100]

# 1. SVD of input matrix: 
U_,S_,V_ = svd(X)

# rank-3 truncation
r = 3
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

# compute time evolution
b = dot(pinv(Phi), X[:,0])
Psi = np.zeros([r, len(t)], dtype='complex')
for i,_t in enumerate(t):
    Psi[:,i] = multiply(power(mu, _t/dt), b)


# compute DMD reconstruction
D2 = dot(Phi, Psi)
np.allclose(D, D2) # True
