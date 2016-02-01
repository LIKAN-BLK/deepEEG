import scipy as sp
import numpy as np
import scipy.optimize
import theano
import theano.tensor as T
M=T.matrix('M')
N=T.matrix('N')
X=T.matrix('X')
Z = T.dot(T.dot(X,M),N).norm(L=2)
z_func=theano.function([M,N,X],Z)
def z_func_num(Param,X):
    return z_func(Param[0],Param[1],X)
z_grad = T.grad(Z,[M,N])
z_grad_func = theano.function([M,N,X],z_grad)
def z_grad_num(P,X):
    return z_grad_func(P[0],P[1],X)

x=np.random.rand(3,4)
m=theano.shared(np.random.rand(4,3))
n=theano.shared(np.random.rand(3,4))
sp.optimize.line_search(z_func_num,z_grad_num,)