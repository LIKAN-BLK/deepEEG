import scipy as sp
import numpy as np
import scipy.optimize
import theano
import theano.tensor as T
M=T.matrix('M')
N=T.matrix('N')
X=T.matrix('X')
np.random.seed(1)
m=theano.shared(np.random.rand(4,3))
np.random.seed(1)
n=theano.shared(np.random.rand(3,4))


z_func=theano.function([M,N,X],Z)
def z_func_num(P,sizeM,sizeN,X):
    Z = (T.dot(T.dot(X,M),N)**2).norm(L=2)
    M=P[0,:sizeM[0]*sizeM[1]]
    M=np.reshape(M,(sizeM[0],sizeM[1]))
    N=P[0,sizeN[0]*sizeN[1]:]
    N=np.reshape(N,(sizeN[0],sizeN[1]))
    return z_func(M,N,X)
z_grad = T.grad(Z,[M,N])
z_grad_func = theano.function([M,N,X],z_grad)

def z_grad_num(P,sizeM,sizeN,X):
    M=P[0,:sizeM[0]*sizeM[1]]
    M=np.reshape(M,(sizeM[0],sizeM[1]))
    N=P[0,sizeN[0]*sizeN[1]:]
    N=np.reshape(N,(sizeN[0],sizeN[1]))
    return np.hstack([np.reshape(z_grad_func(X)[0],(1,sizeM[0]*sizeM[1])),np.reshape(z_grad_func(X)[1],(1,sizeN[0]*sizeN[1]))])


# m=theano.shared(np.array([[2,3,4],[23,32,14],[24,31,42],[23,31,44]]))
# n=theano.shared(np.array([[2,3,4,56],[23,32,14,39],[24,31,42,54]]))

np.random.seed(2)
x=np.random.rand(3,4)
m_num=np.reshape(m.eval(),(1,12))
n_num=np.reshape(n.eval(),(1,12))
# z_func_num(np.hstack([m_num,n_num]),m.eval().shape,n.eval().shape)
# z_grad_num(np.hstack([m_num,n_num]),m.eval().shape,n.eval().shape)
a = sp.optimize.line_search(z_func_num,z_grad_num,np.hstack([m_num,n_num]),-(z_grad_num(np.hstack([m_num,n_num]),m.eval().shape,n.eval().shape,x).transpose()),args = (m.eval().shape,n.eval().shape,x))
print a