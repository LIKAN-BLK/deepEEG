import numpy as np
import scipy as sp
from scipy.io import loadmat
from scipy import optimize
import theano
import theano.tensor as T
from scipy import signal
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression as LR
from theano.compile.debugmode import DebugMode
def man_filter(x,y,beta):
    x_t = x[:,:,np.squeeze(y).astype(bool)]
    x_nt = x[:,:,~(np.squeeze(y).astype(bool))]
    D = np.dot((np.mean(x_t,2) - np.mean(x_nt,2)).transpose(),(np.mean(x_t,2) - np.mean(x_nt,2)))
    DT = 0;
    for i in np.arange(x_t.shape[2]):
        DT=DT+np.dot(x_t[:,:,i].transpose(),x_t[:,:,i])
    DT = DT/x_t.shape[2]
    DT = DT - np.dot(x_t.sum(2).transpose(),x_t.sum(2))/(x_t.shape[2]^2)
    DNT = 0
    for i in np.arange(x_nt.shape[2]):
        DNT=DNT + np.dot(x_nt[:,:,i].transpose(),x_nt[:,:,i])
    DNT = DNT/x_nt.shape[2]
    DNT = DNT - np.dot(x_nt.sum(2).transpose(),x_nt.sum(2))/(x_nt.shape[2]^2)
    R = (beta*DT + (1 - beta)*DNT)
    w, v = sp.linalg.eig(D,R)
    ind = sorted(range(w.size), key=lambda k: w[k])
    v=v[:,ind]
    return v


def csp(x_train_filt, y_train):
    """Calculate Common Spatial Patterns Decompostion and Returns
    spatial filters W"""

    # Calculate correlation matrices
    X0 = x_train_filt[:,:,y_train[:,0]==0]
    X1 = x_train_filt[:,:,y_train[:,0]==1]

    C0 = 0.
    for i in xrange( X0.shape[2] ):
        C0 = C0 + np.dot(X0[:,:,i].transpose() , X0[:,:,i])

    C0 = C0/X0.shape[2]

    C1 = 0.
    for i in xrange( X1.shape[2] ):
        C1 = C1+np.dot(X1[:,:,i].transpose(), X1[:,:,i])

    C1 = C1/X1.shape[2]

    # Calculate CSP
    D, V   = sp.linalg.eig(C1, C1+C0);
    ind = sorted(range(D.size), key=lambda k: D[k])
    V = V[:,ind];
    W = np.hstack([V[:,0:2], V[:,25:]]);

    return W

def classify_csp(W, V, x_train_filt, y_train, x_test_filt, y_test):
    """ Classify data using CSP filter W"""
    # Project data
    proj_train = sp.tensordot(W.transpose(), x_train_filt, axes=[1,1])
    proj_test  = sp.tensordot(W.transpose(), x_test_filt, axes=[1,1])

    # Calculate features

    ftr = np.log( np.tensordot(proj_train**2, V, axes=[1,0]) )[:,:,0]
    fte = np.log( np.tensordot(proj_test **2, V, axes=[1,0]) )[:,:,0]
    # Classify
    logistic = LR()
    logistic.fit(ftr.transpose(), y_train[:,0])
    sc = logistic.score(fte.transpose(), y_test[:,0])

    return sc
def armijo_rule(cost,g):
    sigma = 0.1
    beta = 0.5
    a=0.01




# Load dataset
data = loadmat('sp1s_aa')
x = data['x_train']
y = np.array(data['y_train'], dtype=int)
y=y.transpose()
train_indexes,test_indexes = cross_validation.train_test_split(np.arange(y.size), test_size=0.2,random_state=0)
x_train=x[:,:,train_indexes]
x_test=x[:,:,test_indexes]
y_train=y[train_indexes]
y_test=y[test_indexes]

# Band-pass filter signal
samp_rate = 100.
(b, a) = signal.butter(5, np.array([8., 30.]) / (samp_rate / 2.), 'band')
x_train_filt = signal.filtfilt(b, a, x_train, axis=0)
x_test_filt  = signal.filtfilt(b, a, x_test, axis=0)



W = csp(x_train_filt, y_train) #(CH x 5)

# W = man_filter(x_train_filt,y_train,0.5)
# W = W[:,0:5]
V = np.ones((x.shape[0],1)) #(T x 1)
sc = classify_csp(W, V, x_train_filt, y_train, x_test_filt, y_test)

# Fine tune CSP pipeline
# Note input data dim: [batches, time, channel]
# Filter dim: [channel_in, channel_out]
from logistic_sgd import LogisticRegression

x_train_filt_T = theano.shared(x_train_filt.transpose(2, 0, 1))
x_test_filt_T  = theano.shared(x_test_filt.transpose(2, 0, 1))
y_train_T      = T.cast( theano.shared(y_train[:,0]), 'int32')
y_test_T       = T.cast( theano.shared(y_test[:,0]) , 'int32')

# lr         = 0.01 # learning rate
lr = T.scalar('lr')
batch_size = y_train.size/4
epochs     = 2500
index      = T.lscalar('index')
y          = T.ivector('y')
X          = T.tensor3('X')
csp_w      = theano.shared(W)
avg_v      = theano.shared(V)
u = theano.shared(value=np.zeros((5, 2), dtype=theano.config.floatX),   #TODO change 5 to number of eigenvector , 2 to number of classes
                                name='W', borrow=True)
b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),      #TODO change 2 to number of classes
                               name='b', borrow=True)
# proj_csp   = T.tensordot(X,csp_w,axes=[2,0])
# layer0_out = T.pow(proj_csp, 2)
#
# variance   = T.tensordot(layer0_out, avg_v, axes=[1,0])
#
# layer1_out = T.log((variance))[:,:,0]
# layer2     = LogisticRegression(input=layer1_out, n_in=5, n_out=2)
# cost       = layer2.negative_log_likelihood(y)+.01*T.sum(T.pow(avg_v,2)) - 1000*(T.sgn(T.min(avg_v)) - 1)*T.pow(T.min(avg_v),2)

def full_cost(W,V,U,B,X,y):
    spacial_filtered   = T.tensordot(X,W,axes=[2,0])
    layer0_out = T.pow(spacial_filtered, 2)
    variance   = T.tensordot(layer0_out, V, axes=[1,0])
    layer1_out = T.log((variance))[:,:,0]
    layer2     = LogisticRegression(input=layer1_out,U=U,B=B, n_in=5, n_out=2)
    cost       = layer2.negative_log_likelihood(y)+.01*T.sum(T.pow(V,2)) - 1000*(T.sgn(T.min(V)) - 1)*T.pow(T.min(V),2)
    return cost


params  = [csp_w, avg_v,u,b]
cost = full_cost(csp_w, avg_v,u,b,X,y)
def unrolled_cost_func(P,sizes,Xnum,ynum):
    W = P[:sizes['W'][0]*sizes['W'][1]]
    W = W.reshape(sizes['W'])

    V = P[sizes['W'][0]*sizes['W'][1] : (sizes['W'][0]*sizes['W'][1]+sizes['V'][0]*sizes['V'][1])]
    V = V.reshape(sizes['V'])

    U = P[(sizes['W'][0]*sizes['W'][1]+sizes['V'][0]*sizes['V'][1]) : (sizes['W'][0]*sizes['W'][1]+sizes['V'][0]*sizes['V'][1]+sizes['U'][0]*sizes['U'][1])]
    U = U.reshape(sizes['U'])

    B = P[ (sizes['W'][0]*sizes['W'][1]+sizes['V'][0]*sizes['V'][1]+sizes['U'][0]*sizes['U'][1]) :]
    B = B.reshape(sizes['B'])

    y          = T.ivector('y')
    X          = T.tensor3('X')
    cost = full_cost(W,V,U,B,Xnum,ynum)
    return cost.eval()


grads   = T.grad(cost,params)
grads_func = theano.function([X,y], grads)


def unrolled_grads_func(P,sizes,Xnum,ynum):
    grads_tmp = grads_func(Xnum,ynum)

    dW =grads_tmp[0].reshape(sizes['W'][0]*sizes['W'][1])
    dV =grads_tmp[1].reshape(sizes['V'][0]*sizes['V'][1])
    dU =grads_tmp[2].reshape(sizes['U'][0]*sizes['U'][1])
    dB =grads_tmp[3].reshape(sizes['B'][0])
    return np.hstack((dW,dV,dU,dB))

def armijo_rule(P,sizes,examples,labels,c1=1e-4,c2=0.9,beta=0.1,alpha=0.1):

    y          = T.ivector('y')
    X          = T.tensor3('X')

    pk=-unrolled_grads_func(P,sizes,examples,labels)
    while True:
        if (unrolled_cost_func(P+alpha*pk,sizes,examples,labels) < unrolled_cost_func(P,sizes,examples,labels)+c1*alpha*np.dot(pk,unrolled_grads_func(P,sizes,examples,labels))):
            break
        alpha = alpha*beta
    return alpha
updates = []
for param_i, grad_i in zip(params,grads):
    updates.append((param_i, param_i - lr*grad_i))


train_model = theano.function([index,lr], cost, updates=updates,
      givens={
          X: x_train_filt_T[index * batch_size: (index + 1) * batch_size],
          y: y_train_T[index * batch_size: (index + 1) * batch_size]})

def test_model_functional(W,V,U,B,X,y):
    spacial_filtered   = T.tensordot(X,W,axes=[2,0])
    layer0_out = T.pow(spacial_filtered, 2)
    variance   = T.tensordot(layer0_out, V, axes=[1,0])
    layer1_out = T.log((variance))[:,:,0]
    layer2     = LogisticRegression(input=layer1_out,U=U,B=B, n_in=5, n_out=2)
    return layer2.errors(y)


test_model = theano.function([], test_model_functional(csp_w, avg_v,u,b,X,y), givens = {
        X: x_test_filt_T, y: y_test_T})




for i in range(epochs):
    for j in range(y_train.size/batch_size):
        # W_vect = csp_w.ravel().eval()
        # V_vect = avg_v.ravel().eval()
        # U_vect = params[2].ravel().eval()
        # B_vect = params[3].ravel().eval()
        # sizes = {'W':csp_w.eval().shape,'V':avg_v.eval().shape,'U':params[2].eval().shape,'B':params[3].eval().shape}
        # P = np.hstack((W_vect,V_vect,U_vect,B_vect))
        # Xbatch = x_train_filt_T.eval()[j * batch_size: (j + 1) * batch_size]
        # ybatch = y_train_T.eval()[j * batch_size: (j + 1) * batch_size]
        # alpha=armijo_rule(P,sizes,Xbatch,ybatch)
        # print alpha
        cost_ij = train_model(j,0.01)




    er = test_model()
    print 'Epoch = %i' % i
    print 'Cost = %f' % cost_ij
    print 'Test error = % f' % er
    if np.isnan(cost_ij):
        break

