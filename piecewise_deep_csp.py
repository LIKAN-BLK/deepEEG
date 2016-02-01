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

def armijo_rule(W,V,U,B,examples,labels,num_grads):
    sigma = 0.1
    beta = 0.5
    alpha = 0.1
    grad_norm=0
    y          = T.ivector('y')
    X          = T.tensor3('X')
    for i in np.arange(3):
        grad_norm +=(np.linalg.norm(num_grads[i])**2)
    while True:
        costda = full_cost(W-alpha*num_grads[0],V-alpha*num_grads[1],U-alpha*num_grads[2],B-alpha*num_grads[3],X,y)
        cost = full_cost(W,V,U,B,X,y)


        if (costda.eval({X:examples,y:labels}) < cost.eval({X:examples,y:labels})-sigma*alpha*grad_norm):
            break
        alpha = alpha*beta
    return alpha



params  = [csp_w, avg_v,u,b]
cost = full_cost(csp_w, avg_v,u,b,X,y)
grads   = T.grad(cost,params)

updates = []
for param_i, grad_i in zip(params,grads):
    updates.append((param_i, param_i - lr*grad_i))


train_model = theano.function([index,lr], cost, updates=updates,
      givens={
          X: x_train_filt_T[index * batch_size: (index + 1) * batch_size],
          y: y_train_T[index * batch_size: (index + 1) * batch_size]})



# test_model = theano.function([], layer2.errors(y), givens = {
#         X: x_test_filt_T, y: y_test_T})

num_gradient = theano.function([index], [grads[0],grads[1],grads[2],grads[3]],
                               givens = {X: x_train_filt_T[index * batch_size: (index + 1) * batch_size],
                                         y: y_train_T[index * batch_size: (index + 1) * batch_size]})


for i in range(epochs):
    for j in range(y_train.size/batch_size):
        # lr = sp.optimize.line_search(train_model(j),)
        Xbatch = x_train_filt_T.eval()[j * batch_size: (j + 1) * batch_size]
        ybatch = y_train_T.eval()[j * batch_size: (j + 1) * batch_size]
        alpha = armijo_rule(csp_w,avg_v,u,b,Xbatch,ybatch,num_gradient(j))
        cost_ij = train_model(j,alpha)

        #num_gradij = num_gradient(j)


    # er = test_model()
    print 'Epoch = %i' % i
    print 'Cost = %f' % cost_ij
    # print 'Test error = % f' % er
    if np.isnan(cost_ij):
        break

