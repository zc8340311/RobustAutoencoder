import numpy as np
import math
import numpy.linalg as nplin
import tensorflow as tf
import DAE_tensorflow as DAE
def shrink(epsilon, x):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou

    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(x*0.)
    
    for i in xrange(len(x)):
        if x[i] > epsilon:
            output[i] = x[i] - epsilon
        elif x[i] < -epsilon:
            output[i] = x[i] + epsilon
        else:
            output[i] = 0
    return output

class RDAE():
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, layers_sizes, seeds, lambda_=1.0, error = 1.0e-5, 
                penOfOverfitting = 0.1):
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.penOfOverfitting = penOfOverfitting
        self.seeds = seeds #list of seed number for each layer
        self.errors=[]
        
        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)
        
    def fit(self, X, sess, target = None, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=50, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        
        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        
        penOfOverfitting = self.penOfOverfitting
        
        mu = (X.size) / (4.0 * nplin.norm(X,1))
        print self.lambda_ / mu
        LS0 = self.L + self.S
        
        XFnorm = nplin.norm(X,'fro')
        if verbose:
            print "X shape: ", X.shape
            print "L shape: ", self.L.shape
            print "S shape: ", self.S.shape
            print "mu: ", mu
            print "XFnorm: ", XFnorm
        
        for it in xrange(iteration):
            if verbose:
                print "Out iteration: " , it
            ## alternating project, first project to L
            self.L = X - self.S
            ## Using L to train the auto-encoder
            self.errors.extend(self.AE.fit(X = self.L, sess = sess, target = target, 
                                           iteration = inner_iteration,
                                           learning_rate = learning_rate, 
                                           batch_size = batch_size,
                                           verbose = verbose))
            ## get optmized L
            self.L = self.AE.getRecon(X = self.L, sess = sess)
            ## alternating project, now project to S
            self.S = shrink(self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)
            
            ## the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            ## There is no change for L and S
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm
            
            if verbose:
                print "c1: ", c1
                print "c2: ", c2
                
            if c1 < self.error and c2 < self.error :
                print "early break"
                break
            
            LS0 = self.L + self.S
        #self.S = shrink(self.lambda_/mu,self.S.reshape(X.size)).reshape(X.shape)
        return self.L , self.S
    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X = L, sess = sess)
    def getRecon(self, X, sess):
        return self.AE.getRecon(self.L, sess = sess)
if __name__ == "__main__":
	x = np.load(r"/home/zc/Documents/train_x_small.pkl")
	sess = tf.Session()
	rae = RDAE(sess = sess, seeds = 1.0, lambda_= 80, layers_sizes=[784,400])

	L, S = rae.fit(x ,sess = sess, learning_rate=0.01, batch_size = 40, inner_iteration = 50,
		    iteration=5, verbose=True)

	recon_rae = rae.getRecon(x, sess = sess)

	sess.close()
	print rae.errors
	from collections import Counter
	print Counter(S.reshape(S.size))[0]
	print 
	      

