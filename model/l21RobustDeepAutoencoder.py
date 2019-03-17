import numpy as np
import tensorflow as tf
from BasicAutoencoder import DeepAE as DAE
from shrink import l21shrink as SHR 

class RobustL21Autoencoder():
    """
    @author: Chong Zhou
    first version.
    complete: 10/20/2016
    Update to Python3: 03/15/2019
    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_2,1
        Use Alternating projection to train model
        The idea of shrink the l21 norm comes from the wiki 'Regularization' link: {
            https://en.wikipedia.org/wiki/Regularization_(mathematics)
        }
    Improve:
        1. fix the 0-cost bugs

    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-5):
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]

        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def fit(self, X, sess, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=40, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        
        ## To estimate the size of input X
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            
        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            self.L = X - self.S
            ## Using L to train the auto-encoder
            self.AE.fit(self.L, sess = sess,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose)
            ## get optmized L
            self.L = self.AE.getRecon(X = self.L, sess = sess)
            ## alternating project, now project to S and shrink S
            self.S = SHR.l21shrink(self.lambda_, (X - self.L))
            
        return self.L , self.S
    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X = L, sess = sess)
    def getRecon(self, X, sess):
        return self.AE.getRecon(X, sess = sess)
if __name__ == "__main__":
    x = np.load(r"../data/data.npk")[:500]
    
    with tf.Session() as sess:
        rae = RobustL21Autoencoder(sess = sess, lambda_= 4000, layers_sizes=[784,400,255,100])

        L, S = rae.fit(x, sess = sess, inner_iteration = 60, iteration = 5,verbose = True)

