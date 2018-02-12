import numpy as np
import tensorflow as tf
from BasicAutoencoder import DeepAE as DAE
from shrink import l21shrink as SHR 

class RobustL21Autoencoder(object):
    """
    @author: Chong Zhou
    first version.
    complete: 10/20/2016

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
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-8):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]
        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def fit(self, X, sess, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=133, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        ##LS0 = self.L + self.S
        ## To estimate the size of input X
        if verbose:
            print "X shape: ", X.shape
            print "L shape: ", self.L.shape
            print "S shape: ", self.S.shape

        for it in xrange(iteration):
            if verbose:
                print "Out iteration: " , it
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
            self.S = SHR.l21shrink(self.lambda_, (X - self.L).T).T
        return self.L , self.S
    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X = L, sess = sess)
    def getRecon(self, X, sess):
        return self.AE.getRecon(self.L, sess = sess)
if __name__ == "__main__":
    x = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    with tf.Session() as sess:
        rae = RobustL21Autoencoder(sess = sess, lambda_= 20, layers_sizes=[x.shape[1],int(x.shape[1]*0.5)])

        L, S = rae.fit(x, sess = sess, inner_iteration = 60, iteration = 5,verbose = True)
        