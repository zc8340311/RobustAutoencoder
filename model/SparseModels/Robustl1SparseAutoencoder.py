import numpy as np
import tensorflow as tf
import l1SparseAutoencoder as sdae
import sys
sys.path.append("../")
from shrink import l21shrink as SHR

class Robust_l1_SparseAutoencder():
    """
    @author: Chong Zhou

    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))||+ sparsity * ||Encoder(L)||_2,1 + lambda * ||S||_2,1
        Use Alternating projection to train model
    """
    def __init__(self, sess, layers_sizes, sparsity=0.4, lambda_=1.0, error = 1.0e-5):
        """
        para: 
            sparsity: is a number that indicate the weight of the sparse penalty in the sparse autoencoder 
            lambda_: is weight of the sparsity of S
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.sparcity = sparsity
        self.SAE = sdae.l1_Sparse_Autoencoder(sess = sess, input_dim_list = self.layers_sizes,
                                                    sparsity = self.sparcity)

    def fit(self, X, sess, learning_rate=0.05, inner_iteration = 50,
            iteration=20, batch_size=40, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)

        if verbose:
            print "X shape: ", X.shape
            print "L shape: ", self.L.shape
            print "S shape: ", self.S.shape

        for it in xrange(iteration):
            if verbose:
                print "Out iteration: " , it
            ## alternating project, first project to L
            self.L = np.array(X - self.S,dtype=float)
            ## Using L to train the auto-encoder
            self.SAE.fit(self.L, sess = sess,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose)
            ## get optmized L
            self.L = self.SAE.getRecon(X = self.L, sess = sess)
            ## alternating project, now project to S and shrink S
            self.S = SHR.l21shrink(self.lambda_, (X - self.L).T).T

        return self.L, self.S

    def transform(self, X, sess):
        return self.SAE.transform(X = X, sess = sess)
    def getRecon(self, X, sess):
        return self.SAE.getRecon(self.L, sess = sess)


if __name__ == '__main__':
    x = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    with tf.Session() as sess:
        rsae = Robust_l1_SparseAutoencder(sess = sess, lambda_= 4, layers_sizes=[784,784,784,784],sparsity=0.5)

        L, S = rsae.fit(x, sess = sess, inner_iteration = 20, iteration = 30,verbose = True)
        print L.shape,S.shape
