import numpy as np
import tensorflow as tf
import KLSparseAutoencoder as sdae
import sys
sys.path.append("../")
from shrink import l21shrink as SHR

class Robust_KL_SparseAutoencder():
    """
    @author: Chong Zhou
    modified: 02/12/2018
    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))||+ KL(Encoder(L)) + ||S||_2,1
        Use Alternating projection to train model
 
    """
    def __init__(self, sess, layers_sizes, sparsity, sparse_ratio, lambda_=1.0, error = 1.0e-5):
        """
        sparsity is the weight of penalty term 
        sparse ratio is special for the KL divergence, how much sparsity do you expect on each hidden feature. 
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.sparcity = sparsity
        self.sparse_ratio = sparse_ratio
        self.SAE = sdae.KL_Sparse_Autoencoder( sess = sess, input_dim_list = self.layers_sizes,
                                                    sparsity = self.sparcity, sparse_ratio = self.sparse_ratio)

    def fit(self, X, sess, learning_rate=0.05, inner_iteration = 50,
            iteration=20, batch_size=40, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            
        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
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
            ## alternating project, now project to S and shrink S.T
            self.S = SHR.l21shrink(self.lambda_, (X - self.L).T).T
        return self.L, self.S

    def transform(self, X, sess):
        return self.SAE.transform(X = X, sess = sess)
    
    def getRecon(self, X, sess):
        return self.SAE.getRecon(self.L, sess = sess)

if __name__ == '__main__':
    x = np.load(r"../../data/data.npk")[:500]
    with tf.Session() as sess:
        rsae = Robust_KL_SparseAutoencder(sess = sess, lambda_= 40, layers_sizes=[784,784,784,784],
                                            sparsity= 0.5, sparse_ratio= 0.2)

        L, S = rsae.fit(x, sess = sess, inner_iteration = 20, iteration = 30,verbose = True)
        print (L.shape,S.shape)
