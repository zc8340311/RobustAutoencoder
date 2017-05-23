import numpy as np
import tensorflow as tf
import Sparsel21Autoencoder as sdae


class RobustSparseAutoencder():
    """
    @author: Chong Zhou

    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))||+ ||Encoder(L)||_2,1 + ||S||_2,1
        Use Alternating projection to train model
        The idea of shrink the l21 norm comes from the wiki 'Regularization' link: {
            https://en.wikipedia.org/wiki/Regularization_(mathematics)
        }


    """
    def __init__(self, sess, layers_sizes, sparsities, lambda_=1.0, error = 1.0e-5):
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.sparcities = sparsities
        self.errors=[]
        assert len(sparsities) == len(layers_sizes) - 1
        self.SAE = sdae.Sparsel21_Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes,
                                                    sparsities = self.sparcities)

    def l21shrink(self, epsilon, x):
        """
        auther : Chong Zhou
        Args:
            epsilon: the shrinkage parameter
            x: matrix to shrink on
        Ref:
            wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
        Returns:
            The shrunk matrix
        """
        output = x.copy()
        norm = np.linalg.norm(x, ord=2, axis=0)

        for i in xrange(x.shape[1]):
            if norm[i] > epsilon:
                for j in xrange(x.shape[0]):
                    output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
            elif norm[i] < -epsilon:
                for j in xrange(x.shape[0]):
                    output[j,i] = x[j,i] + epsilon * x[j,i] / norm[i]
            else:
                output[:,i] = 0.
        return output
    def fit(self, X, sess, learning_rate=0.05, inner_iteration = 50,
            iteration=20, batch_size=40, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        #LS0 = self.L + self.S
        ## To estimate the size of input X

        if verbose:
            print "X shape: ", X.shape
            print "L shape: ", self.L.shape
            print "S shape: ", self.S.shape
            print "X sum value", np.linalg.norm(X,'fro')

        for it in xrange(iteration):
            if verbose:
                print "Out iteration: " , it
            ## alternating project, first project to L
            self.L = np.array(X - self.S,dtype=float)
            print self.L.dtype
            ## Using L to train the auto-encoder
            self.SAE.fit(self.L, sess = sess,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose)
            ## get optmized L
            self.L = self.SAE.getRecon(X = self.L, sess = sess)
            ## alternating project, now project to S and shrink S
            self.S = self.l21shrink(self.lambda_, (X - self.L))
            print "S after shrink",self.S.dtype

        return self.L , self.S

    def transform(self, X, sess):
        return self.SAE.transform(X = X, sess = sess)
    def getRecon(self, X, sess):
        return self.SAE.getRecon(self.L, sess = sess)







if __name__ == '__main__':
    x = np.load(r"/home/zc/Documents/train_x_small.pkl")
    x = np.array(x,dtype=float)
    with tf.Session() as sess:
        rsae = RobustSparseAutoencder(sess = sess, lambda_= 4000, layers_sizes=[784,400,255,100],sparsities=[0.5,0.5,0.5])

        L, S = rsae.fit(x, sess = sess, inner_iteration = 20, iteration = 30,verbose = True)
        print L.shape,S.shape
