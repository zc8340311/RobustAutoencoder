"""
@author: Chong Zhou
first complete: 01/29/2016
Des:
    X = L + S
    L is a non-linearly low rank matrix and S is a sparse matrix.
    argmin |L - g(h(L))|_2 + |S|_1
    lagrangian multiplier to train model
"""
import timeit
import numpy as np
import math
import numpy.linalg as nplin
import DeepAE_target as dae
def minShrink1Plus2Norm(A, E, lam, mu):
    """
    @author: Prof. Randy
    from https://bitbucket.org/rcpaffenroth/playground/src/
    19a8a39cfc5b40e7d51697263881132a9d371fc5/src/lib/minShrink1Plus2Norm.py?at=feature-MVU&fileviewer=file-view-default

    Compute a fast minimization of shrinkage plus Frobenius norm.

    The is computes the minium of the following objective.

    .. math::

        \lambda \| \mathcal{S}_{\epsilon}( S_{ij} ) \|_1 +
        \mu / 2 \| S_{ij} - A_{ij} \|_F^2

    Args:
        A: A numpy array.

        E: A numpy array of error bounds.

        lam: The value of :math:`\lambda`.

        mu: The value of :math:`\mu`.

    Returns:
        The value of :math:`S` that achieves the minimum.
    """
    assert len(A.shape) == 1, 'A can only be a vector'
    assert A.shape == E.shape, 'A and E have  to have the same size'
    # Note, while the derivative is always zero when you use the
    # formula below, it is only a minimum if the second derivative is
    # positive.  The second derivative happens to be \mu.
    assert mu >= 0., 'mu must be >= 0'

    S = np.zeros(A.shape)

    for i in range(len(A)):
        if (lam/mu + A[i]) < -E[i]:
            S[i] = lam/mu + A[i]
        elif -E[i] < A[i] < E[i]:
            S[i] = A[i]
        elif E[i] < (-lam/mu + A[i]):
            S[i] = -lam/mu + A[i]
        else:
            Sp = (mu/2.)*(E[i] - A[i])*(E[i] - A[i])
            Sm = (mu/2.)*(-E[i] - A[i])*(-E[i] - A[i])
            if Sp < Sm:
                S[i] = E[i]
            else:
                S[i] = -E[i]
    return S
def shrink(epsilon, x):
    """
    @author: Prof. Randy
    from https://bitbucket.org/rcpaffenroth/playground/src/
    19a8a39cfc5b40e7d51697263881132a9d371fc5/src/lib/shrink.py?at=feature-MVU&fileviewer=file-view-default
    The shrinkage operator.
    This implementation is intentionally slow but transparent as
    to the mathematics.

    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(x*0.)
    if np.isscalar(epsilon):
        epsilon = np.ones(x.shape)*epsilon

    for i in range(len(x)):
        if x[i] > epsilon[i]:
            output[i] = x[i] - epsilon[i]
        elif x[i] < -epsilon[i]:
            output[i] = x[i] + epsilon[i]
        else:
            output[i] = 0
    return output

class DRAE_target():
    """
    @author: Chong Zhou
    first complete: 05/29/2016
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin |L - g(h(L))|_2 + |S|_1
        lagrangian multiplier to train model
    """
    def __init__(self, hidden_layers_sizes, seeds,lambda_=1.0, error = 1.0e-5, corruption_level=0.0, penOfOverfitting = 0.1):
        self.lambda_ = lambda_
        self.hidden_layers_sizes = hidden_layers_sizes
        self.error = error
        self.pennalty = penOfOverfitting
        self.seeds = seeds#list of seed number for each layer
        self.corruption_level = corruption_level
        self.errors=[]
    def fit(self, X, Target,learning_rate=0.1, iteration=20, verbose=False, inner_iteration= 20, batch_size = 25):

        numfea = X.shape[1]
        samples = X.shape[0]
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)

        self.AE = dae.DeepAE_target(visible_size = numfea,
                                    hidden_layers_sizes = self.hidden_layers_sizes,
                                    seed_num = self.seeds,
                                    corruption_level = self.corruption_level)
        if verbose:
            print "X shape", X.shape
            print "L shape", self.L.shape
            print "S shape", self.S.shape

        penOfOverfitting = 0.1

        mu = (X.shape[0] * X.shape[1]) / (4.0 * nplin.norm(X,1))


        LS0 = self.S + self.L
        E = (np.ones(X.shape) * self.lambda_/mu).reshape(X.size)
        XFnorm = nplin.norm(X,'fro')

        for it in xrange(iteration):
            if verbose:
                print "iteration: ", it
                print "mu", mu

            self.L = X - self.S

            if verbose:
                print "2 L shape" ,self.L.shape
                print "L type", type(self.L)
            self.AE.fit(X = self.L, Target = Target, iterations=inner_iteration, batch_size = batch_size, verbose = verbose,errors=self.errors, learning_rate=learning_rate)
            self.L = self.AE.getRecon(self.L)
            A = (X - self.L).reshape(X.size)

            self.S = shrink(E, A).reshape(X.shape)

            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if c1 < self.error and c2 < self.error :
                print "early break"
                break
            LS0 = self.L + self.S
        self.S = shrink(E, self.S.reshape(X.size)).reshape(X.shape)
        return self.L, self.S
    def transform(self, X):
        L = X - self.S
        return self.AE.transform(L)
    def getRecon(self):
        return self.AE.getRecon(self.L)
    def get_Weights(self,layer=0):
        return self.AE.encode_layers[layer].W.get_value(borrow=True).T

def test():
    X = np.load(r"/home/czhou2/Documents/notMNIST_small.npk")
    target = X.copy()
    def corrupt(X,corNum=10):
        N,p = X.shape[0],X.shape[1]
        for i in xrange(N):
            loclist = np.random.randint(0, p, size = corNum)
            for j in loclist:
                if X[i,j] > 0.5:
                    X[i,j] = 0
                else:
                    X[i,j] = 1
        return X
    X = corrupt(X,corNum=100)
    print X.shape
    inputsize=(28,28)

    hidden_layers_sizes = [400,256]
    seed_num=[11,17]
    hidden_size=(16,16)
    image_X = Image.fromarray(I.tile_raster_images(X=X,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    drae = DRAE_target(hidden_layers_sizes = hidden_layers_sizes, seeds = seed_num ,lambda_=10.0, error = 0.0001)
    L , S = drae.fit(X,Target=target,inner_iteration=40,verbose = True)
    #print rae.transform(X).shape
    image_S = Image.fromarray(I.tile_raster_images(X=S,img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_S.save(r"S.png")
    R = drae.getRecon()
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R.save(r"Recon.png")
    W = drae.get_Weights(0)
    image_W = Image.fromarray(I.tile_raster_images(X=W, img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W.save(r"W1.png")
    W1 = drae.get_Weights(1)
    image_W1 = Image.fromarray(I.tile_raster_images(X=W1, img_shape=(20,20), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W1.save(r"W2.png")
    H = drae.transform(X)
    image_H = Image.fromarray(I.tile_raster_images(X=H, img_shape=hidden_size, tile_shape=(20, 25),tile_spacing=(1, 1)))
    image_W.save(r"H.png")

def test2():
    X=np.load(r"/home/czhou2/Documents/NM_G.npk")
    #X= np.load(r"/home/czhou2/Documents/notMNIST_small.npk")
    print X.shape
    inputsize=(28,28)

    hidden_layers_sizes = [400,256]
    seed_num=[11,17]
    hidden_size=(16,16)
    image_X = Image.fromarray(I.tile_raster_images(X=X,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    drae = DRAE2(hidden_layers_sizes = hidden_layers_sizes, seeds = seed_num ,lambda_=20.0, error = 0.0001)
    L , S = drae.fit(X,inner_iteration=40,verbose = True)
    #print rae.transform(X).shape
    image_S = Image.fromarray(I.tile_raster_images(X=S,img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_S.save(r"S.png")
    R = drae.getRecon()
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R.save(r"Recon.png")
    W = drae.get_Weights(0)
    image_W = Image.fromarray(I.tile_raster_images(X=W, img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W.save(r"W1.png")
    W1 = drae.get_Weights(1)
    image_W1 = Image.fromarray(I.tile_raster_images(X=W1, img_shape=(20,20), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W1.save(r"W2.png")
    H = drae.transform(X)
    image_H = Image.fromarray(I.tile_raster_images(X=H, img_shape=hidden_size, tile_shape=(20, 25),tile_spacing=(1, 1)))
    image_W.save(r"H.png")
def test3():
    X = np.load(r"/home/czhou2/Documents/packets_1000.npk")
    #X= np.load(r"/home/czhou2/Documents/notMNIST_small.npk")
    X = X[:,:1512]/float(np.max(X))
    print X.shape
    inputsize=(36,42)

    hidden_layers_sizes = [30*30,25*25,20*20]
    seed_num=[11,17,20]
    hidden_size=(20,20)
    image_X = Image.fromarray(I.tile_raster_images(X = X, img_shape = inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    drae = DRAE2(hidden_layers_sizes = hidden_layers_sizes, seeds = seed_num ,lambda_=10.0, error = 0.0001)
    L , S = drae.fit(X,inner_iteration=40,iteration=15,verbose = True)
    #print rae.transform(X).shape
    image_S = Image.fromarray(I.tile_raster_images(X=S,img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_S.save(r"S.png")
    R = drae.getRecon()
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R.save(r"Recon.png")
    W = drae.get_Weights(0)
    image_W = Image.fromarray(I.tile_raster_images(X=W, img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W.save(r"W1.png")
    W1 = drae.get_Weights(1)
    image_W1 = Image.fromarray(I.tile_raster_images(X=W1, img_shape=(30,30), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W1.save(r"W2.png")
    W2 = drae.get_Weights(2)
    image_W2 = Image.fromarray(I.tile_raster_images(X=W2, img_shape=(25,25), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W2.save(r"W3.png")
    H = drae.transform(X)
    image_H = Image.fromarray(I.tile_raster_images(X=H, img_shape=hidden_size, tile_shape=(20, 25),tile_spacing=(1, 1)))
    image_W.save(r"H.png")
if __name__ == "__main__":
    import PIL.Image as Image
    import ImShow as I
    test()
    #test3()
