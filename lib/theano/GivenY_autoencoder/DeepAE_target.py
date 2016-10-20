"""
@author: Chong Zhou
first complete:
reference:
    deep learning tutorial sda.py
"""
import os
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):
    def __init__(self, rng, in_dim, out_dim, W=None, b=None, activation=T.nnet.sigmoid):
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (in_dim + out_dim)),
                    high=np.sqrt(6. / (in_dim + out_dim)),
                    size=(in_dim, out_dim)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value= W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((out_dim,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        self.activation=activation
        self.params=[self.W, self.b]
    def output(self,x):
        linear_out = T.dot(x, self.W) + self.b
        if self.activation is None:
            return linear_out
        else:
            return self.activation(linear_out)

class DeepAE_target(object):
    def __init__(self,
            visible_size,
            hidden_layers_sizes,
            seed_num,
            corruption_level=None,
            theano_rng=None,
            activation=T.nnet.sigmoid
        ):
        self.encode_layers=[]
        self.decode_layers=[]
        self.n_hidden_layers=len(hidden_layers_sizes)
        self.params=[]
        assert self.n_hidden_layers > 0
        assert len(hidden_layers_sizes) == len(seed_num)
        #assert len(hidden_layers_sizes) == len(corruption_levels)

        ##Encode :
        rng_list = [np.random.RandomState(seed = seed) for seed in seed_num]
        encode_order = range(self.n_hidden_layers)##index for the encoder dimension
        for i in encode_order:
            if i==0:
                input_size = visible_size
            else:
                input_size = hidden_layers_sizes[i-1]
            rng = rng_list[i]
            encode_layer = HiddenLayer(rng, in_dim = input_size, out_dim = hidden_layers_sizes[i],activation=activation)
            self.encode_layers.append(encode_layer)
            self.params.extend(encode_layer.params)

        ##Decode in a inversed order of encode

        ##just index for decoder the dimension
        decode_order = [self.n_hidden_layers-i-1 for i in encode_order]
        for j in decode_order:
            if j == 0:
                output_size = visible_size
            else:
                output_size = hidden_layers_sizes[j-1]
            rng = rng_list[i]
            decode_layer = HiddenLayer(rng,in_dim = hidden_layers_sizes[j], out_dim = output_size,W=self.encode_layers[j].W.T,activation=activation)
            self.decode_layers.append(decode_layer)
            self.params.extend([decode_layer.b])

        self.corruption_level = corruption_level
        if not theano_rng:
            numpy_rng = np.random.RandomState(97965)
            self.corruption_seed = RandomStreams(numpy_rng.randint(2 ** 30))
    def get_corrupted_input(self, input, corruption_level):
        return self.corruption_seed.binomial(size = input.shape, n = 1,
                                        p = 1 - corruption_level,
                                        dtype = theano.config.floatX) * input
    def getcost(self,x, target,learning_rate, cost_function=r'logloss'):
        ## forward computation
        assert cost_function in ['euclidean','logloss']
        ## corruption
        if self.corruption_level is None:
            print "corruption_level: None"
            input = x
        else :
            input =  self.get_corrupted_input(x, corruption_level=self.corruption_level)
        self.corrupted = input
        for encode in self.encode_layers:
            output = encode.output(input)
            input = output
        for decode in self.decode_layers:
            output = decode.output(input)
            input = output
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        #cost = T.mean(L)
        if cost_function == 'euclidean':
            cost =  T.sum((target - output)**2)
        if cost_function == 'logloss':
            #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
            cost =  T.mean( - T.sum(target * T.log(output) + (1-target) * T.log(1 - output), axis=1))
        gparams = T.grad(cost, self.params)
        updates = [(param , param - learning_rate * gparam) for param,gparam in zip(self.params , gparams)]
        return cost,updates
    def fit(self, X , Target, iterations=15, batch_size = 10 ,learning_rate=0.15,errors=[], verbose=False,cost_function='logloss'):
        index = T.lscalar()
        x = T.matrix('x')
        target = T.matrix('target')
        n_train_batches = X.shape[0] / batch_size
        ##forward compute the cost
        cost,updates = self.getcost(x, target, learning_rate=learning_rate, cost_function=cost_function)
        X = theano.shared(np.asarray(X, dtype=theano.config.floatX))
        Target = theano.shared(np.asarray(Target, dtype=theano.config.floatX))

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: X[index * batch_size : (index + 1) * batch_size],
                target: Target[index * batch_size : (index + 1) * batch_size]
            }
        )
        if verbose:
            print "type n_train_batches: ", type(n_train_batches)
        start_time = timeit.default_timer()
        for epoch in xrange(iterations):
            c = []
            for batch_index in xrange(n_train_batches):
                #print batch_index
                c.append(train_da(batch_index))## training

            if verbose:
                errors.append(np.mean(c))
                print 'Training epoch %d, cost ' % epoch, np.mean(c)
        end_time = timeit.default_timer()
        training_time = float(end_time - start_time)
        if verbose:
            print 'AutoEncoder run for %.2fm ' % (training_time/60.)
        return errors
    def _geth(self, x):
        for encode in self.encode_layers:
            output = encode.output(x)
            x = output
        return output
    def transform(self,X):
        x = T.matrix('x')
        gethidden = theano.function([x],self._geth(x))
        return gethidden(X)
    def getRecon(self,X):
        def recon(x):
            input = x
            for encode in self.encode_layers:
                output = encode.output(input)
                input = output
            for decode in self.decode_layers:
                output = decode.output(input)
                input = output
            return output
        x = T.matrix('x')
        getRe = theano.function([x],recon(x))
        return getRe(X)

def testDAE(size=(100,10)):
    data = np.random.rand(size[0],size[1])
    data_prime = np.random.rand(size[0],size[1])
    visible_size =size[1]
    hidden_layers_sizes = [5,1]
    seed_num=[0,1]
    corruption_level=None
    dae = DeepAE(visible_size=visible_size, hidden_layers_sizes=hidden_layers_sizes,seed_num=seed_num,corruption_level=corruption_level)
    #print data[0:25]
    iterations=100
    batch_size = 10
    learning_rate=0.05
    dae.fit(X = data,Target = data_prime,iterations=iterations, batch_size = batch_size,learning_rate=learning_rate,)
    print "error",np.sum((dae.getRecon(X = data)-data_prime)**2)
    print "two norm of data",np.sum((data)**2)
def testDAE2():
    import PIL.Image as Image
    import ImShow as I
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
    data = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    #data = np.load(r"/home/czhou2/Documents/notMNIST_small.npk")
    #data = np.load(r"/home/czhou2/Documents/NM_G.npk")
    target = data.copy()
    data = corrupt(data,100)
    features = data.shape[1]
    visible_size = features
    hidden_layers_sizes = [400,256]
    seed_num=[11,17]
    corruption_level=None
    dae = DeepAE(visible_size=visible_size, hidden_layers_sizes=hidden_layers_sizes,seed_num=seed_num,corruption_level=corruption_level)
    iterations=500
    batch_size = 30
    learning_rate=0.1
    costf = r'logloss'
    print "fit, cost function is ", costf
    dae.fit(X = data, Target=target, iterations=iterations, batch_size = batch_size,learning_rate=learning_rate,cost_function=costf,verbose=True)
    print "fit done"
    image_X = Image.fromarray(I.tile_raster_images(X=data,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    print "get corrupted data"
    image_cX = Image.fromarray(I.tile_raster_images(X=target,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_cX.save(r"target.png")
    print "get W"
    W = dae.encode_layers[0].W.get_value(borrow=True).T
    image_W = Image.fromarray(I.tile_raster_images(X=W,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W.save(r"W.png")
    print "get recon"
    R = dae.getRecon(X = data)
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R.save(r"Recon.png")
    print "error",np.sum((dae.getRecon(X = data)-target)**2)
    print "two norm of data",np.sum((data)**2)
def testIter():
    import PIL.Image as Image
    import ImShow as I
    #data = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    data = np.load(r"/home/czhou2/Documents/notMNIST_small.npk")
    #data = np.load(r"/home/czhou2/Documents/NM_G.npk")
    features = data.shape[1]
    visible_size = features
    hidden_layers_sizes = [400,256]
    seed_num=[11,17]
    corruption_level=0.0
    dae = DeepAE(visible_size=visible_size, hidden_layers_sizes=hidden_layers_sizes,seed_num=seed_num,corruption_level=corruption_level)
    iterations=200
    batch_size = 30
    learning_rate=0.1
    costf = r'logloss'
    print "fit, cost function is ", costf
    dae.fit(X = data,iterations=iterations, batch_size = batch_size,learning_rate=learning_rate,cost_function=costf,verbose=True)
    print "fit done"
    image_X = Image.fromarray(I.tile_raster_images(X=data,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    print "get corrupted data"
    input = T.matrix('input')
    get_cX = theano.function([input],dae.get_corrupted_input(input, corruption_level))
    cX =  get_cX(data)
    image_cX = Image.fromarray(I.tile_raster_images(X=cX,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_cX.save(r"cX.png")
    print "get W"
    W = dae.encode_layers[0].W.get_value(borrow=True).T
    image_W = Image.fromarray(I.tile_raster_images(X=W,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W.save(r"W.png")
    print "get recon"
    R = dae.getRecon(X = data)
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R.save(r"Recon.png")

    print "next iteration:"
    ##seconde iteration
    dae.fit(X = R,iterations=iterations, batch_size = batch_size,learning_rate=learning_rate,cost_function=costf,verbose=True)
    print "get W1"
    W1 = dae.encode_layers[0].W.get_value(borrow=True).T
    image_W1 = Image.fromarray(I.tile_raster_images(X=W1,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_W1.save(r"W1.png")
    print "get recon1"
    R1 = dae.getRecon(X = data)
    image_R1 = Image.fromarray(I.tile_raster_images(X=R1,img_shape=(28,28), tile_shape=(20, 20),tile_spacing=(1, 1)))
    image_R1.save(r"Recon1.png")
def testDAE3():
    import PIL.Image as Image
    import ImShow as I
    from sklearn.datasets import load_digits
    digits=load_digits()
    data = digits.data
    #Y = digits.target
    features = data.shape[1]
    visible_size = features
    hidden_layers_sizes = [100]
    seed_num=[7]
    corruption_level=0.0
    dae = DeepAE(visible_size=visible_size, hidden_layers_sizes=hidden_layers_sizes,seed_num=seed_num,corruption_level=corruption_level)
    iterations=500
    batch_size = 40
    learning_rate=0.1
    print "fit"
    dae.fit(X = data,iterations=iterations, batch_size = batch_size,learning_rate=learning_rate,verbose=True)
    print "fit done"
    image_X = Image.fromarray(I.tile_raster_images(X=data,img_shape=(8,8), tile_shape=(30, 30),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    print "get recon"
    R = dae.getRecon(X = data)
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=(8,8), tile_shape=(30, 30),tile_spacing=(1, 1)))
    image_R.save(r"Recon.png")
def testHiddenLayer(size = (10,100), seed = 0):
    rng = np.random.RandomState(seed = seed)
    value = np.random.rand(size[0],size[1])
    print type(value)
    print value.shape
    input_x = T.matrix('x')
    n_in = size[1]
    n_out = 3
    h = HiddenLayer(rng=rng, in_dim = n_in , out_dim = n_out)
    op = h.output(input_x)
    out = theano.function([input_x], op)
    print out(value)
if __name__=="__main__":
    #testHiddenLayer()
    testDAE2()
    #testIter()
