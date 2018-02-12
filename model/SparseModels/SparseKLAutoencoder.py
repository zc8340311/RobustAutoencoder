import tensorflow as tf
import numpy as np

class KL_Sparse_Autoencoder():
    def __init__(self, sess, input_dim_list=[784,400], sparsities = [0.5],sparse_ratios = [0.2]):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        self.sparsities = sparsities
        self.penal_term = []
        ## Encoders parameters
        for i in range(len(input_dim_list)-1):
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i],self.dim_list[i+1]],-0.5,0.5)))
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]],-0.5,0.5)))
        ## Decoders parameters
        for i in range(len(input_dim_list)-2,-1,-1):
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]],-0.5,0.5)))

        self.input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        self.sparse_ratios = sparse_ratios
        ## coding graph :
        hidden_layer_count = 0
        last_layer = self.input_x
        for weight,bias,sparse_ratio in zip(self.W_list,self.encoding_b_list,self.sparse_ratios):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            self.penal_term.append(self.KL(hidden,sparse_ratio))
            last_layer = hidden
        ## decode graph:
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        recon = last_layer

        ## cost definition
        cost = tf.losses.log_loss(recon, self.input_x)
        ## add penalty term
        for penalty_term, sparcity in zip(self.penal_term,self.sparsities):
            cost += sparcity * penalty_term

        self.cost = cost

        sess.run(tf.global_variables_initializer())

    def KL(self,hidden, sparsity):
        hidden_avg = tf.reduce_sum(hidden,0) / tf.to_float(tf.shape(hidden)[0])
        kl_list = sparsity * tf.log(sparsity / hidden_avg) + (1 - sparsity) * tf.log((1-sparsity)/(1-hidden_avg))
        return tf.reduce_sum(kl_list)

    def fit(self, X, sess, learning_rate=0.01,
            iteration=200, batch_size=50, verbose=False):
        assert X.shape[1] == self.dim_list[0]

        opt = tf.train.GradientDescentOptimizer(learning_rate)

        train_step = opt.minimize(self.cost)

        error = []
        sample_size = X.shape[0]

        def batches(l, n):
            """Yield successive n-sized chunks from l."""
            for i in xrange(0, l, n):
                yield range(i,min(l,i+n))

        for i in xrange(iteration):
            for one_batch in batches(sample_size, batch_size):
                sess.run(train_step,feed_dict = {self.input_x:X[one_batch]})

            if verbose:
                e = self.cost.eval(session = sess,feed_dict = {self.input_x: X[one_batch]})
                error.append(e)
                if i%20==0:
                    print "    iteration : ", i ,", cost : ", e
        return error
    def transform(self, X, sess):
        new_input = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        last_layer = new_input
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        return hidden.eval(session = sess, feed_dict={new_input: X})

    def getRecon(self, X, sess):
        hidden_data = self.transform(X, sess)
        hidden_layer = tf.placeholder(tf.float32,[None,self.dim_list[-1]])
        last_layer = hidden_layer
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        recon = last_layer
        return recon.eval(session = sess,feed_dict={hidden_layer:hidden_data})
if __name__ == '__main__':
    x = np.load(r"/home/czhou2/Documents/train_x_small.pkl")

    with tf.Session() as sess:
        sae = Sparse_KL_Deep_Autoencoder(sess = sess, input_dim_list=[784,784,784],sparsities=[0.5,0.5],sparse_ratios = [0.2,0.2])
        print "x type",x.shape,x.dtype
        error = sae.fit(x, sess = sess, iteration = 400,batch_size = 100,verbose = True)
        print error
        h = sae.transform(x,sess=sess)
        print "h shape",h.shape
        R = sae.getRecon(x,sess=sess)
        print "R",R.shape,R.dtype
        error = sae.fit(R, sess = sess, iteration = 400,batch_size = 100,verbose = True)
        print error
