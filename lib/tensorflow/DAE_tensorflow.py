import tensorflow as tf
import numpy as np
class Deep_Autoencoder():
    def __init__(self, sess, input_dim_list=[784,400]):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        ## Encoders parameters
        for i in range(len(input_dim_list)-1):
            
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i],self.dim_list[i+1]],-0.1,0.1)))
            
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]],-0.1,0.1)))
        ## Decoders parameters 
        for i in range(len(input_dim_list)-2,-1,-1):
            
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]],-0.1,0.1)))
        sess.run(tf.initialize_all_variables())
        
    def fit(self, X, sess, learning_rate=0.15, 
            iteration=200, batch_size=50, verbose=False):
        assert X.shape[1] == self.dim_list[0]
        
        input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        
        ## coding graph :
        last_layer = input_x
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        ## decode graph:
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        recon = last_layer
        
        #cost = tf.reduce_mean(tf.square(input_x - recon))
        cost = 200 *tf.contrib.losses.log_loss(recon, input_x)

        opt = tf.train.GradientDescentOptimizer(learning_rate) 
                        
        train_step = opt.minimize(cost)
        
        error = []
        sample_size = X.shape[0]
        turns = sample_size / batch_size
        
        for i in xrange(iteration):
            for turn in xrange(turns):

                start = (batch_size*turn) % sample_size
                end = (batch_size*(turn+1)) % sample_size

                sess.run(train_step,feed_dict = {input_x:X[start:end]})
            e = cost.eval(session = sess,feed_dict = {input_x: X[start:end]})
            if verbose and i%20==0:
                print "    iteration : ", i ,", cost : ", e
            error.append(e)
        
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
if __name__ == "__main__":
    x = np.load(r"/home/zc/Documents/train_x_small.pkl")
    
    sess = tf.Session()
    ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,225,100])

    error = ae.fit(x ,sess = sess, learning_rate=0.1, batch_size = 40, iteration = 100, verbose=True)

    recon1 = ae.getRecon(x, sess)

    sess.close()   
    print error



