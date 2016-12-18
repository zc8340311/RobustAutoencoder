import numpy as np
import tensorflow as tf
import DAE_tensorflow as DAE_normal

class DAE(DAE_normal.Deep_Autoencoder):
    def fineTune(self,sess, X, y, learning_rate=0.05,iteration=200, batch_size=49):
        assert X.shape[1] == self.dim_list[0]
        input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        ## encoding
        last_layer = input_x
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        self.predict_W = tf.Variable(tf.zeros([self.dim_list[-1],1]))
        self.predict_b = tf.Variable(tf.zeros([1]))
        y_hat = tf.matmul(last_layer,self.predict_W) + self.predict_b
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_hat))
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = opt.minimize(cross_entropy)

        e = cross_entropy.eval(session = sess,feed_dict = {input_x: X})
        print "cost before fine tune: ", e
        sample_size = X.shape[0]
        turns = sample_size / batch_size
        for i in xrange(iteration):
            for turn in xrange(turns):

                start = (batch_size*turn) % sample_size
                end = (batch_size*(turn+1)) % sample_size

                sess.run(train_step,feed_dict = {input_x:X[start:end]})
        e = cross_entropy.eval(session = sess,feed_dict = {input_x: X})
        print "cost after fine tune: ", e

        return y_hat.eval(session = sess, feed_dict={input_x: X})

if __name__=="__main__":
    X = np.load(r"/home/zc8304/Documents/train_x_small.pkl")
    y = np.load(r"/home/zc8304/Documents/train_y_small.pkl")
    print "X shape, y shape", X.shape, y.shape
    with tf.Session() as sess:
        ae = DAE(sess = sess, input_dim_list=[784,625,225,100])

        error = ae.fit(X ,sess = sess, learning_rate=0.1, batch_size = 40, iteration = 100, verbose=True)
        print "error",error
        print "predict:",ae.fineTune(sess, X, y)
