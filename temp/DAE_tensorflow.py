class Deep_Autoencoder():
    def __init__(self, sess, input_dim_list=[784,400]):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        ## Encoder initialize
        for i in range(len(input_dim_list)-1):
            
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i],self.dim_list[i+1]],-0.1,0.1)))
            
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]],-0.1,0.1)))
            
        for i in range(len(input_dim_list)-2,-1,-1):
            
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]],-0.1,0.1)))
        sess.run(tf.initialize_all_variables())
        
    def fit(self, X, sess, learning_rate=0.15,iteration=200,batch_size=50,verbose=False):
        input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        ## coding phase :
        last_layer = input_x
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        ## decode phase
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        recon = last_layer
        
        cost = tf.reduce_sum(tf.square(input_x - recon))
        
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
                print "iteration : ", i ,", cost : ", e
            error.append(e)
        
        return error
    def transform(self, X, sess):
        new_input = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        last_layer = new_input
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        return hidden.eval(session = sess,feed_dict={new_input:X})
    
    def get_recon(self, X, sess):
        hidden_data = self.transform(X, sess)
        hidden_layer = tf.placeholder(tf.float32,[None,self.dim_list[-1]])
        last_layer = hidden_layer
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        recon = last_layer
        return recon.eval(session = sess,feed_dict={hidden_layer:hidden_data})
