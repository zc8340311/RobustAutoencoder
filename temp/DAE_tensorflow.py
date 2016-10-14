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
        
sess = tf.Session()
ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,500,200])
