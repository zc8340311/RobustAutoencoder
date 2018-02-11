import numpy as np
import SparseAutoencoder as l1sdae
import SparseKLAutoencoder as klsdae
import Sparsel21Autoencoder as l21sdae
import os
import tensorflow as tf
def one_run(x, layers, beta):

    with tf.Session() as sess:
        klsae = klsdae.Sparse_KL_Deep_Autoencoder(sess = sess, input_dim_list=layers,
                                        sparsities=[beta] * (len(layers)-1), sparse_ratios = [0.2] * (len(layers)-1) )
        error = klsae.fit(x, sess = sess, iteration = 400,batch_size=100,verbose = True)
        klh = klsae.transform(x,sess=sess)
        klh.dump("klh.npk")
        print "KL h shape",klh.shape
    with tf.Session() as sess:

        l21sae = l21sdae.Sparsel21_Deep_Autoencoder(sess = sess, input_dim_list=layers, sparsities=[beta] * (len(layers)-1))
        error = l21sae.fit(x, sess = sess, iteration = 400,batch_size=100,verbose = True)
        l21h = l21sae.transform(x,sess=sess)
        l21h.dump("l21h.npk")
        print "l21 h shape",l21h.shape

    with tf.Session() as sess:
        l1sae = l1sdae.Sparse_Deep_Autoencoder(sess = sess, input_dim_list=layers,sparsities=[beta] * (len(layers)-1))
        error = l1sae.fit(x, sess = sess, iteration = 400,batch_size=100,verbose = True)
        l1h = l1sae.transform(x,sess=sess)
        l1h.dump("l1h.npk")
        print "l1h shape",l1h.shape

if __name__ == "__main__":
    x = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    lam_list = [0.001,0.01,0.1,1.0,10.0,100.0]
    layers = [784,784]

    for lam in lam_list:
        if not os.path.isdir(str(lam)):
            os.makedirs(str(lam))
        os.chdir(str(lam))
        one_run(x=x, layers=layers, beta = lam)
        os.chdir("../")
