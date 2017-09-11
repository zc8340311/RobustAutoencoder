import numpy as np
import RobustL1SparseAutoencoder as RSA
import ImShow as I
import tensorflow as tf
import os

def oneRun(x,layers_sizes,sparsities,lam):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rsa = RSA.Robust_L1_SparseAutoencder(sess, layers_sizes = layers_sizes, sparsities = sparsities, lambda_=lam)
            L,S = rsa.fit(x, sess, learning_rate=0.05, inner_iteration = 40, iteration = 10, batch_size = 100,
                          verbose=False)
            L[:200].dump("L_short.npk")
            S.dump("S.npk")
            # hidden = rsa.transform(x, sess = sess)
            # hidden[:1000].dump("hidden.npk")

if __name__ == "__main__":
    sparce_list = np.arange(0.0,0.3,0.01)
    lam_list = [0.0,0.05,0.1,0.2,0.3,0.5,1.0,3.0,3.5,4.0,4.5,5.0]
    #lam_list = [1.5,2.0,2.5,5.5,6.0,6.5,7.0,7.5,8.0,8.5]
    x = np.load(r"/home/czhou2/Documents/4_other_x.npk")
    layers_sizes = [784,784,784,784]
    for sparce in sparce_list:

        if not os.path.isdir(str(sparce)):
            os.makedirs(str(sparce))
        os.chdir(str(sparce))
        for lam in lam_list:
            if not os.path.isdir(str(lam)):
                os.makedirs(str(lam))
            os.chdir(str(lam))
            oneRun(x=x,layers_sizes = layers_sizes,sparsities = [sparce] * (len(layers_sizes)-1), lam = lam)

            os.chdir("../")
        os.chdir("../")
