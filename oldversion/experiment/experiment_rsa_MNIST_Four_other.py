import numpy as np
import RobustSparseAutoencoder as RSA
import ImShow as I
import tensorflow as tf
import os

def oneRun(x,layers_sizes,sparsities,lam):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rsa = RSA.RobustSparseAutoencder(sess, layers_sizes = layers_sizes, sparsities = sparsities, lambda_=lam)
            L,S = rsa.fit(x, sess, learning_rate=0.001, inner_iteration = 20, iteration = 20, batch_size = 17,
                          verbose=False)
            L[:100].dump("L_100.npk")
            S[:100].dump("S_100.npk")
            np.linalg.norm(S,ord=1,axis=1).dump("S_result.npk")
            # hidden = rsa.transform(x, sess = sess)
            # hidden[:1000].dump("hidden.npk")

if __name__ == "__main__":
    sparce_list = np.arange(0.001,0.05,0.01)
    # lam_list = [0.0,0.05,0.1,0.2,0.3,0.5,1.0,3.0,3.5,4.0,4.5,5.0]
    lam_list = [6.,7.,8.,9.,11.]
    x = np.load(r"/home/czhou2/Documents/Four_with_other_p5.pkl")
    layers_sizes = [784,600,400]
    result_target_folder = r"/research/czhou2/Result/rsa_tuned_layer_FourWithOther_2"
    if not os.path.isdir(result_target_folder):
        os.makedirs(result_target_folder)
    os.chdir(result_target_folder)
    for sparce in sparce_list:
        if not os.path.isdir(str(sparce)):
            os.makedirs(str(sparce))
        os.chdir(str(sparce))
        for lam in lam_list:
            if not os.path.isdir(str(lam)):
                os.makedirs(str(lam))
            os.chdir(str(lam))
            oneRun(x=x,layers_sizes = layers_sizes,sparsities = [sparce]*(len(layers_sizes)-1), lam = lam)

            os.chdir("../")
        os.chdir("../")
