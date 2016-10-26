import PIL.Image as Image
#import ImShow as I
import numpy as np
import RDAE_tensorflow as RDAE
import os
from collections import Counter
import tensorflow as tf

def tune_RDAE(x, lam_list = [50], learn_rates = [0.1], inner = 50, outter = 5):

    sess = tf.Session()
    sparsities = []
    error = []
    for lam in lam_list:
        for rate in learn_rates:
            rae = RDAE.RDAE(sess = sess, lambda_ = lam, layers_sizes =[784,400,225,100,50])
            L, S = rae.fit(x ,sess = sess, learning_rate = rate, batch_size = 130, inner_iteration = inner, iteration=outter, verbose=True)
            recon_rae = rae.getRecon(x, sess = sess)
            error.append([lam,np.sum(X-L-S)])
            sparsities.append([lam,Counter(S.reshape(S.size))[0]])
            ## save the training error
            #error_file_name = "lam"+str(lam)+"_rate"+str(rate)+"_inner"+str(inner)+"_outter"+str(outter)+".pkl"
            #np.array(rae.errors).dump(error_file_name)
	sess.close()
	return error,sparsities



if __name__ == "__main__":

    X = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    lam_list = range(50,130,10)
    learn_rates = [0.1]
    inner = 50
    outter = 5

    error,sp = tune_RDAE(x = X, lam_list = lam_list, learn_rates = learn_rates, inner = inner, outter = outter)
    np.array(error).dump(r"diff_X_L_S.pkl")
    np.array(sp).dump(r"sparsities")
