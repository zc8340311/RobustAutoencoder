import numpy as np
import os
import tensorflow as tf
import scipy.misc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
import RobustSparseAutoencoder as RSA
def one_run(X,dim, sparsities, lambda_ ,inner,outter):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rsa = RSA.RobustSparseAutoencder(sess = sess, layers_sizes=dim, sparsities=sparsities, lambda_= lambda_)

            L, S = rsa.fit(X, sess, learning_rate=0.005, inner_iteration = inner,iteration=outter, batch_size=200)
            ##save S to see the sparsity
            scipy.misc.imsave(r"S.png",S)
            S.dump("S.npk")
            hidden = rsa.transform(X, sess)
            hidden.dump("RSA_hidden.npk")

if __name__ == '__main__':
    sparse_list = [[i] for i in np.arange(0.0,0.5,0.05)]
    lam_list = np.arange(0.0,0.5,0.05)
    folder = r'/research/czhou2/215'
    topics = [folder+'/'+x for x in os.listdir(folder)]
    inner = 25
    outter = 12
    for f in topics:
        case = f.split('/')[-1].split('+')[1]
        if not os.path.isdir(case):
            os.makedirs(case)
        os.chdir(case)

        # X = np.load(f)
        X = np.array(pd.read_csv(f,header=0))
        index = np.random.choice(range(X.shape[0]),size=8000,replace=False)
        X = X[index]
        X = MMS().fit_transform(X)
        dim = [X.shape[1],X.shape[1]]
        for sparse in sparse_list:
            sparse_case_name = "_".join([str(x) for x in sparse])
            if not os.path.isdir(sparse_case_name):
                os.makedirs(sparse_case_name)
            os.chdir(sparse_case_name)

            for lam in lam_list:
                lam_level = str(lam)
                if not os.path.isdir(lam_level):
                    os.makedirs(lam_level)
                os.chdir(lam_level)
                one_run(X = X, dim = dim, sparsities = sparse, lambda_ = lam, inner = inner, outter = outter)
                os.chdir('../')
            os.chdir('../')
        os.chdir('../')
