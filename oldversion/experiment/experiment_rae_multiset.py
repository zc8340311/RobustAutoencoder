import numpy as np
import Sparsel21Autoencoder as SA
# import PIL.Image as Image
import scipy.misc
import os
import tensorflow as tf
import RobustSparseAutoencoder as RSA
from sklearn.preprocessing import MinMaxScaler as MMS
def pred(x):
    if x == 0:
        return 0
    else:
        return 1

def one_run(X,dim, sparsities, lambda_ ,inner,outter):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rsa = RSA.RobustSparseAutoencder(sess = sess, layers_sizes=dim, sparsities=sparsities, lambda_= lambda_)

            L, S = rsa.fit(X, sess, learning_rate=0.005, inner_iteration = inner,iteration=outter, batch_size=100)
            ##save S to see the sparsity
            scipy.misc.imsave(r"S.png",S)

            hidden = rsa.transform(X, sess)
            hidden.dump("RSA_hidden.npk")

            prediction = np.array(map(pred,np.linalg.norm(S,ord=1,axis=1)))
            prediction.dump("Outlier_prediction.npk")
if __name__ == "__main__":
    folder = r"/home/zc8304/Documents/OutlierDatasets"
    topics = ['arrhythmia_X.csv', 'glass_X.csv', 'letter_X.csv', 'lympho_X.csv', 'musk_X.csv', 'satellite_X.csv', 'satimage_2_X.csv', 'speech_X.csv', 'wbc_X.csv']
    print topics
    sparse_list = np.arange(0.0,0.5,0.05)
    lam_list = np.arange(0.0,0.5,0.05)
    inner = 25
    outter = 12
    for topic in topics:
        topic_name = topic.split('_')[0]
        if not os.path.isdir(topic_name):
            os.makedirs(topic_name)
        os.chdir(topic_name)
        X = np.loadtxt(folder+r"/"+topic,delimiter=',')
        X = MMS().fit_transform(X)
        dim = [X.shape[1],X.shape[1]]
        for sparse in sparse_list:
            sparse_name = str(sparse)
            sparse = [sparse]
            if not os.path.isdir(sparse_name):
                os.makedirs(sparse_name)
            os.chdir(sparse_name)
            for lam in lam_list:
                lam_level = str(lam)
                if not os.path.isdir(lam_level):
                    os.makedirs(lam_level)
                os.chdir(lam_level)
                one_run(X=X,dim=dim, sparsities=sparse, lambda_ = lam,inner=inner,outter=outter)
                os.chdir('../')
            os.chdir('../')
        os.chdir('../')
