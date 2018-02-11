import numpy as np
import Sparsel21Autoencoder as SA
# import ImShow as I
# import PIL.Image as Image
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler as MMS

def one_run(X,dim, sparsities, iteration):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            sae = SA.Sparsel21_Deep_Autoencoder(sess, input_dim_list = dim, sparsities = sparsities)
            _ = sae.fit(X=X, sess=sess, learning_rate=0.01, iteration=iteration, batch_size=300, verbose=False)
            hidden = sae.transform(X, sess)
            hidden.dump("hidden.npk")

            # Recon = sae.getRecon(X, sess)
            # Image.fromarray(I.tile_raster_images(X=Recon,img_shape=(28,28), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"Recon.png")

if __name__ == '__main__':
    sparse_list = [[i] for i in np.arange(0.0,0.5,0.05)]
    folder = r'/home/zc8304/Documents/Raytheon/CleanedUserProg212'
    topics = [folder+'/'+x for x in os.listdir(folder)]
    for f in topics:
        case = f.split('/')[-1].split('+')[1]
        if not os.path.isdir(case):
            os.makedirs(case)
        os.chdir(case)

        X = np.load(f)
        index = np.random.choice(range(X.shape[0]),size=8000,replace=False)
        X = X[index]
        X = MMS().fit_transform(X)
        dim = [X.shape[1],X.shape[1]]
        for sparse in sparse_list:
            sparse_case_name = "_".join([str(x) for x in sparse])
            if not os.path.isdir(sparse_case_name):
                os.makedirs(sparse_case_name)
            os.chdir(sparse_case_name)
            one_run(X=X,dim=dim, sparsities=sparse, iteration=300)
            os.chdir('../')
        os.chdir('../')
