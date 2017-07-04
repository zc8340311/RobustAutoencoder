import numpy as np
import Sparsel21Autoencoder as SA
import ImShow as I
import PIL.Image as Image
import os
import tensorflow as tf
def one_run(X,dim, sparsities, iteration):
    with tf.Session() as sess:
        sae = SA.Sparsel21_Deep_Autoencoder(sess, input_dim_list = dim, sparsities = sparsities)
        _ = sae.fit(X=X, sess=sess, learning_rate=0.05, iteration=iteration, batch_size=100, verbose=False)
        hidden = sae.transform(X, sess)
        hidden.dump("hidden.npk")

        Recon = sae.getRecon(X, sess)
        Image.fromarray(I.tile_raster_images(X=Recon,img_shape=(28,28), tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"Recon.png")


if __name__ == "__main__":
    dim_list = [[784,400],[784,600],[784,784],[784,1000],[784,1200]]
    sparse_list = [[i] for i in np.arange(0.0,0.5,0.04)]
    X = np.load(r'/home/zc8304/Documents/train_x_small.pkl')
    for dim in dim_list:
        dimfolder = "_".join([str(a) for a in dim])
        if not os.path.isdir(dimfolder):
            os.makedirs(dimfolder)
        os.chdir(dimfolder)
        for sp in sparse_list:
            spfolder = "".join([str(a) for a in sp])
            if not os.path.isdir(spfolder):
                os.makedirs(spfolder)
            os.chdir(spfolder)

            one_run(X=X,dim=dim, sparsities = sp, iteration=400)

            os.chdir('../')
        os.chdir('../')
