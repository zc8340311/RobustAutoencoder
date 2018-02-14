import PIL.Image as Image
import ImShow as I
import numpy as np
import tensorflow as tf
import DAE_tensorflow as DAE
import RDAE_tensorflow as RDAE

def compare_RDAE_DAE(X, layers, lamda, folder, learning_rate = 0.15, inner = 100, outer = 10, batch_size = 133,inputsize = (28,28)):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = DAE.Deep_Autoencoder(sess = sess, input_dim_list = layers)
            ae.fit(X = X ,sess = sess, learning_rate = learning_rate, iteration = inner * outer, batch_size = batch_size, verbose=True)

            dR = ae.getRecon(X = X, sess = sess)
            dH = ae.transform(X, sess)
            Image.fromarray(I.tile_raster_images(X=dR,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"dR.png")
            dH.dump("dH.pkl")
            
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rae = RDAE.RDAE(sess = sess, lambda_ = lamda * 10, layers_sizes = layers)
            rL, rS = rae.fit(X = X ,sess = sess, learning_rate = learning_rate, batch_size = batch_size, inner_iteration = inner, iteration = outer, verbose=True)
            rR = rae.getRecon(X, sess)
            rH = rae.transform(X, sess)
            Image.fromarray(I.tile_raster_images(X=rR,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"rR.png")
            Image.fromarray(I.tile_raster_images(X=rS,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"rS.png")
            Image.fromarray(I.tile_raster_images(X=rL,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"rL.png")
            rH.dump(r"rH.pkl")
            rS.dump("rS.pkl")
    os.chdir("../")