import PIL.Image as Image
import ImShow as I
import numpy as np
import tensorflow as tf
import DAE_tensorflow as DAE
import RDAE_tensorflow as RDAE
import l21RobustAutoencoder as l21RDAE
import os


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

def tune_l21RDAE(x, lam_list = [50], learn_rates = [0.1], inner = 150, outter = 10, batch_size=133):
    #x = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    with tf.Session() as sess:
        rae = l21RDAE.RobustL21Autoencoder(sess = sess, lambda_= lam_list[0], layers_sizes=[784,400,255,100])

        L, S = rae.fit(x, sess = sess, inner_iteration = inner, iteration = outter, batch_size = batch_size, verbose = True)

        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,400,225,100])

        error = ae.fit(x ,sess = sess, learning_rate = learn_rates[0], iteration = inner * outer, batch_size = batch_size ,verbose=True)
    return rae.errors,error

def compare_RDAE_DAE_l21RDAE(X, layers, lamda, folder, learning_rate = 0.15, inner = 100, outer = 10, batch_size = 133,inputsize = (28,28)):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)


    with tf.Session() as sess:
        ae = DAE.Deep_Autoencoder(sess = sess, input_dim_list = layers)
        error = ae.fit(X = X ,sess = sess, learning_rate = learning_rate, iteration = inner * outer, batch_size = batch_size, verbose=True)
        dR = ae.getRecon(X = X, sess = sess)
        dH = ae.transform(X, sess)
        Image.fromarray(I.tile_raster_images(X=dR,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"dR.png")
        dH.dump("dH.pkl")
        np.array(error).dump(r"DAEerror.pkl")

        rael21 = l21RDAE.RobustL21Autoencoder(sess = sess, lambda_= lamda, layers_sizes=layers)
        l21L, l21S = rael21.fit(X = X, sess = sess, inner_iteration = inner, iteration = outer, batch_size = batch_size, learning_rate = learning_rate,  verbose = True)
        l21R = rael21.getRecon(X = X, sess = sess)
        l21H = rael21.transform(X, sess)
        Image.fromarray(I.tile_raster_images(X=l21S,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21S.png")
        Image.fromarray(I.tile_raster_images(X=l21R,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21R.png")
        Image.fromarray(I.tile_raster_images(X=l21L,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"l21L.png")
        l21H.dump(r"l21H.pkl")
        np.array(rael21.errors).dump(r"l21error.pkl")


        rae = RDAE.RDAE(sess = sess, lambda_ = lamda, layers_sizes = layers)
        rL, rS = rae.fit(X = X ,sess = sess, learning_rate = learning_rate, batch_size = batch_size, inner_iteration = inner, iteration = outer, verbose=True)
        rR = rae.getRecon(X, sess)
        rH = rae.transform(X, sess)
        Image.fromarray(I.tile_raster_images(X=rR,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"rR.png")
        Image.fromarray(I.tile_raster_images(X=rS,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"rS.png")
        Image.fromarray(I.tile_raster_images(X=rL,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1))).save(r"rL.png")
        rH.dump(r"rH.pkl")
        np.array(rae.errors).dump(r"RDAEerror.pkl")
if __name__ == "__main__":

    X = np.load(r"/home/czhou2/Documents/mnist_noise_variations_all_1_data_small.pkl")

    inner = 50
    lamda_list = [0.1,1.,5.,10.,15.,20.,25.,50.,70.,100.]

    # error,sp = tune_RDAE(x = X, lam_list = lam_list, learn_rates = learn_rates, inner = inner, outter = outter, batch_size )
    # np.array(error).dump(r"diff_X_L_S.pkl")
    # np.array(sp).dump(r"sparsities")
    #error,sp = tune_l21RDAE(x = X, lam_list = lam_list, learn_rates = learn_rates, inner = inner, outter = outter)
    image_X = Image.fromarray(I.tile_raster_images(X = X, img_shape = (28,28), tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    for lam in lamda_list:
        folder = "lam" + str(lam)
        compare_RDAE_DAE_l21RDAE(X, layers=[784,625,400,225], lamda = lam, folder = folder, learning_rate = 0.15, inner = 100, outer = 10, batch_size = 133,inputsize = (28,28))
