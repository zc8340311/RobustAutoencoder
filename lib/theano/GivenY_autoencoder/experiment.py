import PIL.Image as Image
import ImShow as I
import numpy as np
import DeepRAE_target as DRAE
import DeepAE_target as DAE
import os
def compareDRAEandDAE(X, target,inputsize, hidden_layers_sizes, seed_num, learning_rate, lamda, batch_size, corruption_level, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    image_X = Image.fromarray(I.tile_raster_images(X = X, img_shape = inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    image_X = Image.fromarray(I.tile_raster_images(X = target, img_shape = inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"target.png")
    drae = DRAE.DRAE_target(hidden_layers_sizes = hidden_layers_sizes, seeds = seed_num ,lambda_=lamda, error = 0.001,
                corruption_level = corruption_level)
    L , S = drae.fit(X, Target=target ,learning_rate=learning_rate, inner_iteration = 8, iteration = 50,verbose = True,batch_size = batch_size)
    #print rae.transform(X).shape
    image_S = Image.fromarray(I.tile_raster_images(X=S,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_S.save(r"S.png")
    R = drae.getRecon()
    image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_R.save(r"rRecon.png")
    #W = drae.get_Weights(0)
    #image_W = Image.fromarray(I.tile_raster_images(X=W, img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    #image_W.save(r"rW1.png")
    rH = drae.transform(X)
    rH.dump(r"rH.npk")
    R.dump(r"rR.npk")
    S.dump(r"rS.npk")
    np.array(drae.errors).dump(r"DRAEerror.npk")


    dae = DAE.DeepAE_target(visible_size=inputsize[0]*inputsize[1], hidden_layers_sizes=hidden_layers_sizes,seed_num=seed_num,corruption_level=corruption_level)
    error = []
    iteration = 8 * 50
    dae.fit(X = X,Target=target,iterations= iteration, batch_size = batch_size,learning_rate=learning_rate,errors=error,verbose=True)
    np.array(error).dump(r"DAEerror.npk")
    dH = dae.transform(X)
    dH.dump(r"dh.npk")
    #W1 = dae.encode_layers[0].W.get_value(borrow=True).T
    #image_W1 = Image.fromarray(I.tile_raster_images(X=W1,img_shape=inputsize, tile_shape=(20, 20),tile_spacing=(1, 1)))
    #image_W1.save(r"deW1.png")
    print "get recon1"
    R1 = dae.getRecon(X = X)
    R1.dump(r"dR.npk")
    image_R1 = Image.fromarray(I.tile_raster_images(X=R1,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_R1.save(r"dRecon1.png")
    os.chdir('../')
def add_noise(X,ratio=0.1,seed=1):
    dim = X.shape[1]
    np.random.seed(seed)
    for line in X:
        index = np.random.choice(range(dim),int(ratio*dim),replace=False)
        print index
        for i in index:
            if line[i] == 0:
                line[i] = 1
            else:
                line[i] = 0
def corrupt(X,corNum=10):
    N,p = X.shape[0],X.shape[1]
    for i in xrange(N):
        loclist = np.random.randint(0, p, size = corNum)
        for j in loclist:
            if X[i,j] > 0.5:
                X[i,j] = 0
            else:
                X[i,j] = 1
    return X
if __name__ =="__main__":
    # X = np.load(r"/home/czhou2/Documents/packets_1000.npk")
    # X = X[:,:1512]/float(np.max(X))
    # inputsize=(36,42)
    # data_loc = r"/home/czhou2/Documents/00001/packets.npy"
    # data_loc = r"/home/czhou2/Documents/00000/packets.npy"
    # data_loc = r"/home/czhou2/Documents/maccdc2012.npy"
    # import cPickle
    # loaded = cPickle.load(open( "subnet_202_101_and_28_1.pkl", "rb" ))
    # X = np.load(data_loc)
    # X = np.delete(X,range(30,34),1)
    # X = X[ loaded, :1512]/float(np.max(X))
    #X = X[ :, :1512]/float(np.max(X))

    # X = np.delete(X, range(30,38), 1)
    # X = X[loaded]
    # em = np.zeros(2 * X.shape[0]).reshape(X.shape[0], 2)
    # X = np.concatenate((X, em), axis=1)
    # X = X/float(np.max(X))
    # add_noise(X,ratio=0.002,seed=46432)
    #
    # inputsize=(36,42)



    #X= np.load(r"/home/czhou2/Documents/corruption.pkl")
    #X= np.load(r"/home/czhou2/Documents/train_x_small.pkl")
    #X= np.load(r"/home/czhou2/Documents/train_x.pkl")
    #X= np.load(r"/home/czhou2/Documents/twowith7.pkl")
    #X= np.load(r"/home/czhou2/Documents/twowith7_uncor.pkl")
    #X= np.load(r"/home/czhou2/Documents/twowith7_only_cor.pkl")
    #X= np.load(r"/home/czhou2/Documents/twowith7_only_uncor.pkl")



    inputsize = (28,28)

    hidden1 = (25,25)
    hidden2 = (15,15)
    hidden3= (10,10)
    #
    hidden_layers_sizes = [hidden1[0]*hidden1[1],hidden2[0]*hidden2[1],hidden3[0]*hidden3[1]]
    seed_num=[33,786,3231]
    learning_rate = 0.1
    batch_size = 30
    corruption_level = 0.0

    cor_list = [50,100,150,200,250,300,350,400,450]
    lam_list = [1.,10.,15.,20.,25.,50.,70.]
    for cor in cor_list:
        X= np.load(r"/home/czhou2/Documents/train_x_small.pkl")
        target = X.copy()
        X = corrupt(X, corNum=cor)
        up_folder = r"Cor"+str(cor)
        if not os.path.isdir(up_folder):
            os.makedirs(up_folder)
        os.chdir(up_folder)
        for lamda in lam_list:
            folder = r"lam" + str(lamda)
            compareDRAEandDAE(X = X, target= target,inputsize = inputsize, hidden_layers_sizes = hidden_layers_sizes,
                        seed_num = seed_num, learning_rate = learning_rate, lamda=lamda, batch_size = batch_size,
                        corruption_level=corruption_level,folder=folder)
        os.chdir('../')
