import numpy as np

def onePixel_uniformNoise(data, corNum=10):
    corruption_index = map(np.int,np.floor(np.random.random_sample(size = corNum)*data.shape[1]))
    for i in range(data.shape[0]):
        for j in corruption_index:
            #corrupted[i,j] = corrupted[i,j] + np.random.normal(loc=0.5, scale=0.25)
            data[i,j] = np.random.uniform()
    return data

def onePixel_fixedNoise(data, corNum=10):
    corruption_index = map(np.int,np.floor(np.random.random_sample(size = corNum)*data.shape[1]))
    for j in corruption_index:
        corruption_amplitude = np.random.random()
        data[:,j] = corruption_amplitude
    return data
def onePixel_GaussianNoise(data, corNum=10):
    corruption_index = map(np.int,np.floor(np.random.random_sample(size = corNum)*data.shape[1]))
    for i in range(data.shape[0]):
        for j in corruption_index:
            #corrupted[i,j] = corrupted[i,j] + np.random.normal(loc=0.5, scale=0.25)
            data[i,j] = np.random.normal(loc=0, scale=1)
    return data