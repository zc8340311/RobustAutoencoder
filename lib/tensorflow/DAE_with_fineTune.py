import numpy as np
import tensorflow as tf
import DAE_tensorflow as DAE

class DAE(DAE.Deep_Autoencoder):
    def fineTune(self,X,y):
        
