# Robust Autoencoder
Robust autoencoder is a model that combines Autoencoder and Robust PCA which can detect both noise and outliers. This repo offers an implementation based on Tensorflow.
## Updates
02/12/2018: remove theano implementation. 

02/14/2018: clean up codes and put implementation into model/ 

04/06/2018: Thanks to [Tengke-Xiong](https://github.com/Tengke-Xiong). delete wrong part on l21shrink. 

12/13/2018: Thanks to [Roberto](https://github.com/robroe-tsi). change getRecon function which will accept X instead of L. This change allows Robust Autoencoder can detect anomalies in new data.
## Prerequisities

- Python 2.7
- Numpy
- Tensorflow

# Shortcut:
 - **Denoising Model** with l1 regularization on S is at:<br>
["l1 Robust Autoencoder"](https://github.com/zc8340311/RobustAutoencoder/blob/master/model/RobustDeepAutoencoder.py) <br>
 - **Outlier Detection Model** with l21 regularization on S.T is at:<br>
["l21 Robust Autoencoder"](https://github.com/zc8340311/RobustAutoencoder/blob/master/model/l21RobustDeepAutoencoderOnST.py) <br>
 - **Dataset and demo**: The outlier detection data is sampled from famous MNIST dataset. The .npk file and .txt file are same, but .npk is only load by python2 numpy. Please file more details at demo:<br>
["Demo"](https://github.com/zc8340311/RobustAutoencoder/blob/master/data/Data%20Load%20and%20Show.ipynb) <br>
 - **Repeating Experiments in paper**. Please go to ["Outlier Detection"](https://github.com/zc8340311/RobustAutoencoder/tree/master/experiments/Outlier%20Detection) <br>
This folder also contains an l21 robust autoencoder implementation which need different lambdas with the lambdas used by those under model/ folder. These lambdas are chosend exactly the same as the lambda in our paper. <br>
Please follow these steps: <br>
python experiment1 <br>
python experiment2 <br>
open ipython notebook and check the results. <br>

## Citation
If you find this repo useful and would like to cite it, citing our paper as the following will be really appropriate: <br>

```
@inproceedings{zhou2017anomaly,
  title={Anomaly detection with robust deep autoencoders},
  author={Zhou, Chong and Paffenroth, Randy C},
  booktitle={Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={665--674},
  year={2017},
  organization={ACM}
}
```
## Reference
[1]Abadi, Martín, et al. "TensorFlow: A System for Large-Scale Machine Learning." OSDI. Vol. 16. 2016. <br>
[2]LeCun, Yann, Corinna Cortes, and C. J. Burges. "MNIST handwritten digit database." AT&T Labs [Online]. Available: [MNIST](http://yann.lecun.com/exdb/mnist) (2010).