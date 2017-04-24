## Unsupervised Image to Image Translation with Generative Adversarial Networks

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/results.png" width="70%" height="70%"/>
</div>
</a>

##### Paper: [Unsupervised Image to Image Translation with Generative Adversarial Networks](https://arxiv.org/abs/1701.02676)

### Dataset
* Before training the network, please prepare the data
* CelebA [download](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* SVHN [download]()
* MNIST [download](https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz), and put to `data/mnist_png`

### Usage

#### Step 1: Learning shared feature
```
python train.py --train_step="ac_gan" --retrain=1
```
#### Step 2: Learning image encoder
```
python train.py --train_step="imageEncoder" --retrain=1
```
#### Step 3: Translation
```
python translate_image.py
```
* Samples of all steps will be saved to data/samples/

### Network
<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/network.png" width="70%" height="70%"/>
</div>
</a>

#### What to run different datasets?
* in `train.py` and `translate_image.py` modify the name of dataset `flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, obama_hillary]")`
* write your own `data_loader` in `data_loader.py`


