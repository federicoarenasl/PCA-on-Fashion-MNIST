# PCA on Fashion MNIST Tutorial
The full tutorial can be found in this link: https://federicoarenasl.github.io/PCA-on-Fashion-MNIST/. There is also a ```PCA-FMNIST-Tutorial.md``` file in this repository for those who prefer reading it from here directly.
# Fashion MNIST PCA Tutorial
In this notebook we will explore the impact of implementing Principal Component Anlysis to an image dataset. For this, we will use the benchmark Fashion MNIST dataset, the link to this dataset can be found [here](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion). Indeed, the images from the dataset are 784-dimensional images. In this small tutorial we seek to explore if we can further compress the dimension of these images without loosing valuable information.

## Fashion MNIST dataset
As described [here](https://github.com/zalandoresearch/fashion-mnist), the dataset contains 60k training examples, and 10k testing examples. Each training example is accomanied with a respective label, which can either be:
- 0 	T-shirt/top
- 1 	Trouser
- 2 	Pullover
- 3 	Dress
- 4 	Coat
- 5 	Sandal
- 6 	Shirt
- 7 	Sneaker
- 8 	Bag
- 9 	Ankle boot


## Final results
After a thorough analysis on how to choose the right number of Principal Components to reduce the dimension of our data with, we look at its impact with different numbers of PCs.

    
![png](PCA-FMNIST-Tutorial_files/PCA-FMNIST-Tutorial_28_0.png)
    
