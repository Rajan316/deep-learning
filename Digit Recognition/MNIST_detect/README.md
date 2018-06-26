# Digit Recognition through static images
<p align="center">
<img src="https://cdn-images-1.medium.com/max/718/1*4TJWlK-FPhskEIJshfEx5g.jpeg" alt="neofetch" align="middle" height="200px">
</p>
Numbers are everywhere! Humans can read and recognize digits within few milliseconds. Let us teach computers to do the same through convolutional neural networks.   <br />


## Objective

This project aims to build a convolutional neural network that classifies images of digits from 0-9 . 

## Dataset description

The National Institute for Standards and Technology (NIST) has amassed a large database of digits which is freely available, and has become somewhat of a benchmark for testing classification algorithms. The MNIST database is a modified set compiled from several NIST databases. It consists of 60,000 training examples and 10,000 test samples, each of which is a 28 x 28 pixel greyscale image.

The Keras framework comes already with a MNIST Dataset that can be downloaded. It contains 60.000 images of handwritten images that can be used to train a neural network.

A glimpse into the dataset is provided below:

<img src="https://dmtyylqvwgyxw.cloudfront.net/instances/132/uploads/images/photo/image/33904/large_29158d37-4da1-490b-80d2-c98e8c9287b4.jpg" alt="neofetch" height="200px" >

More information about the data set used in the study can be found at: 
http://yann.lecun.com/exdb/mnist/
## Tools required

- [Python](https://www.python.org/). Python 3 is the best option.
- [IPython and the Jupyter Notebook](http://ipython.org/). (FKA IPython and IPython Notebook.)
- Some scientific computing packages:
	- numpy
	- pandas
	- scikit-learn
	- matplotlib
- Deep learning packages
	- keras
	- tensorflow
	- theano
- Computer vision packages
	- OpenCV
  
## Installation of Python and packages

Install Python 3 and all of these packages in a few clicks with the [Anaconda Python distribution](https://www.continuum.io/downloads). 

Anaconda is very popular amongst Data Science and Machine Learning communities.

The packages used in this project are:

1. numpy
2. keras
3. opencv
4. tensorflow
5. matplotlib

Commands for installation of packages in Anaconda are:

1. conda install -c conda-forge keras 
2. conda install -c conda-forge tensorflow
3. conda install -c conda-forge opencv

## The Analytical Approach

### Data Loading

The MNIST dataset is available by default in the keras package.

Read the dataset MNIST
```{r load_data}
from keras.datasets import mnist
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
<img src="input_data_sample.PNG" alt="neofetch" align="middle" >


### Preparing the dataset

#### Steps:

* Reshape data

```{python impute function}
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
```

* Normalize data

```{python impute function}
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
```
* Convert number of classes to categorical

```{python impute function}
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
```

This can be illustrated using a correlation plot. From this plot we can understand how some of the columns are correlated to each other thus using them in filling the missing values.

<p align="center">
<img src="images/corrplot.PNG" alt="neofetch" align="middle" >
</p>

```{python impute function}
by_currentSmoker_class=framingham.groupby(['currentSmoker'])
framingham.cigsPerDay=by_currentSmoker_class['cigsPerDay'].transform(impute_median)
by_age_class=framingham.groupby(['age','male','diabetes'])
framingham.BMI=by_age_class['BMI'].transform(impute_median)
```





#### Compare confusion matrix between imbalanced vs balanced data



<img src="images/conf_imbalanced.PNG" width="425"/> <img src="images/conf_balanced.PNG" width="450"/> 

#### Compare between ROC curve of predicted probabilities of imbalanced vs balanced data

<img src="images/roc_curve_imbalanced.PNG" width="425"/> <img src="images/roc_curve_balanced.PNG" width="425"/> 
<p align="left">
<img src="images/roc_curve_area_imbalanced.PNG" width="350" hspace="50"/> <img src="images/roc_curve_area_balanced.PNG" width="350"/> 
</p>
 

##### Click [here](./Framingham.ipynb) to go to the notebook where the entire case study steps has been performed.


## Future Scope





