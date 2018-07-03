# Face Recognition through static images

<p align="center">
<img src="https://4.bp.blogspot.com/-0N9UXFGHrok/WbvQrlXAZNI/AAAAAAAAEQU/q-KnCQpf-Vkj_enWRo5Ayqn3ca5Kcq1DACLcBGAs/s1600/Face-Recognition-Access-Control-System.jpg" alt="neofetch" align="middle" height="200px">
</p>

Everyday humans are exposed to dozens of faces some familiar, some not. Yet with barely a glance, your brain assesses the features on those faces and fits them to the corresponding individual. Research shows that many people recognize faces even if they forget other key details about a person, like their name or their job. But how exactly does this remarkable process work in the brain?

Let's replicate the same through convolutional neural networks that mimic operations in our brain.

## Objective

This project aims to build a convolutional neural network that classifies faces of individuals from static images. 

## Dataset description

Labeled Faces in the Wild, a database of face photographs is designed for studying the problem of unconstrained face recognition. The data set contains more than 13,000 images of faces collected from the web. Each face has been labeled with the name of the person pictured. 

A glimpse into the dataset is provided below:

<img src="http://vis-www.cs.umass.edu/lfw/Six_Face_Panels_sm.jpg" height="200px" >

More information about the data set used in the study can be found at: 
http://vis-www.cs.umass.edu/lfw/

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
  - Dlib
  
 ## Installation of Python and packages

Install Python 3 and all of these packages in a few clicks with the [Anaconda Python distribution](https://www.continuum.io/downloads). 

Anaconda is very popular amongst Data Science and Machine Learning communities.

The packages used in this project are:

1. numpy
2. keras
3. opencv
4. tensorflow
5. matplotlib
6. dlib

Commands for installation of packages in Anaconda are:

1. conda install -c conda-forge keras 
2. conda install -c conda-forge tensorflow
3. conda install -c conda-forge opencv
4. conda install -c conda-forge dlib

## The Analytical Approach

### Data Loading

The LFW dataset has to be downloaded. For this case study we have used images of the following people from the LFW database:

1. Ariel Sharon
2. Arnold Schwarzenegger
3. Colin Powell
4. Donald Rumsfeld
5. George W Bush
6. Gerhard Schroeder
7. Hugo Chavez
8. Jacques Chirac
9. Junichiro Koizumi
10. Tony Blair
11. Vladimir Putin

Images from the internet were downloaded for the following person:

12.Lionel Messi

Images were collected from self:

13.Roshni Rajan

Load Metadata for images stored in file path
```{r load_data}
import numpy as np
import os.path

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

metadata = load_metadata('images')
```
Read images stored in file path
```{r load_data}
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
```
<img src="input_data_sample.PNG" alt="neofetch" align="middle" >


