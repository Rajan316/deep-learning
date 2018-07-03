# Face Recognition through static images

<p align="center">
<img src="https://4.bp.blogspot.com/-0N9UXFGHrok/WbvQrlXAZNI/AAAAAAAAEQU/q-KnCQpf-Vkj_enWRo5Ayqn3ca5Kcq1DACLcBGAs/s1600/Face-Recognition-Access-Control-System.jpg" alt="neofetch" align="middle" height="200px">
</p>

Everyday humans are exposed to dozens of faces some familiar, some not. Yet with barely a glance, your brain assesses the features on those faces and fits them to the corresponding individual. Research shows that many people recognize faces even if they forget other key details about a person, like their name or their job. But how exactly does this remarkable process work in the brain?

Let's replicate the same through convolutional neural networks that mimic operations in our brain.

## Objective

This project aims to build a convolutional neural network that classifies faces of individuals from webcam. 

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
<img src="input_data_sample.jpg" alt="neofetch" align="middle" >

### Building the network

This notebook uses a deep convolutional neural network (CNN) to extract features from input images. It follows the approach described in  the OpenFace project. Keras is used for implementing the CNN, Dlib and OpenCV for aligning faces on input images.

The CNN architecture used here is a variant of the inception architecture. More precisely, it is a variant of the NN4 architecture and identified as nn4.small2 model in the OpenFace project. This notebook uses a Keras implementation of that model whose definition was taken from the Keras-OpenFace project. 

There is a fully connected layer with 128 hidden units followed by an L2 normalization layer on top of the convolutional base. These two top layers are referred to as the embedding layer from which the 128-dimensional embedding vectors can be obtained. The complete model is defined in model.py and a graphical overview is given below. A Keras version of the nn4.small2 model can be created with create_model().

```{}
from model import create_model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
```
<img src="model.png" width="800"/>

The OpenFace project provides pre-trained models that were trained with the public face recognition datasets FaceScrub and CASIA-WebFace. The Keras-OpenFace project converted the weights of the pre-trained nn4.small2.v1 model to CSV files which were then converted here to a binary format that can be loaded by Keras with load_weights:
```{}
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
```
### Face Alignment

The nn4.small2.v1 model was trained with aligned face images, therefore, the face images from the custom dataset must be aligned too. Here, we use Dlib for face detection and OpenCV for image transformation and cropping to produce aligned 96x96 RGB face images. We use the AlignDlib utility to achieve the same.

```{}
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib

%matplotlib inline

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/shape_predictor_68_face_landmarks.dat')

# Load an image of Jacques Chirac
jc_orig = load_image(metadata[65].image_path())

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
plt.subplot(131)
plt.imshow(jc_orig)

# Show original image with bounding box
plt.subplot(132)
plt.imshow(jc_orig)
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

# Show aligned image
plt.subplot(133)
plt.imshow(jc_aligned);
```
The detection of face from an image is illustrated below along with corresponding alignment.
<img src="recog_image_detect.PNG" />

### Embedding vectors
Embedding vectors can now be calculated by feeding the aligned and scaled images into the pre-trained network.

```{}
embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
 
```
<img src="distance_between.PNG" />

With embedding we can find out the distance between two images and then decide how same or how different one is from the other.
As expected, the distance between the two images of Arial Sharon is smaller than the distance between an image of Arial Sharon and an image of Messi (0.16 < 1.46). 
But we still do not know what distance threshold $\tau$ is the best boundary for making a decision between same identity and different identity.

For finding optimal distance threshold we employ F1 score to understand at which threshold the accuracy is the best. We find that at a threshold of 0.65 the accuracy is 95.9% but since nn4.small2.v1 is a relatively small model it is still less than what can be achieved by state-of-the-art models (> 99%).

<img src="accracy_threshold.PNG" />

### Face recognition 

Using Support vector machines(SVM) classifier we can train the embeddings and then test it on unknown embedded data and understand whether the face has been recognized correctly.

```{}
import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

example_idx = 38

example_image = load_image(metadata[test_idx][example_idx].image_path())
example_prediction = svc.predict([embedded[test_idx][example_idx]])
example_identity = encoder.inverse_transform(example_prediction)[0]

plt.imshow(example_image)
plt.title(f'Recognized as {example_identity}');
```

<img src="recognized_as.PNG" />

#### Visualization of embeddings

To embed the dataset into 2D space for displaying identity clusters, t-distributed Stochastic Neighbor Embedding (t-SNE) is applied to the 128-dimensional embedding vectors. Except from a few outliers, identity clusters are well separated.

<img src="./face_recog_images/cluster_embeddings.PNG" />

#### To detect faces through webcam

We have to use the opencv videocapture function to capture webcam video feed. The model has been used to predict on single faces as well as multiple faces.

```{}
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)



# Initialize some variables
bb=[]
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = small_frame[...,::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        #face_locations = face_recognition.face_locations(small_frame)
        #face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        # Detect face and return bounding box
        bb = alignment.getLargestFaceBoundingBox(small_frame)
        
        # Transform image using specified face landmark indices and crop image to 96x96
        #jc_aligned = alignment.align(96, small_frame, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        if bb!=None:
            print(bb.left(), bb.top(), bb.width(), bb.height())
            # Draw a box around the face
            
            big_frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
            big_frame = big_frame[...,::-1]
            bb1 = alignment.getLargestFaceBoundingBox(big_frame)
            jc_aligned = alignment.align(96, big_frame, bb1, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            # scale RGB values to interval [0,1]
            img = (jc_aligned / 255.).astype(np.float32)
            # obtain embedding vector for image
            embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            example_prediction = svc.predict([embedded])
            example_identity = encoder.inverse_transform(example_prediction)[0]
            print(example_identity)
            #
            cv2.rectangle(frame, (bb.left()*4, bb.top()*4), ((bb.left()+bb.width())*4, (bb.top()+bb.height())*4), (0, 0, 255), 2)
            #cv2.rectangle(frame, (bb1.left(), (bb1.top()+bb1.height()) - 35), ((bb1.left()+bb1.width()), ((bb1.top()+bb1.height())*4), (0, 0, 255), cv2.FILLED))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, example_identity, ((bb1.left()) + 6, ((bb1.top()+bb1.height())) - 6), font, 0.6, (255, 255, 255), 1)
        
    process_this_frame = not process_this_frame

   
    
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
```
#### Images of prediction

<img src="Video_screenshot_03.07.2018_1.PNG" width="525"/> 


##### Click [here](./Face_recog_webcam.ipynb) to go to the notebook where the entire case study steps has been performed.



