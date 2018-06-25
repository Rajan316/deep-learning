# Digit Recognition through static images
<p align="center">
<img src="https://cdn-images-1.medium.com/max/718/1*4TJWlK-FPhskEIJshfEx5g.jpeg" alt="neofetch" align="middle" height="200px">
</p>
Recognizing  <br />


## Objective


## Dataset description

The Keras framework comes already with a MNIST Dataset that can be downloaded. It contains 60.000 images of handwritten images that can be used to train a neural network.

A glimpse into the dataset is provided below:

<img src="https://cdn-images-1.medium.com/max/800/1*UKXLbb40YjviJWerqz3D2Q.png" alt="neofetch" >


    


More information about the data set used in the study can be found at: 

## Tools required

- [Python](https://www.python.org/). Python 3 is the best option.
- [IPython and the Jupyter Notebook](http://ipython.org/). (FKA IPython and IPython Notebook.)
- Some scientific computing packages:
	- numpy
	- pandas
	- scikit-learn
	- matplotlib
  
## Installation of Python and packages

Install Python 3 and all of these packages in a few clicks with the [Anaconda Python distribution](https://www.continuum.io/downloads). 

Anaconda is very popular amongst Data Science and Machine Learning communities.

The scientific computing packeges used in this project are:

1. numpy
2. pandas
3. scikit-learn
4. seaborn
5. matplotlib

## The Analytical Approach

### Data Loading

Read the dataset framingham
```{r load_data}
framingham = pd.read_csv('framingham.csv')
# print the first 5 rows of data
framingham.head()
```
<img src="images/profile.PNG" alt="neofetch" align="middle" >


### Data Cleaning

#### Steps:

* Count the number of null values present in each column.
<img src="images/null_values.PNG" alt="neofetch" align="middle" >

* All columns are having less than 10% of missing values and thus they can be filled in with appropriate values using the following user defined function:

```{python impute function}
def impute_median(series):
    return series.fillna(series.median())
```
* Few of the columns like BMI and cigsPerDay can be filled in using groupby by comparing values in columns directly related to them.

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

This helps in making a more informed guess about the missing values.

Also, the correlation plot helps decide which columns need to be dropped from the dataset using the correlation plot. The column "education" has very less correlation with other columns of the dataset and thus can be dropped in our future analysis.




#### Compare confusion matrix between imbalanced vs balanced data

Our aim is to be able to detect the risk of heart disease more precisely even if we have to compromise on the false positives. It is very crucial to reduce the false negatives as a person's health is jeopardised once he is wrongly identified as healthy when in actual he is sick. 

<img src="images/conf_imbalanced.PNG" width="425"/> <img src="images/conf_balanced.PNG" width="450"/> 

#### Compare between ROC curve of predicted probabilities of imbalanced vs balanced data

<img src="images/roc_curve_imbalanced.PNG" width="425"/> <img src="images/roc_curve_balanced.PNG" width="425"/> 
<p align="left">
<img src="images/roc_curve_area_imbalanced.PNG" width="350" hspace="50"/> <img src="images/roc_curve_area_balanced.PNG" width="350"/> 
</p>
 

##### Click [here](./Framingham.ipynb) to go to the notebook where the entire case study steps has been performed.


## Future Scope

The Framingham case study provides a foundation for research into clinical case studies related to occurence of heart disease which claims millions of lives every year. Machine learning models helps doctors identify patients who have chances of contracting one. This technique can be replicated for prevention of various other diseases thus saving a lot of lives. 



