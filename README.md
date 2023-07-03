# Titanic-Survival-Rate-Predictor

This repository contains code for performing supervised k-nearest neighbors classification. The code uses the Titanic dataset to predict the survival status of passengers based on their age and fare.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Class: `supervised_k_nearest_neighbors`](#class-supervised_k_nearest_neighbors)
  - [Constructor](#constructor)
  - [Method: `create_dataframe`](#method-create_dataframe)
  - [Method: `k_classifer`](#method-k_classifer)
  - [Method: `show_confusion_matrix`](#method-show_confusion_matrix)
  - [Method: `inputs_from_user`](#method-inputs_from_user)
  - [Method: `prediction`](#method-prediction)
  - [Method: `scatterplot`](#method-scatterplot)
  - [Method: `main`](#method-main)

## Introduction
This code implements a k-nearest neighbors classification algorithm using the Titanic dataset. It provides functions to create a dataframe from the dataset, perform k-classification, display a confusion matrix, prompt the user for inputs, make predictions, and generate a scatterplot based on user inputs.

## Dependencies
The code relies on the following dependencies:
- `numpy` (version >= 1.21.0)
- `pandas` (version >= 1.3.0)
- `seaborn` (version >= 0.11.1)
- `matplotlib` (version >= 3.4.3)
- `scikit-learn` (version >= 0.24.2)
- `plotly` (version >= 5.1.0)

You can install the required dependencies using pip:

```
pip install numpy pandas seaborn matplotlib scikit-learn plotly
```

## Usage
To use this code, follow these steps:

1. Clone the repository or download the source code files.
2. Make sure you have installed all the necessary dependencies.
3. Open a Python environment or Jupyter Notebook.
4. Import the required libraries:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
from sklearn import neighbors
```

5. Copy and paste the code into your Python environment or Jupyter Notebook.
6. Instantiate the `supervised_k_nearest_neighbors` class with the desired parameters:

```python
data = supervised_k_nearest_neighbors('titanic', 3, 'alive', ['age', 'fare'])
```

7. Call the `main` method to execute the code:

```python
data.main()
```

## Class: `supervised_k_nearest_neighbors`

### Constructor

```python
def __init__(self, df_name, k, y_feat, x_feat_list = [])
```

The constructor initializes the instance variables of the class.

**Parameters:**
- `df_name` (str): The name of the dataframe that the instance will use.
- `k` (int): The number of folds for cross-validation.
- `y_feat` (str): The name of the dependent variable in the dataframe.
- `x_feat_list` (List[str]): The list of independent variables in the dataframe. Defaults to an empty list.

### Method: `create_dataframe`

```python
def create_dataframe(self)
```

This method creates a pandas DataFrame object from a seaborn dataset.

**Parameters:**
None

**Returns:**
- `dataframe` (DataFrame): A pandas DataFrame

 object that has been loaded from a seaborn dataset and has any rows with missing values dropped.

### Method: `k_classifer`

```python
def k_classifer(self, dataframe)
```

This method performs k-classification on the given dataframe.

**Parameters:**
- `dataframe` (DataFrame): A pandas DataFrame containing the features to be used for classification.

**Returns:**
- `y_true` (array): True labels
- `y_pred` (array): Predicted labels

### Method: `show_confusion_matrix`

```python
def show_confusion_matrix(self, y_true, y_pred)
```

This method displays a confusion matrix for the given true and predicted labels.

**Parameters:**
- `y_true` (array): True labels
- `y_pred` (array): Predicted labels

**Returns:**
None

### Method: `inputs_from_user`

```python
def inputs_from_user(self)
```

This method prompts the user to enter numerical values for each feature in the model's `x_feat_list` attribute. The user inputs are stored in a dictionary, where each key represents a feature name, and each value is the corresponding numerical value input by the user.

**Parameters:**
None

**Returns:**
- `list_of_user_input` (dict): A dictionary containing user inputs for each feature in `self.x_feat_list`, where each key is a feature name, and each value is a numerical value input by the user. The dictionary also includes a default entry for the model's `y_feat` attribute, with a value of 'Unknown'.

### Method: `prediction`

```python
def prediction(self, dataframe, user_inputs)
```

This method performs classification prediction using the K-Nearest Neighbors algorithm.

**Parameters:**
- `dataframe` (DataFrame): The input dataframe that contains the training data.
- `user_inputs` (dict): A dictionary that contains user-provided inputs for the features.

**Returns:**
- `dataClass` (int or str): The predicted class of the input data based on the trained K-Nearest Neighbors model.

### Method: `scatterplot`

```python
def scatterplot(self, dataframe, user_input)
```

This method creates a scatterplot using the Plotly library based on the user's input and a given dataframe.

**Parameters:**
- `dataframe` (DataFrame): The pandas DataFrame containing the data to be plotted.
- `user_input` (dict): A dictionary containing the user's input data.

**Returns:**
None

### Method: `main`

```python
def main(self)
```

This method runs the main program, displays a dataframe, and executes all the necessary functions.

**Parameters:**
None

**Returns:**
None

---
**Note:** This README file provides an overview of the code structure and functionality. For detailed usage and implementation instructions, refer to the code comments and docstrings within the source code itself.
