import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
from sklearn import neighbors # imported libraries

class supervised_k_nearest_neighbors(): # class call
    def __init__(self, df_name, k, y_feat, x_feat_list = []): # init function call with necessary variables
        """
        Constructor for the class. Initializes the instance variables of the class.

        Parameters:
            df_name (str): The name of the dataframe that the instance will use.
            k (int): The number of folds for cross-validation.
            y_feat (str): The name of the dependent variable in the dataframe.
            x_feat_list (List[str]): The list of independent variables in the dataframe. Defaults to an empty list.

        Returns: 
            None
        """
        self.df_name = df_name # define variables
        self.k = k # define variables
        self.x_feat_list = x_feat_list # define variables
        self.y_feat = y_feat # define variables
    
    def create_dataframe(self): # function call for create_dataframe 
        """
        Creates a pandas DataFrame object from a seaborn dataset.

        Parameters:
            None
    
        Returns:
            dataframe: A pandas DataFrame object that has been loaded from a seaborn dataset and has any rows with missing values dropped.
        """
        dataframe = sns.load_dataset(self.df_name) # loads dataframe
        dataframe.dropna(axis = 0, inplace = True) # drops bad data such as Na
        return dataframe # returns new dataframe with cleaned data
    
    def k_classifer(self, dataframe): # function call for 
        """
        Performs a k-classification on the given dataframe.

        Parameters:
            dataframe (dataframe): A pandas dataframe containing the features to be used for classification.

        Returns:
            y_true (array): True Labels
            y_pred (array): Predicted Labels
        """
        x = dataframe.loc[:, self.x_feat_list].values # independent variables for the dataframe
        y_true = dataframe.loc[:, self.y_feat].values # dependent variables for the dataframe

        knn_classifer = KNeighborsClassifier(n_neighbors = self.k) # calls KneighborsClassifier function from Sk learn
        knn_classifer.fit(x, y_true) # fits the classifer

        y_pred = knn_classifer.predict(x) # makes the prediction
        return y_true, y_pred # returns the true dependent variable as well as the predicted value

    def show_confusion_matrix(self, y_true, y_pred): # function call for the confusion matrix
        """
        Displays a confusion matrix for the given true and predicted labels.

        Parameters:
            y_true (array): True labels
            y_pred (array): Predicted labels

        Returns:
            None
        """
        conf_mat = confusion_matrix(y_true = y_true, y_pred = y_pred)
        conf_mat_disp = ConfusionMatrixDisplay(conf_mat, display_labels = np.unique(y_true))
        conf_mat_disp.plot() # plots the confusion matrix

        plt.gcf().set_size_inches(8, 8) # size of the confusion matrix

        plt.grid(False) # grid lines
        
    def inputs_from_user(self): # function call for user inputs
        """
        This method prompts the user to enter numerical values for each feature in the model's `x_feat_list` attribute. The user inputs are stored in a dictionary, where each key represents a feature name, and each value is the corresponding numerical value input by the user.

        Parameters:
            None

        Returns:
            A dictionary containing user inputs for each feature in `self.x_feat_list`, where each key is a feature name, and each value is a numerical value input by the user. The dictionary also includes a default entry for the model's `y_feat` attribute, with a value of 'Unknown'.
        """
        list_of_user_input = {self.y_feat: 'Unknown'} # user inputs
        for item in self.x_feat_list:
            value_from_user = float(input(f'please enter the {item}')) # prompts user to enter a value
            list_of_user_input[item] = value_from_user # stores value in a list
        return list_of_user_input # returns list of inputs

    def prediction(self, dataframe, user_inputs): # prediction function call
        """
        Perform classification prediction using K-Nearest Neighbors algorithm.

        Parameters:
            dataframe (pandas.DataFrame): The input data frame that contains the training data.
            user_inputs (dict): A dictionary that contains user-provided inputs for the features.

        Returns:
            dataClass (int or str): The predicted class of the input data based on the trained K-Nearest Neighbors model.

        """
        x = dataframe.loc[:, self.x_feat_list].values 
        y_true = dataframe.loc[:, self.y_feat].values # prediction values


        clf = neighbors.KNeighborsClassifier(self.k, weights='distance') # calls on KNeighbor classifier
        clf.fit(x, y_true)

        list_of_data = [] # creates list of data
        for feat in self.x_feat_list:
            list_of_data.append(user_inputs[feat]) # appends user inputs to the list of data
    
        dataClass = clf.predict([list_of_data]) # predicts using the list of data
        return dataClass # returns prrediction
    
    def scatterplot(self, dataframe,user_input): # scatterplot function call
        """
        This function creates a scatterplot using the Plotly library, based on the user's input and a given dataframe.

        Parameters:
            dataframe: The pandas DataFrame containing the data to be plotted.
            user_input: A dictionary containing the user's input data.
        Returns:
            None
        """
        needed_columns = self.x_feat_list.append(self.y_feat)
        df_want = pd.DataFrame(dataframe, columns= needed_columns)
        feat0 = self.x_feat_list[0] # age
        feat1 = self.x_feat_list[1] # fare
        items = user_input.items()
        user_inputs_df = pd.DataFrame({'keys': [i[0] for i in items], 'values': [i[1] for i in items]})
        sns.set(font_scale=1.2)
        
        dataframe_added_element = pd.concat([df_want, user_inputs_df], ignore_index=True)

        fig = px.scatter(data_frame = dataframe_added_element, x=feat0, y=feat1, color='alive')
        fig.show() # creates and shows scatterplot
        
    def main(self): # main function call
        """
        Runs the main program and displays a dataframe

        Parameters:
            None

        Returns:
            None
        """
        df = self.create_dataframe()
        y_true, y_pred = self.k_classifer(df)
        self.show_confusion_matrix(y_true, y_pred)
        user_inputs = self.inputs_from_user()
        predicted_value = self.prediction(df,user_inputs)
        print(f'Based on the information above the predicted {self.y_feat} is {predicted_value}')
        self.scatterplot(df, user_inputs) # calls all functions mentioned before

if __name__ == '__main__':
    data = supervised_k_nearest_neighbors('titanic', 3, 'alive', ['age', 'fare']) # class call
    data.main() # runs main function
