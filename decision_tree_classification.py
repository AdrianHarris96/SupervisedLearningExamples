#! /usr/bin/env python3

#Example input: ./decision_tree_classification.py -i breast_cancer.csv -t "Breast Cancer" -T diagnosis -o ~/Desktop

#Loading my packages 
import pandas as pd 
import numpy as np 
import argparse as ap 
import os 
import time 

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

#Accept arguments 
parser = ap.ArgumentParser(prog = "Classification via Decision Tree", description="Explore classification via decision tree. Create plots showing the relationship between accuracy and training size or accuracy and hyperparameters")
parser.add_argument("-i", "--input_csv", help="path to input CSV", required=True)
parser.add_argument("-t", "--title", help="title for figures", default="Input Data")
parser.add_argument("-T", "--target_col", help="name of the target column in the input CSV", required=True)
parser.add_argument("-o", "--output_dir", help="output directory for storing figures", required=True)
args = parser.parse_args()

def decision_tree_learning_curve(df, target, title_prefix):
    print("Generating decision tree learning curves ...")
    #Split the data into features and target
    X = df.drop(target, axis=1) #New dataframe without the target column
    y = df[target] #Extracting the target column

    #Split into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=76)

    #Initialize the Decision Tree Classifier
    """
    Can split using entropy or gini
    Entropy entails that the split is chosen based on the largest reduction of entropy (highest information gain)
    Gini entails that the split is chosen based on the maximum gini gain 
        """
    #Can consider other splitting criterion like gini 
    clf = DecisionTreeClassifier(criterion="entropy", random_state=76)

    #Fit the model to the training data
    clf.fit(X_train, y_train)

    #Compute the learning curves with varying training set sizes
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy", n_jobs=-1, random_state=76)

    #Calculate the mean and standard deviation of the training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    #Constructing plot title 
    learning_plot_title = "{0} Decision Tree - Learning Curve".format(title_prefix)

    #Constructing output name 
    outputname = learning_plot_title.replace(" - ", "")
    outputname = outputname.replace(" ", "_").lower()
    outputname += ".png"

    #Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.title(learning_plot_title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Testing score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, outputname))
    #plt.show()

    return "Done"

if __name__ == "__main__":
    #Load the CSV data into dataframe
    input_df = pd.read_csv(args.input_csv)

    #Run classification via the functions
    start_time = time.time()
    decision_tree_learning_curve(input_df, args.target_col, args.title)
    end_time = time.time()
    print(f"Learining time: {end_time - start_time} seconds")