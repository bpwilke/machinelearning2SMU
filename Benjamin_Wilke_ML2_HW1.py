# adapt this to run

# 1. write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each

# 2. expand to include larger number of classifiers and hyperparmater settings

# 3. find some simple data

# 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings

# 5. Please set up your code to be run and save the results to the directory that its executed from

# 6. Collaborate to get things

# 7. Investigate grid search function

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier # added this to include as another classifier
from statistics import mean # to calculate mean
from statistics import median # to calculate median
from sklearn.model_selection import KFold  # EDIT: I had to import KFold
import itertools as it # need for grid search combinations
import matplotlib.pyplot as plt # import for box plot display

# load iris dataset (fulfills #3 requirement)
iris_dat = np.loadtxt("iris.data", delimiter=",", usecols=(0,1,2,3))
iris_labels = np.loadtxt("iris.data", delimiter=",", usecols=(4), dtype="U25")
M = iris_dat
L = iris_labels

# instantiate data object to pass to the classificationListRunner()
n_folds = 5
data = (M, L, n_folds)

################################################################################################################################################################
# 1. write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each
################################################################################################################################################################

clf_list = [{"model": LogisticRegression, "hypers": {"penalty":"l1", "random_state":69, "n_jobs":1, "solver":"liblinear"}},
            {"model": LogisticRegression, "hypers": {"penalty":"l2", "random_state":6969, "n_jobs":-1, "solver":"liblinear"}}]

def classificationListRunner(clf_list_w_hyper, data):

    M, L, n_folds = data  # unpack the "data" container of arrays
    kf = KFold(n_splits=n_folds)  # Establish the cross validation

    ret = []  # initialize list to store all results (return)

    for clf in clf_list_w_hyper:  # iterate each item in classifiers list
        entry = {} # initialize/clear entry dict for each iteration
        entry_accuracy_scores = [] # initialize empty list to hold each k-fold score to add to our entry

        model = clf['model'](**clf['hypers']) # instantiate the classifier, using unpacked hypers
        entry["model"] = model # add the model to the results entry
        print(model) # log current classifier with current hypers
        for ids, (train_index, test_index) in enumerate(kf.split(M, L)): # iterate over k-fold object for the selected classifier from previous line
            model.fit(M[train_index], L[train_index])
            pred = model.predict(M[test_index])
            print("Fold {0} Accuracy:".format(ids + 1), accuracy_score(L[test_index], pred)) # print the fold accuracy to the screen
            entry_accuracy_scores.append(accuracy_score(L[test_index], pred))

        # print some stuff to the screen
        print("Cross Validation Mean Accuracy:", mean(entry_accuracy_scores)) # display mean accuracy score to log
        print("Cross Validation Median Accuracy:", median(entry_accuracy_scores)) # display median accuracy score to log
        print("#################################################################################")

        # update our entry and add to the return list
        entry["accuracy_scores"] = entry_accuracy_scores # add accuracy scores list to entry
        entry["mean_accuracy"] = mean(entry_accuracy_scores) # add mean accuracy score to entry
        entry["median_accuracy"] = median(entry_accuracy_scores)
        ret.append(entry) # add the entry to our return list

    return ret # return our list of dict results

initialClfs = classificationListRunner(clf_list, data)

################################################################################################################################################################
# 2. expand to include larger number of classifiers and hyperparmater settings
################################################################################################################################################################

clf_list_large = [{"model": LogisticRegression, "hypers": {"penalty":"l1", "random_state":10, "solver":"liblinear", "multi_class":"auto"}},
            {"model": LogisticRegression, "hypers": {"penalty":"l2", "random_state":1000, "multi_class":"auto"}},
            {"model": RandomForestClassifier, "hypers": {"n_estimators":5, "random_state":10, "max_depth":3}},
            {"model": RandomForestClassifier, "hypers": {"n_estimators":25, "random_state":1000, "max_depth":5}},
            {"model": KNeighborsClassifier, "hypers": {"n_neighbors":3, "weights":"uniform"}},
            {"model": KNeighborsClassifier, "hypers": {"n_neighbors":5, "weights":"distance"}}]

expandedClfs = classificationListRunner(clf_list_large, data)

################################################################################################################################################################
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings
################################################################################################################################################################

def generateBoxplots(clfsResults, filename):
    models = [] # initialize empty list to hold our models for plot iris_labels
    all_accuracy_scores = [] # initialize empty list to hold each models accuracy for each k-fold
    # iterate over clfsResults list to prep data for plotting
    for result in clfsResults:
        models.append(result["model"])
        all_accuracy_scores.append(result["accuracy_scores"])

    fig = plt.figure(figsize = (20,20))
    axes = fig.add_subplot()
    axes.boxplot(all_accuracy_scores, vert=False, labels=models)
    plt.xlabel("Accuracy Score")
    plt.xlim(0,1)
    fig.tight_layout()
    fig.savefig(filename)

generateBoxplots(initialClfs, "Wilke_ML2_HW1_InitialBoxplots.png")
generateBoxplots(expandedClfs, "Wilke_ML2_HW1_ExpandedBoxplots.png")

################################################################################################################################################################
# 7. Investigate grid search function
################################################################################################################################################################

clf_grid = {"model": RandomForestClassifier, "hypers" : {"n_estimators": [5,15,25], "random_state":[5], "max_depth": [3,7,10]}} # this will be my "grid search" format

def generateGridSearchList(grid_dict):
    clf_list = []  # initialize a list for my clfs + hypers (in same format as earlier)
    hypers = grid_dict["hypers"]
    keys = hypers.keys() # get all keys under hypers in input dict
    values = (hypers[key] for key in keys) # get all values for all keys
    combinations = [dict(zip(keys, combination)) for combination in it.product(*values)] # create all combos of keys and values

    for combo in combinations:  # create the list of clfs in the same format as used above
        clf_list.append({"model": grid_dict["model"], "hypers": combo})

    return clf_list

gridSearchClfs = classificationListRunner(generateGridSearchList(clf_grid), data)

generateBoxplots(gridSearchClfs, "Wilke_ML2_HW1_DEATHtoGridSearchBoxPlots.png")
