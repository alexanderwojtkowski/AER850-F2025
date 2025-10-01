""" AER850: Project 1 """
# Name: Alexander Wojtkowski
# Student #: 501168859

# Due Date: October 6th, 2025

""" Imports """

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
import joblib

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

""" Part 1: Data Processing - 2 marks """
# Read data from a csv file and convert that into a dataframe, which will allow
# for all further analysis and data manipulation.

data = pd.read_csv("Project 1 Data.csv")
X = data[["X", "Y", "Z"]]
y = data["Step"]

""" Part 2: Data Visualization - 8 marks """
# Perform statistical analysis on the dataset and visualize the dataset 
# behaviour within each class. This will provide an initial understanding of 
# the raw data behaviour. You are required to include the plots and explain the
# findings.

data.hist(figsize=(10,6))
plt.show()

""" Part 3: Correlation Analysis - 15 marks """
# Assess the correlation of the features with the target variable. A 
# correlation study provides an understanding of how the features impact the 
# target variable. A common correlation method used is Pearson Correlation. 
# You are required to include the correlation plot, and explain the correlation
# between the features and the target variables, along with the impact it could
# have on your predictions.

corr_matrix = data.corr()
plt.figure()
sns.heatmap(np.abs(corr_matrix), annot=True)
plt.title("Correlation Matrix")
plt.show()

""" Part 4: Classification Model Development/Engineering - 20 marks """
# Prepare the data to create three classification models (based on ML 
# algorithms). The dataset needs to be split into training and testing 
# categories to develop the models. For each ML model, utilize grid search 
# cross-validation to assess the hyperparameters that give you the best 
# results. You are required to explain your selected choice of classification 
# algorithms. In addition to the three classification models with grid search 
# cross-validation, you must make one model based on using RandomizedSearchCV. 
# this will provide another method of determining the best hyperparameters to
# optimize your results.

# Splitting the data in 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=42
                                                    )

# Some algorithms benefit from scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Specifying the models and their parameters
LogReg     = LogisticRegression(max_iter=1000, class_weight="balanced")
svc        = SVC(class_weight="balanced")
RandForest = RandomForestClassifier(class_weight="balanced", random_state=42)

params = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "saga", "newton-cholesky"]},
    "SVC": {"C": [0.01, 0.1, 1, 10], "kernel": ["linear", "rbf", "poly"], "gamma": [0.01, 0.1, 1, 10]},
    "Random Forest": {"max_depth": [None, 5, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "n_estimators": [50, 100, 200, 400]}
}

# Running GridSearchCV on each model to obtain best parameters
models = {"Logistic Regression": LogReg, "SVC": svc, "Random Forest": RandForest}
best_models = {}
reports = {}

for name, model in models.items():
    grid = GridSearchCV(model, params[name], cv=5, scoring="f1_weighted")
    grid.fit(X_train_scaled, y_train)
    
    y_pred = grid.predict(X_test_scaled)
    best_models[name] = grid.best_estimator_
    
    reports[name] = classification_report(y_test, y_pred)
    
# Using RandomizedSearchCV on the Random Forest model to determine alternative 
# parameters
randforest_random = RandomizedSearchCV(
    estimator=RandForest,
    param_distributions=params["Random Forest"],
    n_iter=30,
    cv=5,
    scoring="f1_weighted",
    random_state=42,
    n_jobs=-1
)    

randforest_random.fit(X_train, y_train)
y_pred = randforest_random.predict(X_test)

best_models["Random Forest_Random"] = randforest_random.best_estimator_
reports["Random Forest_Random"] = classification_report(y_test, y_pred)

""" Part 5: Model Performance Analysis - 20 marks """
# Compare the overall performance of each model based on f1 score, precision 
# and accuracy. You are required to provide an explanation to what these 
# metrics mean and which metric to prioritize for this use-case. Based on the
# selected model, create a confusion matrix to visualize the performance of
# your model. Include the confusion matrix in the report as well

# Printing out the classification reports for each model created
for name, rep in reports.items():
    print(f"\n== {name} ==")
    print(best_models[name])
    print(rep)

# Creating a confusion matrix for the SVC model
conf_model = best_models["SVC"]
y_pred = conf_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=conf_model.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - SVC")
plt.show()

""" Part 6: Stacked Model Performance Analysis - 5 marks """
# Using scikit-learn’s StackingClassifier, combine two of the previously
# trained models to analyze the impact of model stacking on overall
# performance. Evaluate the performance of the stacked model based on their f1
# score, precision and accuracy along with a confusion matrix to provide a
# clear visual representation. Include this confusion matrix in the report for
# detailed performance analysis. If a significant increase in accuracy is
# observed, discuss how combining complementary strengths of the models
# contributed to the improvement. Conversely, if the change is minimal, explain
# why you think the stacking models had limited effectiveness.

# Creating the Stacked Model
estimators = [
    ('Random Forest', best_models["Random Forest"]),
    ('SVC', best_models["SVC"]),
    ]

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
    cv=5,
    n_jobs=-1
    )

# Training and running the model
stacked_model.fit(X_train_scaled, y_train)
y_pred = stacked_model.predict(X_test_scaled)

print("\n== Stacked Model (Random Forest + SVC) ==")
print(classification_report(y_test, y_pred))

# Making the confusion matrix and displaying it
conf_model = stacked_model
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=conf_model.classes_)
disp.plot(cmap="Reds", xticks_rotation=45)
plt.title("Confusion Matrix - Stacked Model")
plt.show()

""" Part 7: Model Evaluation - 10 marks """
# Package the selected model and save it in a joblib format, this allows you to
# call the model to predict the class based on random set of coordinates given.

#Based on the data set provide, you are required to predict the corresponding
# maintenance step:
# [9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]

test_points = pd.DataFrame([[9.375,3.0625,1.51],
                            [6.995,5.125,0.3875],
                            [0,3.0625,1.93],
                            [9.4,3,1.8],
                            [9.4,3,1.3]],
                           columns = ["X","Y","Z"]
                           )

# Creating Model Pipeline
svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("SVC", best_models["SVC"])
    ])
svc_pipeline.fit(X_train, y_train)

# Creating Joblib files
joblib.dump(svc_pipeline, "best_model.joblib")
loaded_model = joblib.load("best_model.joblib")

# Predicting the values
prediction = loaded_model.predict(test_points)

print("\nPredictions for test points:", prediction)
