import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV

# Read data
train_X = pd.read_csv('./data/train_x.csv')
train_y = pd.read_csv("./data/raw/train_y.csv")
test_X = pd.read_csv("./data/test_x.csv")

# drop id col in target (id in feature dataframes will get dropped during preprocessing
train_y = train_y.iloc[:, 1]

# Which cols are numeric and categorical
categorical_cols = train_X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = train_X.select_dtypes(include=['number']).columns.tolist()

# Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create preprocessor for column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)], remainder='drop')

# Storage for fitted models
models_clf = {}
# Range of l1_ratios to explore for elastic net
l1_ratios = list(np.linspace(0, 1, 5))

for penalty in ['l1', 'l2', 'elasticnet']:

    # Append classifier to preprocessing pipeline.
    lr_clf = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', LogisticRegressionCV(penalty=penalty, Cs=[0.1, 1, 10], l1_ratios=l1_ratios,
                                                                 solver='saga', max_iter=200))])
    # Fit the model
    lr_clf.fit(train_X, train_y)
    print(f"Best model score for {penalty} : {lr_clf['classifier'].scores_}")

    # Store fitted model
    models_clf[penalty] = deepcopy(lr_clf)

# Create a pickle object containing all the fitted models

with open("output/models_clf.pkl", "wb") as model_file:
    pickle.dump(models_clf, model_file)