import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib


dataframe = pd.read_csv("fake_data.csv")
X = dataframe.drop(columns=["score"])
y = dataframe["score"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

my_model = DecisionTreeClassifier()
my_model.fit(X_train, y_train)
# joblib.dump(my_model, "my_score_prediction_model.joblib")
# my_model = joblib.load("my_score_prediction_model.joblib")

tree.export_graphviz(my_model,
                     out_file="my_model_graph.dot",
                     feature_names=["age", "gender"],
                     class_names=[str(i) for i in y.unique()],
                     label="all",
                     rounded=True,
                     filled=True,
                     )
predictions = my_model.predict(X_test)
acc_score = accuracy_score(y_test, predictions)

print(acc_score)

