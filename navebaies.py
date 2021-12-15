from numpy.core.fromnumeric import ravel
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import random as rd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split as tts 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix as CXM
from sklearn.cluster import KMeans as KM
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz as egx
from sklearn.naive_bayes import GaussianNB 
from six import StringIO 
from IPython.display import Image
import pydotplus


data_file = pd.read_csv("c120/diabetes.csv")
# print(data_file.head())
X = data_file[["glucose","bloodpressure"]]
Y = data_file["diabetes"]

x_trainnb ,x_testnb ,y_trainnb ,y_testnb = tts(X,Y ,test_size=0.25 , random_state=1)

sc = StandardScaler()
x_trainnb = sc.fit_transform(x_trainnb)
x_testnb = sc.fit_transform(x_testnb)

GNB = GaussianNB()
GNB = GNB.fit(x_trainnb,y_trainnb)
y_predictnb = GNB.predict(x_testnb)


acc_nb = AS(y_testnb,y_predictnb)
print(acc_nb)
# ---------------------------------------------
# logic reg 

x_trainlr,x_testlr ,y_trainlr ,y_testlr = tts(X,Y ,test_size=0.25 , random_state=1)

sc = StandardScaler()
x_trainlr = sc.fit_transform(x_trainlr)
x_testlr = sc.fit_transform(x_testlr)

lr = LogisticRegression(random_state = 0)
lr = lr.fit(x_trainlr,y_trainlr)
y_predictlr = lr.predict(x_testlr)

acc_lr = AS(y_testlr,y_predictlr)
print(acc_lr)

