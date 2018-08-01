import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#load train and test dataset
train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')

#join train and test datasets before categorical conversion
testID = test["PassengerId"]
trainLength = len(train)
dataset = pd.concat(objs=[train, test], axis = 0).reset_index(drop = True)

#DATA OVERVIEW

#Dataset info 
print("\nDataset Info:")
print(dataset.info())

#Dataset sample - 10 samples
print("\nDataset Sample:")
print(dataset.sample(10))

#Dataset summary
print("\nDataset statistic summary:")
print(dataset.describe())

#MISSING DATA

#AGE
# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
plt.show()

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
plt.show()

#Correlation matrix
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
plt.show()

# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")
plt.show()

# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
indexNaNAge = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in indexNaNAge :
    ageMed = dataset["Age"].median()
    agePred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"])
     & (dataset['Parch'] == dataset.iloc[i]["Parch"])
     & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(agePred) :
        dataset['Age'].iloc[i] = agePred
    else :
        dataset['Age'].iloc[i] = ageMed

#CABIN
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'NODATA' for i in dataset['Cabin']])
g = sns.countplot(dataset["Cabin"])
plt.show()

#EMBARKED
dataset['Embarked'].value_counts().plot(kind='bar')
plt.xlabel("Embarked")
plt.ylabel("count")
plt.show()
#Filling missing values with most frequent one
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

# convert Embarked into categorical value 0 for S and 1 for Q and 2 for C
dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "Q":1, "C": 2})

#Fare
dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#FEATURE ENGINEERING

#Name/Title
datasetTitle = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(datasetTitle)
g = sns.countplot(x="Title", data = dataset)
g = plt.setp(g.get_xticklabels(), rotation = 45)
plt.show()
dataset["Title"] = dataset["Title"].replace(['Lady','the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master" : 0, "Miss" : 1, "Ms" : 1, "Mme" : 1, "Mlle" : 1, "Mrs" : 1, "Mr" : 2, "Rare" : 3})

dataset["Title"] = dataset["Title"].astype(int)

g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])

g = sns.factorplot(x = "Title", y = "Survived", data = dataset, kind = "bar")
g = g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Rare"])
g = g.set_ylabels("survival probability")
plt.show()

