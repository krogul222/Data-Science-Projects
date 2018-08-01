import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *

from collections import Counter

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

#FEATURE ANALYSIS

#Age
ageBins = (0, 10, 18, 30, 60, 120)
ageGroups = ['Child', 'Teenager','Young Adult', 'Adult', 'Senior']
ageCategories = pd.cut(dataset['Age'], ageBins, labels=ageGroups)
dataset['Age'] = ageCategories
g = sns.factorplot(x = "Age", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()
dataset['Age'] = dataset['Age'].map({"Child": 0, "Teenager":1, "Young Adult": 2, 'Adult': 3, 'Senior': 4 })

#Fare"
g = sns.kdeplot(train["Fare"][(train["Survived"] == 0) & (train["Fare"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Fare"][(train["Survived"] == 1) & (train["Fare"].notnull())], color="Blue", shade= True)
g.set_xlabel("Fare")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
plt.show()

fareBins = (0, 8, 17, 25, 50, 1000)
fareGroups = ['VeryLow', 'Low','Average', 'High', 'VeryHigh']
fareCategories = pd.cut(dataset['Fare'], fareBins, labels=fareGroups)
dataset['Fare'] = fareCategories
g = sns.factorplot(x = "Fare", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()
dataset['Fare'] = dataset['Fare'].map({"VeryLow": 0, "Low":1, "Average": 2, 'High': 3, 'VeryHigh': 4 })

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

#Family size
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.factorplot(x = "Fsize", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()

#Ticket
ticketGroupCount = Counter(dataset["Ticket"])
ticketGroup = [ticketGroupCount.get(i) for i in dataset['Ticket']]
dataset["TicketSize"] = pd.Series(ticketGroup)
g = sns.factorplot(x = "TicketSize", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("survival probability")
plt.show()

#Groups
dataset["Group"] = dataset[["Fsize", "TicketSize"]].max(axis=1)
g = sns.factorplot(x = "Group", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()

#Drop unused data
dataset.drop(labels = ["Cabin"], axis = 1, inplace = True)
dataset.drop(labels = ["Fsize"], axis = 1, inplace = True)
dataset.drop(labels = ["TicketSize"], axis = 1, inplace = True)
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
dataset.drop(labels = ["Parch"], axis = 1, inplace = True)
dataset.drop(labels = ["SibSp"], axis = 1, inplace = True)
dataset.drop(labels = ["Ticket"], axis = 1, inplace = True)
dataset.drop(labels = ["Age"], axis = 1, inplace = True)
dataset.drop(labels = ["Fare"], axis = 1, inplace = True)

#scaling data
scaler = StandardScaler() # MinMaxScaler()
#dataset[['SibSp','Parch']] = scaler.fit_transform(dataset[['SibSp','Parch']])
dataset[['FareBin_Code','AgeBin_Code','Pclass', 'Group']] = scaler.fit_transform(dataset[['FareBin_Code','AgeBin_Code','Pclass', 'Group']])
