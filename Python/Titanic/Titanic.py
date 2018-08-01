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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold, learning_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

#load train and test dataset
train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')

#join train and test datasets before categorical conversion
testID = test["PassengerId"]
trainLength = len(train)
dataset = pd.concat(objs=[train, test], axis = 0).reset_index(drop = True)
print(dataset)

# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)
print("\nNull values summary:")
print(dataset.isnull().sum())

#Info
print("\nTrainDataSet Info:")
print(train.info())
print("\nTrainDataSet missing data:")
print(train.isnull().sum())
print("\nTestDataSet missing data:")
print(test.isnull().sum())
print("\nTrainDataSet overview:")
print(train.head())

# Summary and statistics of TrainDataSet
print("\nTrainDataSet summary and statistics:")
print(train.describe())

print("\nWholeDataSet missing data before filling:")
print(dataset.isnull().sum())

# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
print("Outliers:")
print(train.loc[Outliers_to_drop])

# Drop outliers
#train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)



#MISSING DATA

#Fare
dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#Embarked
dataset['Embarked'].value_counts().plot(kind='bar')
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

#Age
# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])

# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Embarked", data=dataset,kind="box")

# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
# convert Embarked into categorical value 0 for S and 1 for Q and 2 for C
dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "Q":1, "C": 2})

g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass", "Embarked"]].corr(),cmap="BrBG",annot=True)

# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med


print("\nWholeDataSet missing data after filling:")
print(dataset.isnull().sum())


#FEATURE ANALYSIS

dataset['AgeBin'] = pd.qcut(dataset['Age'], 4)

label = LabelEncoder()
dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
dataset.drop(labels = ["AgeBin"], axis = 1, inplace = True)
dataset.drop(labels = ["Age"], axis = 1, inplace = True)



# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare", "Pclass"]].corr(),annot = True, fmt = ".2f", cmap = "coolwarm")

#SibSp vs Survived"
g = sns.factorplot(x = "SibSp", y = "Survived", data = train, kind = "bar", size = 6, palette = "muted")
g.despine(left = True)
g = g.set_ylabels("survival probability")

#Parch vs Survived
g = sns.factorplot(x = "Parch", y = "Survived", data = train, kind = "bar", size = 6, palette = "muted")
g.despine(left = True)
g = g.set_ylabels("survival probability")

#Fare"
g = sns.distplot(dataset["Fare"], color ="m", label = "Skewness: %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = "best")

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(dataset["Fare"], color = "b", label = "Skewness: %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = "best")

# Feature Engineering

#Name/Title
print(dataset["Name"].head())

datasetTitle = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(datasetTitle)
print(dataset["Title"].head())

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

#Cabin

#dataset["Cabin"].head()
#dataset["Cabin"].describe()
#dataset["Cabin"].isnull().sum()

#dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
#g = sns.countplot(dataset["Cabin"])
#g = sns.factorplot(y = "Survived", x = "Cabin", data = dataset, kind="bar")
#g = g.set_ylabels("Survival Probability")

#dataset = pd.get_dummies(dataset, columns = ["Cabin"], prefix = "Cabin")
dataset.drop(labels = ["Cabin"], axis = 1, inplace = True)

#Ticket

TicketGroupCount = Counter(dataset["Ticket"])
print("Ticketcount\n")
print(TicketGroupCount)
TicketGroup = [TicketGroupCount.get(i) for i in dataset['Ticket']]
dataset["TicketSize"] = pd.Series(TicketGroup)

print("TicketSize\n")
print(dataset["TicketSize"])

g = sns.factorplot(x = "TicketSize", y = "Survived", data = dataset, kind = "bar", size = 6, palette = "muted")
g.despine(left = True)
g = g.set_ylabels("survival probability")
plt.show()

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("X")

dataset["Ticket"] = Ticket
dataset["Ticket"].head()

# dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix = "T")
dataset.drop(labels = ["Ticket"], axis = 1, inplace = True)



# Family Size

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
dataset["Group"] = dataset[["Fsize", "TicketSize"]].max(axis=1)

print("Group:")
print(dataset["Group"].head())

g = sns.factorplot(x = "Group", y = "Survived", data = dataset)
g = g.set_ylabels("Survival Probability")
plt.show()
#Create new Feature of family size
#dataset['Single'] = dataset['Group'].map(lambda s: 1 if s==1 else 0)
#dataset['Duo'] = dataset['Group'].map(lambda s: 1 if s==2 else 0)
#dataset['MedGroup'] = dataset['Group'].map(lambda s: 1 if 3<=s<=4 else 0)
#dataset['LargeG'] = dataset['Group'].map(lambda s: 1 if s>=5 else 0)

dataset.drop(labels = ["Fsize"], axis = 1, inplace = True)
#dataset.drop(labels = ["Group"], axis = 1, inplace = True)
dataset.drop(labels = ["TicketSize"], axis = 1, inplace = True)

# convert to indicator values Title and Embarked 
dataset = pd.get_dummies(dataset, columns = ["Title"])
#dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

# Create categorical values for Pclass
#dataset["Pclass"] = dataset["Pclass"].astype("category")
#dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")

# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)

print(dataset.head())

#Drop Name
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

dataset.drop(labels = ["Embarked"], axis = 1, inplace = True)
dataset.drop(labels = ["Parch"], axis = 1, inplace = True)
dataset.drop(labels = ["SibSp"], axis = 1, inplace = True)



#Drop Cabin
#dataset.drop(labels = ["Cabin"], axis = 1, inplace = True)

#Drop Tickets
#dataset.drop(labels = ["Ticket"], axis = 1, inplace = True)

#Family size

dataset['FareBin'] = pd.qcut(dataset['Fare'], 5)

label = LabelEncoder()
dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
dataset.drop(labels = ["FareBin"], axis = 1, inplace = True)
dataset.drop(labels = ["Fare"], axis = 1, inplace = True)

#scaling data
scaler = StandardScaler() # MinMaxScaler()
#dataset[['SibSp','Parch']] = scaler.fit_transform(dataset[['SibSp','Parch']])
dataset[['FareBin_Code','AgeBin_Code','Pclass', 'Group']] = scaler.fit_transform(dataset[['FareBin_Code','AgeBin_Code','Pclass', 'Group']])

print("After scaling:\n")
print(dataset.head())

#Modelling

#Separate train and data
train = dataset[:trainLength]
test = dataset[trainLength:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)

#Separate train features and label

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"], axis = 1)

print("TRAIN:\n")
print(X_train)

# Cross validate model with Kfold stratified cross val
kfold = KFold(n_splits=5, random_state=22)

# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_+state))

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs = 4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans" : cv_means, "CrossValerrors": cv_std, "Algorithm" : ["SVC", "KNeighbors", "RandomForest", "DecisionTree", "GradientBoosting"]})

g = sns.barplot("CrossValMeans", "Algorithm", data = cv_res, palette="Set3", orient = "h", **{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

plt.show()

dataset.info()

#Hyperparameter tunning for best models

#RandomForest

RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "max_features": [2, 4, 6, 'auto'],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "n_estimators" :[100, 200 ,400],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
print("Best RF score:\n")
print(gsRFC.best_score_)

#### Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

# Best score
print("Best Ada score:\n")
print(gsadaDTC.best_score_)

#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 6],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_

# Best score
print("Best ExtC score:\n")
print(gsExtC.best_score_)


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
print("Best SVMC score:\n")
print(gsSVMC.best_score_)

# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300, 400],
              'learning_rate': [0.1, 0.05, 0.01, 0.001],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.2, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
print("Best Gradient score:\n")
print(gsGBC.best_score_)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    print("""Generate a simple plot of the test and training learning curve""")
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#g = plot_learning_curve(RFC_best,"RF mearning curves",X_train,Y_train,cv=kfold)
#g = plot_learning_curve(SVMC_best,"SVC learning curves",X_train,Y_train,cv=kfold)

#Test

test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_SVMC = pd.Series(GBC_best.predict(test), name="GBC")
test_Survived_ADAC = pd.Series(ada_best.predict(test), name="ADAC")
test_Survived_EXTC = pd.Series(ExtC_best.predict(test), name="EXTC")

print(np.corrcoef(test_Survived_RFC,test_Survived_SVMC))

#Ensemble modelling

votingC = VotingClassifier(estimators=[('rfc', RFC_best),
('svc', SVMC_best), ('gbc', GBC_best), ('adac', ada_best), ('extc', ExtC_best)], voting='hard', n_jobs=1)
votingC = votingC.fit(X_train, Y_train)

#Prediction

test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([testID,test_Survived],axis=1)
results.to_csv("Titanic_ensemble_python_voting.csv",index=False)

test_Survived = pd.Series(RFC_best.predict(test), name="Survived")
results = pd.concat([testID,test_Survived],axis=1)
results.to_csv("Titanic_ensemble_python_voting_RF.csv",index=False)

test_Survived2 = pd.Series(SVMC_best.predict(test), name="Survived")
results = pd.concat([testID,test_Survived2],axis=1)
results.to_csv("Titanic_ensemble_python_voting2.csv",index=False)