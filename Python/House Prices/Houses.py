import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#load train and test dataset
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

print("Features:\n", train.columns)

#Missing values

def printMissingValues(data, heading):
    totalMissingValues = data.isnull().sum().sort_values(ascending = False)
    missingData = pd.concat([totalMissingValues], axis = 1)
    print(heading + ":\n",missingData.head(100))

printMissingValues(train, "Missing Values")

train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("No")
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("NoBasement")
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")

printMissingValues(train, "Missing Values after filing")