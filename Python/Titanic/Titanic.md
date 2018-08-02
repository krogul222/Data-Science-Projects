# <p style="text-align: center;">Titanic Project: Machine Learning from Disaster</p>
## 1. Problem definition
When Titanic hit an iceberg a lot of passengers and crew died. In this challenge I need to predict what sorts of people were likely to survive. Data on which I will do my prediction was included into task.

## 2. Preparation of Data
### 2.1 Data overview
To take first overview of the data let's explore whole dataset (train and test data joined) using:
* info() function

<<p align ="center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Info.png?raw=true"></p>
<p align ="center"> **Fig 2.1.1** Dataset info. </p>

* sample() function

 <p align ="center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/SampleFunctionpart1.png?raw=true"></p>
 
 <p style="text-align: center;"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/SampleFunctionpart2.png?raw=true"></p>
<p style="text-align: center;"> **Fig 2.1.2** Ten samples of dataset. </p>

* description() function
<p style="text-align: center;">![Description Function](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/DescriptionFunctionpart1.png?raw=true)</p>
<p style="text-align: center;">![Description Function](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/DescriptionFunctionpart2.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.1.3** Dataset description.</p>

There are 12 features which can be used for data analysis:
- **Survived** is a variable which need to be predicted
- **PassengerID** is assumed to be a random identifier and therefore will be exclded from analysis
- **Pclass** variable represents passenger status: 1 - upper class, 2 - middle class, 3 - lower class
- **Ticket** variable is a nominal datatype that can be used in feature engineering to check which passengers were travelling together (same ticket number)
- **Cabin** vaiable is a nominal dataype  that can be used in feature engineering to approximate passenger's position on the ship when Titanic hit an iceberg
- **Name** vaiable is a nominal dataype  that can be used in feature engineering to obtain passenger title
- **Sex** and **Embarked** variables are a nominal datatypes
- **Age** and **Fare** variables are continuous quantitative datatypes
- **SibSp** represents number of related siblings/spouse on board
- **Parch** represents number of related parents/children on board. Togather with SibSp can be used for feature engineering to create a family size variable

### 2.2 Outliers
Since outliers can have a dramatic effect on the prediction I decided to inspect if all numerical data look reasonable. From all numerical variables I analysed Fare, SibSP, Parch and Age (Pclass is not really  numerical variable but categorised, PassengerID is excluded and Survived is categorical variable which I need to predict).

I based my judgement on statisic summary of the dataset (Fig 2.1.3):
- **Age** values are between 0.17 and 80 which is reasonable
- **Parch** are between 0 and 9  which is reasonable.  Nine  children/parents may look high at first but it wasn't unusual at that time to have many children.
- **SibSP** values are between 0 and 8  which is also reasonable.  
- **Fare** values are between 0 and 512.3292 with mean 33.295479. Maximum value may look high at fist but there were high difference in passenger status. Some of them was really rich and had high quality rooms which could cost significantly higher comparing to standard tickets.

Since all numerical features have reasonable values I decided not to exclude outliers from original data. 

### 2.3 Missing values
There ae missing values in Age, Embarked, Fare and Cabin fields. Before modelling records with missing values should be deleted, fixed or feature should be excluded from analysis. Let analyse each featue one by one.

#### 2.3.1 Age
First of all I am going to plot age distribution for people who survived the tragedy and who died.

<p style="text-align: center;">![Age distribution](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSurvived.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.1** Age distribution vs Survived.</p>
<p style="text-align: center;">![Age distribution](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSurvived_two_on_one.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.2** Age distribution vs Survived. Two distributions in one plot.</p>

There is a clear difference between these two distributions. For “survival” distribution there is a peak for young passengers and group of old people (over 60) seems to be smaller comparing to “non survival” distribution. It looks like age is a valuable feature for survival prediction. Because of that it shouldn’t be excluded from analysis even there is significant amount of missing data in this field. 

To impute the missing age values I used other features to calculate reasonable guess. I based on Parch, SibSp, Pclass and Sex .

<p style="text-align: center;">![Correlation matrix](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Age correlation matrix.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.3** Correlation matrix fo Age, SibSp, Parch and Pclass features.</p>

According to correlation matrix age is not correlated with sex but it is negatively correlated with Parch, SibSp and Pclass, positively with Fare. To visualize it more I created aproperiate plots.

<p style="text-align: center;">![Age vs Sex box plots.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSex.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.4** Age vs Sex box plots.</p>

<p style="text-align: center;">![Age vs Pclass (male and female) box plots.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsPclass.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.5** Age vs Pclass (male and female) box plots.</p>

<p style="text-align: center;">![Age vs Parch box plots.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsParch.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.6** Age vs Parch box plots.</p>

<p style="text-align: center;">![Age vs SibSp box plots.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSibSp.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.7** Age vs SibSp box plots.</p>

I decided to use Parch, SibSp and Pclass features to predict missing age values.

<p style="text-align: center;">![Filling missing Age values.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Agemissingvaluescode.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.1.8** Filling missing Age values.</p>

#### 2.3.2 Cabin
Cabin field is a source of valuable information – where passenger probably was at the time when Titanic hit an iceberg. However, it has a lot of missing values (over 1000) which can introduce a lot of noise in prediction step. Eventually I decided to exclude this feature from future analysis.

#### 2.3.3 Embarked
There are only 2 missing values so I decided to fill them with most frequent value which is “S”.

<p style="text-align: center;">![Countplot of Embarked feature.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Embarked count plot.png?raw=true)</p>
<p style="text-align: center;">**Fig 2.3.3.1**  Countplot of Embarked feature.</p>

#### 2.3.4 Fare
There is only 1 missing values so I decided to fill it with median.

## 3 Feature Analysis
### 3.1 Age
Many machine learning algorithms are known to produce better model by discretizing continous features. Because of that fact I decided to arrange Age feature into 5 groups. I based on data from Fig. 2.3.1.1 and chose following groups:
* Child – age between 0 and 10
* Teenager - age between 10 and 18
* Young Adult - age between 18 and 30
* Adult - age between 30 and 60
* Senior – age between 60 and 120

<p style="text-align: center;">![Age groups vs survival probability.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgeBins.png?raw=true)</p>
<p style="text-align: center;">**Fig 3.1.1**  Age groups vs survival probability.</p>

## 4 Feature Engineering
### 4.1 Title
Name feature despite first and second name includes passenger title. It is interesting to separate title as a feature to see what is survival probability for passengers with different titles.
There are 17 different titles in the dataset but most of them are very rare. They can be grouped into 4 categories: 
- Mr
- Master
- Mrs/ Ms/Mme/ Miss
- Rare (Don, Sir, …)

<p style="text-align: center;">![Countplot of Title feature.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Titles.png?raw=true)</p>
<p style="text-align: center;">**Fig 4.1.1**  Countplot of Title feature.</p>

<p style="text-align: center;">![Title vs survival probability.](https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/TitlesvsSurvived.png?raw=true)</p>
<p style="text-align: center;">**Fig 4.1.2**  Title vs survival probability.</p>

It looks like people with rare title had more chance to survive comparing to male with basic title (Mr.).

### 4.2 Family size
It is possible that size of a family had impact on survival probability of passengers. It is not easy to evacuate big family. It is easy to construct family size feature basing on SibSp and Parch features. “Family” value is sum of SibSp, Parch and 1. 