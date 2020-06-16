'''
Created on Mar 27, 2020

@author: gaurav
'''
from json.decoder import NaN
from math import nan
from unittest.mock import inplace
from pandas.core.dtypes.missing import isnull

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as AS
import numpy as np
import pandas as pd
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
import xgboost as xgb
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

''' There are missing values present in the dataset. There are two kinds of missing values in conditions: unkown and missing value. 
The missing value present in pressure is also shown in two forms: -9999 and unkown values. 
Heat index has a lot of missing values.
In case of humidity the missing values are represented by N/A and also there are some missing values.
In dew there are missing values nothing else.
No missing values in fog, rain, snow, thunder, tornados and hail.
Normal missing values present in temperature.


The different conditions are: 

Smoke; Clear; Haze; Unkown; Scattered Clouds; Shallow fog; Mostly Cloudy; Fog; Partly Cloudy; Patches of Fog; Thunderstorms and rain;
Overcast; Rain; Light rain; Light drizzle; Mist; Volcanic Ash; Thunderstorm; Light thunderstorms and rain; Light thunderstorm; squalls
heavy rain; light haze; Sandstorm; Widespread dust; Funnel cloud; Heavy thunderstorm and rain; Heavy thunderstorms and hail
Light rain showers; thunderstorms and hail; partial fog; light fog; heavy fog; blowing sand; light hail showers
light sand storms; light freezing rain; rain showers
'''


#the following code is for replacing missing values. First we replace the missing values group wise according to the conditions.
# missing values are still present as some groups have medians as NaN itself. So we replace those NaNs 
# with median of the whole field.
dataset = pd.read_csv('testset.csv')
groups=dataset.groupby('_conds')
field=['_dewptm','_heatindexm','_hum','_pressurem','_tempm','_vism','_wdird','_wspdm']
 
for f in field:
    print("field", f)
    temp=groups[f].median()
    for i in range(0, 100945):
        if(isnull(dataset.loc[i,f])):
            condition=dataset.loc[i,'_conds']
            dataset.loc[i,f]=temp[condition]
            print("values: ", dataset.loc[i,f]," ; ",temp[condition])         
             

dataset['_heatindexm'].fillna(dataset['_heatindexm'].median(), inplace=True)
dataset['_hum'].fillna(dataset['_hum'].median(), inplace=True)
dataset['_tempm'].fillna(dataset['_tempm'].median(), inplace=True)
dataset['_vism'].fillna(dataset['_vism'].median(), inplace=True)

dataset = dataset.values
X = dataset[:,1:len(dataset[0])]
Y = dataset[:,0]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
for dept in range(5,8):
    for feats in range(5,8):
        classifier= GradientBoostingClassifier(max_depth=dept,max_features=feats)
        classifier.fit(X_train, Y_train)
        print("depth:",dept,"features:",feats)
        print("Score",classifier.score(X_train, Y_train))
