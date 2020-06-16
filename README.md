# Delhi-Weather-Condition-Prediction
Delhi weather dataset is used to train a machine learning model which can predict the weather conditions of a particular day using ensemble classifiers.
The dataset contains different features like humidity, temperature, wind speed, wind direction, etc. These features are used to predict the conditions of a particular day. The different conditions are cloudy, fog, rain, thunderstorm, etc.

There are missing values present in the dataset. There are two kinds of missing values in conditions: unkown and missing value. 
The missing value present in pressure is also shown in two forms: -9999 and unkown values. 
Heat index has a lot of missing values.
In case of humidity the missing values are represented by N/A and also there are some missing values.
In dew there are missing values nothing else.
No missing values in fog, rain, snow, thunder, tornados and hail.
Normal missing values present in temperature.

The missing values are replaced by the mean of the group. If the mean of th e group is N/A then it is replaced by the mean of the whole feature.
