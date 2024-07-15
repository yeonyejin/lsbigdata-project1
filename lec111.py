#교재 63페이지

#!pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt

var=['a','a', 'b', 'c']
var

sns,countplot(x=var)
plt.show()
plt.clf()

df = sns.load_dataset('titanic')
sns.countplot(data = df, x = "sex")
sns.countplot(data = df, x = "sex", hue = "sex")
plt.show()

?sns.countplot
sns.countplot(data=df, x="class")
sns.countplot(data=df, x="class",hue="alive")
sns.countplot(data=df, 
             y="class",
             hue="alive", 
             orient="v")
plt.show()
#!pip install scikit-learn
!pip install metrics
import sklearn.metrics
#sklearn,metrics.accuracy_score()

from sklearn import metrics
#metrics.accuracy_score()

import sklearn.metrics as met
#metrics.accuracy_score()
