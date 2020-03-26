import pandas
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
import time

#Android_Feats
malDataSet = pandas.read_csv('Android_Feats.csv')#, sep=';'
#malDataSet = pandas.read_csv('dataset.csv')#, sep=';'

origin_headers = list(malDataSet.columns.values)

total_data = malDataSet[origin_headers[:-1]]
total_data = total_data.as_matrix()
target_strings = malDataSet[origin_headers[-1]]

train, test, target_train, target_test = train_test_split(total_data, target_strings, test_size=0.33, random_state=int(time.time()))


"""
classifiers = [
    RandomForestClassifier(n_estimators=100),
    DecisionTreeClassifier(),
    AdaBoostClassifier()
]
"""

classifiers = {
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "AdaBoost Classifier": AdaBoostClassifier()
}

for name, clf in classifiers.items():
    clf.fit(train, target_train)
    score = clf.score(test, target_test)
    print(name, score)

