import pandas
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
import time

#Android_Feats
malDataSet = pandas.read_csv('dataset.csv', sep=';')#low_memory=False, error_bad_lines=False

origin_headers = list(malDataSet.columns.values)

total_data = malDataSet[origin_headers[:-1]]
total_data = total_data.as_matrix()
target_strings = malDataSet[origin_headers[-1]]

train, test, target_train, target_test = train_test_split(total_data, target_strings, test_size=0.33, random_state=int(time.time()))

clf = RandomForestClassifier(n_estimators=100)
clf.fit(train, target_train)
score = clf.score(test, target_test)
print(score)

classifiers = [
    RandomForestClassifier(n_estimators=100),
    DecisionTreeClassifier(),
    AdaBoostClassifier()
]