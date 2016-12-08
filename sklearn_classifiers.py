import sklearn
import itertools

from sklearn import ensemble
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

classifiers = [tree.DecisionTreeClassifier(),
               neighbors.KNeighborsClassifier(n_neighbors=3),
               ensemble.RandomForestClassifier()]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
for j in range(len(classifiers)):
    clf = classifiers[j].fit(X_train, Y_train)
    Y_predicted = clf.predict(X_test)
    print(type(clf).__name__, 'mean accuracy: ', accuracy_score(Y_test, Y_predicted))
