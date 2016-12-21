"""
Spyder Editor

This is my first machine learning program
based off of the iris dataset as a hello world to
machine learning.

based off of Jason Browniee's tutorial
"""
#Loading librarieis 
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Shape = Rows vs Attributes
print(dataset.shape)

# Head or listed values
print(dataset.head(20))

# Descriptions 
print(dataset.describe())

# Class Distributions or number of instances(rows)
print(dataset.groupby('class').size())

# box and whisker plots of the iris datasets
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Histogram of the same dataset
dataset.hist()
plt.show()

# This will show multivariate plots to help visualize
# The interactions between the variables
scatter_matrix(dataset)
plt.show()

"""
Now that we have our models we need to find out if the
models and the data are accurate and good for predications
"""

#Splitting our data: 80% = training and 20% = validation
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X,Y,test_size=validation_size, random_state=seed)

# Now that its been trained we can test our harness or accuracy
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

# Building the new prediction models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
 
# From the output we choose the model that gave the best results
# KNN has the largest estimated accuracy score
 
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Finally lets make a predication
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))







