# Machine Learning Project 2
# Tharun Dharmawickrema

# Implementing a SVM model on the data

import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

train_data = pandas.read_csv('recipe_train.csv')
labels = train_data['duration_label']
y = numpy.array(labels)

# Using one dataset to find the optimal configuration for the parameters
attributes = pandas.read_csv("train_name_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
X = numpy.array(attributes)

# Comparing accuracy of multiclass methods: one-vs-all and one-vs-one
linearSVM = LinearSVC()  # one-vs-one
linearSVC = SVC(kernel = 'linear')  # one-vs-all

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
    
linearSVM.fit(X_train, y_train)
acc = linearSVM.score(X_test, y_test)
print("Linear SVM score: " + str(acc))

linearSVC.fit(X_train, y_train)
acc = linearSVC.score(X_test, y_test)
print("Linear SVC score: " + str(acc) + "\n")

# Finding optimal polynomial degree value (one that gives highest accuracy)
scores = []
for i in range(1, 6):
    polySVC = SVC(kernel = 'poly', degree = i)
    polySVC.fit(X_train, y_train)
    scores.append(polySVC.score(X_test, y_test))

plt.figure(figsize = (10,6))
plt.plot(range(1, 6), scores, color = "red", linestyle = "dashed", marker = "o", markerfacecolor = "black")
plt.title("Accuracy vs Polynomial Degree")
plt.xlabel("Degree")
plt.ylabel("Accuracy")
plt.show()

# Trying SVM models with 3 different kernels
rbfSVC = SVC(kernel = 'rbf')
polySVC = SVC(kernel = 'poly', degree = 3) # degree 3 works best

# Training and testing the model on various datasets
filenames = ["train_name_doc2vec50.csv", "train_steps_doc2vec50.csv",
             "train_ingr_doc2vec50.csv", "train_name_doc2vec100.csv",
             "train_steps_doc2vec100.csv", "train_ingr_doc2vec100.csv"]

desc = ['Name - 50 features', 'Steps - 50 features', 'Ingredients - 50 features',
        'Name - 100 features', 'Steps - 100 features', 'Ingredients - 100 features']

for title, file in zip(desc, filenames):
    attributes = pandas.read_csv(file, index_col = False, delimiter = ',', header = None)
    X = numpy.array(attributes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    print(title)
    linearSVM.fit(X_train, y_train)
    predictions = linearSVM.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("Linear SVM Accuracy score: " + str(acc))
    print("Linear SVM F1-score: " + str(f1))
    
    rbfSVC.fit(X_train, y_train)
    predictions = rbfSVC.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("rbf SVC Accuracy score: " + str(acc))
    print("rbf SVC F1-score: " + str(f1))

    polySVC.fit(X_train, y_train)
    predictions = polySVC.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("polynomial(degree 3) SVC Accuracy score: " + str(acc))
    print("polynomial(degree 3) SVC F1-score: " + str(f1) + '\n')

# Training model on doc2vec Steps dataset and making predictions
# on test set

attributes = pandas.read_csv("train_steps_doc2vec100.csv", index_col = False, delimiter = ',', header = None)
X = numpy.array(attributes)
test_data = pandas.read_csv("test_steps_doc2vec100.csv", index_col = False, delimiter = ',', header = None)
testdata_array = numpy.array(test_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
rbfSVC.fit(X_train, y_train)
predictions = list(rbfSVC.predict(testdata_array))

# creating csv file of predictions
id = list(range(1,10001))
df = pandas.DataFrame(list(zip(id, predictions)), columns = ['id', 'duration_label'])
df.to_csv('svm.csv', index = False)