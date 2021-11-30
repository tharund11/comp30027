# Machine Learning Project 2
# Tharun Dharmawickrema

# Implementing a kNN model on the data

import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split

train_data = pandas.read_csv('recipe_train.csv')
labels = train_data['duration_label']
y = numpy.array(labels)

# Using one dataset to find the optimal configuration for the parameters
attributes = pandas.read_csv("train_name_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
X = numpy.array(attributes)

# Finding optimal k value (one that gives highest accuracy)
scores = []
for i in range(1, 210, 10):
    knn = KNeighborsClassifier(n_neighbors = i)
    acc = numpy.mean(cross_val_score(knn, X, y, cv = 10))
    scores.append(acc)

plt.figure(figsize = (10,6))
plt.plot(range(1, 210, 10), scores, color = "red", linestyle = "dashed", marker = "o", markerfacecolor = "black")
plt.title("Accuracy vs K-value")
plt.xlabel("K-value")
plt.ylabel("Accuracy")
plt.show()

# Picking k = 60, finding best distance metric
knn = KNeighborsClassifier(n_neighbors = 60)
acc = numpy.mean(cross_val_score(knn, X, y, cv = 10))
print("Euclidean distance score: " + str(acc) + "\n")

knn2 = KNeighborsClassifier(n_neighbors = 60, metric = "manhattan")
acc = numpy.mean(cross_val_score(knn2, X, y, cv = 10))
print("Manhattan distance score: " + str(acc) + "\n")

# Checking if distance weighting is better than equal weighting
knn3 = KNeighborsClassifier(n_neighbors = 60, weights = 'distance')
acc = numpy.mean(cross_val_score(knn3, X, y, cv = 10))
print("Weighted distance score: " + str(acc) + "\n")

# Training and testing the model on various datasets
filenames = ["train_name_doc2vec50.csv", "train_steps_doc2vec50.csv",
             "train_ingr_doc2vec50.csv"]

desc = ['Name - 50 features', 'Steps - 50 features', 'Ingredients - 50 features']

knn = KNeighborsClassifier(n_neighbors = 60, weights = "distance")

for title, file in zip(desc, filenames):
    attributes = pandas.read_csv(file, index_col = False, delimiter = ',', header = None)
    X = numpy.array(attributes)

    print(title)
    acc = numpy.mean(cross_val_score(knn, X, y, cv = 10))
    print("kNN score: " + str(acc) + "\n")

# Training model on doc2vec Steps dataset and making predictions
# on test set

attributes = pandas.read_csv("train_steps_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
X = numpy.array(attributes)
test_data = pandas.read_csv("test_steps_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
testdata_array = numpy.array(test_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
knn.fit(X_train, y_train)
predictions = list(knn.predict(testdata_array))

# creating csv file of predictions
id = list(range(1,10001))
df = pandas.DataFrame(list(zip(id, predictions)), columns = ['id', 'duration_label'])
df.to_csv('kNN.csv', index = False)