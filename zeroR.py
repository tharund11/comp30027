# Machine Learning Project 2
# Tharun Dharmawickrema

# Implementing a Zero-R Baseline

import pandas
import numpy
from sklearn.dummy import DummyClassifier

train_data = pandas.read_csv('recipe_train.csv')
labels = train_data['duration_label']
y = numpy.array(labels)

# Using the doc2vec50 name file to train the model, choice of dataset
# doesn't matter here as the most frequent class should be the same
# for each dataset
attributes = pandas.read_csv("train_name_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
X = numpy.array(attributes)

zero_r = DummyClassifier(strategy = 'most_frequent')
zero_r.fit(X, y)
acc = zero_r.score(X, y)
print("Score: " + str(acc) + "\n") # Scoring on the training set to see accuracy

# Predicting on corresponding test dataset
attributes = pandas.read_csv("test_name_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
test_data = numpy.array(attributes)
predictions = list(zero_r.predict(test_data))

# creating csv file of predictions
id = list(range(1,10001))
df = pandas.DataFrame(list(zip(id, predictions)), columns = ['id', 'duration_label'])
df.to_csv('zeroR.csv', index = False)