# Machine Learning Project 2
# Tharun Dharmawickrema

# Implementing a Decision Tree model on the data

import numpy
import scipy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import f1_score, accuracy_score

train_data = pandas.read_csv('recipe_train.csv')
labels = train_data['duration_label']
y = numpy.array(labels)

# Attempting to discretise one dataset, train the model and test
attributes = pandas.read_csv("train_name_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
X = numpy.array(attributes)
dt = DecisionTreeClassifier()

# Equal-width discretisation
discretizer = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy ='uniform')
X_discrete = discretizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, test_size = 0.6)
dt.fit(X_train, y_train)
acc = dt.score(X_test, y_test)
print("Equal-width discretisation")
print("DT score: " + str(acc) + "\n")

# Equal-frequency discretisation
discretizer2 = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy ='quantile')
X_discrete = discretizer2.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, test_size = 0.6)
dt.fit(X_train, y_train)
acc = dt.score(X_test, y_test)
print("Equal-frequency discretisation")
print("DT score: " + str(acc) + "\n")

# Discretisation using k-means
discretizer3 = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy ='kmeans')
X_discrete = discretizer3.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, test_size = 0.6)
dt.fit(X_train, y_train)
acc = dt.score(X_test, y_test)
print("Discretisation using k-means")
print("DT score: " + str(acc) + "\n")

# Training and testing the model on different datasets (without discretising)
filenames = ["train_name_doc2vec50.csv", "train_steps_doc2vec50.csv",
             "train_ingr_doc2vec50.csv", "train_name_doc2vec100.csv",
             "train_steps_doc2vec100.csv", "train_ingr_doc2vec100.csv"]

desc = ['Name - 50 features', 'Steps - 50 features', 'Ingredients - 50 features',
         'Name - 100 features', 'Steps - 100 features', 'Ingredients - 100 features']

dt = DecisionTreeClassifier(criterion = "entropy")
dt_gini = DecisionTreeClassifier()

# Trying out both Information Gain & Gini Coefficient
for title, file in zip(desc, filenames):
    attributes = pandas.read_csv(file, index_col = False, delimiter = ',', header = None)
    X = numpy.array(attributes)

    print(title)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6)
    
    dt.fit(X_train, y_train)
    predictions = dt.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("DT (Info Gain) Accuracy score: " + str(acc))
    print("DT (Info Gain) F1-score: " + str(f1))
    
    dt_gini.fit(X_train, y_train)
    predictions = dt_gini.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("DT (Gini coefficient) Accuracy score: " + str(acc))
    print("DT (Gini coefficient) F1-score: " + str(f1) + '\n')
    
filenames2 = ['train_name_vec.npz', 'train_steps_vec.npz', 'train_ingr_vec.npz']
desc2 = ['Name - CountVectorizer', 'Steps - CountVectorizer', 'Ingredients - CountVectorizer']

# Trying out both Information Gain & Gini Coefficient
for title, file in zip(desc2, filenames2):
    matrix = scipy.sparse.load_npz(file)
    X = matrix.toarray()

    print(title)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6)
    
    dt.fit(X_train, y_train)
    predictions = dt.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("DT (Info Gain) Accuracy score: " + str(acc))
    print("DT (Info Gain) F1-score: " + str(f1))
    
    dt_gini.fit(X_train, y_train)
    predictions = dt_gini.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("DT (Gini coefficient) Accuracy score: " + str(acc))
    print("DT (Gini coefficient) F1-score: " + str(f1) + '\n')

# Picking the Gini Coefficient criterion and the CountVectorizer steps data
# to make the predictions for the test data

matrix = scipy.sparse.load_npz('train_steps_vec.npz')
test_matrix = scipy.sparse.load_npz('test_steps_vec.npz')

X = matrix.toarray()
test_data = test_matrix.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6)
dt_gini.fit(X_train, y_train)
predictions = list(dt_gini.predict(test_data))

# creating csv file of predictions
id = list(range(1,10001))
df = pandas.DataFrame(list(zip(id, predictions)), columns = ['id', 'duration_label'])
df.to_csv('decisionTree.csv', index = False)