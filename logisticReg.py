# Machine Learning Project 2
# Tharun Dharmawickrema

# Implementing a Logistic Regression model on the data

import numpy
import scipy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

train_data = pandas.read_csv('recipe_train.csv')
labels = train_data['duration_label']
y = numpy.array(labels)

# Using one dataset to find the optimal configuration for the parameters
attributes = pandas.read_csv("train_name_doc2vec50.csv", index_col = False, delimiter = ',', header = None)
X = numpy.array(attributes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# Comparing accuracy of multiclass methods: one-vs-rest and one-vs-one
logisticReg = LogisticRegression(multi_class = 'ovr')   # one-vs-rest
logisticReg.fit(X_train, y_train)
acc = logisticReg.score(X_test, y_test)
print("Logistic Regression (one-vs-rest) score: " + str(acc))

logisticReg2 = LogisticRegression(multi_class = 'multinomial') # one-vs-one
logisticReg2.fit(X_train, y_train)
acc = logisticReg2.score(X_test, y_test)
print("Logistic Regression (cross-entropy loss) score: " + str(acc) + "\n")

# Comparing different optimization algorithms
newton = LogisticRegression(solver = 'newton-cg')
newton.fit(X_train, y_train)
acc = newton.score(X_test, y_test)
print("Logistic Regression (newton solver) score: " + str(acc))

lbfgs = LogisticRegression(solver = 'lbfgs')
lbfgs.fit(X_train, y_train)
acc = lbfgs.score(X_test, y_test)
print("Logistic Regression (lbfgs) score: " + str(acc))

liblinear = LogisticRegression(solver = 'liblinear')
liblinear.fit(X_train, y_train)
acc = liblinear.score(X_test, y_test)
print("Logistic Regression (liblinear) score: " + str(acc))

sag = LogisticRegression(solver = 'sag')
sag.fit(X_train, y_train)
acc = sag.score(X_test, y_test)
print("Logistic Regression (sag) score: " + str(acc))

saga = LogisticRegression(solver = 'saga')
saga.fit(X_train, y_train)
acc = saga.score(X_test, y_test)
print("Logistic Regression (saga) score: " + str(acc) + "\n")

# selecting the liblinear optimization algorithm
logisticRegmodel = LogisticRegression(solver = 'liblinear')

# testing model on dataset based on 2 features I selected
X_myfeatures = numpy.array(train_data[['n_steps', 'n_ingredients']])
X_train, X_test, y_train, y_test = train_test_split(X_myfeatures, y, test_size = 0.33)
logisticRegmodel.fit(X_train, y_train)
predictions = logisticRegmodel.predict(X_test)
print("Logistic Regression (my features) score: " + str(accuracy_score(y_test, predictions)) + '\n')

# Training and testing the model on different datasets
filenames = ["train_name_doc2vec50.csv", "train_steps_doc2vec50.csv",
             "train_ingr_doc2vec50.csv", "train_name_doc2vec100.csv",
             "train_steps_doc2vec100.csv", "train_ingr_doc2vec100.csv"]

desc = ['Name - 50 features', 'Steps - 50 features', 'Ingredients - 50 features',
         'Name - 100 features', 'Steps - 100 features', 'Ingredients - 100 features']

for title, file in zip(desc, filenames):
    attributes = pandas.read_csv(file, index_col = False, delimiter = ',', header = None)
    X = numpy.array(attributes)

    accs = []
    f1scores = []
    print(title)
    
    # Training & testing 3 times to get average
    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=i)
        logisticRegmodel.fit(X_train, y_train)
        predictions = logisticRegmodel.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average = 'micro')
        accs.append(acc)
        f1scores.append(f1)
        
    print('Avg Logistic Regression Accuracy score: ' + str(numpy.mean(accs)))
    print('Avg Logistic Regression F1-score: ' + str(numpy.mean(f1scores)) + '\n')
    
filenames2 = ['train_name_vec.npz', 'train_steps_vec.npz', 'train_ingr_vec.npz']
desc2 = ['Name - CountVectorizer', 'Steps - CountVectorizer', 'Ingredients - CountVectorizer']

for title, file in zip(desc2, filenames2):
    matrix = scipy.sparse.load_npz(file)
    X = matrix.toarray()

    accs = []
    f1scores = []
    print(title)
    
    # Training & testing 3 times to get average
    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=i)
        logisticRegmodel.fit(X_train, y_train)
        predictions = logisticRegmodel.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average = 'micro')
        accs.append(acc)
        f1scores.append(f1)
        
    print('Avg Logistic Regression Accuracy score: ' + str(numpy.mean(accs)))
    print('Avg Logistic Regression F1-score: ' + str(numpy.mean(f1scores)) + '\n')

# Picking the CountVectorizer steps data to make the predictions for the test data

matrix = scipy.sparse.load_npz('train_steps_vec.npz')
test_matrix = scipy.sparse.load_npz('test_steps_vec.npz')

X = matrix.toarray()
test_data = test_matrix.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
logisticRegmodel.fit(X_train, y_train)
predictions = list(logisticRegmodel.predict(test_data))

# creating csv file of predictions
id = list(range(1,10001))
df = pandas.DataFrame(list(zip(id, predictions)), columns = ['id', 'duration_label'])
df.to_csv('logisticReg.csv', index = False)