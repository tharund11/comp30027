# Machine Learning Project 2
# Tharun Dharmawickrema

# Implementing a Naive Bayes model on the data
# Some of the code has been adapted from Practical 4

import pandas
import numpy
import scipy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import f1_score, accuracy_score

# Training and testing on different data features (training datasets)
train_data = pandas.read_csv('recipe_train.csv')
labels = train_data['duration_label']
y = numpy.array(labels)

filenames = ["train_name_doc2vec50.csv", "train_steps_doc2vec50.csv",
             "train_ingr_doc2vec50.csv", "train_name_doc2vec100.csv",
             "train_steps_doc2vec100.csv", "train_ingr_doc2vec100.csv"]

desc = ['Name - 50 features', 'Steps - 50 features', 'Ingredients - 50 features',
         'Name - 100 features', 'Steps - 100 features', 'Ingredients - 100 features']

gnb = GaussianNB()

# Using a GNB model for the doc2Vec data
for title, file in zip(desc, filenames):
    attributes = pandas.read_csv(file, index_col = False, delimiter = ',', header = None)
    X = numpy.array(attributes)

    gnb_accs = []
    gnb_f1 = []
    print(title)
    
    # Training & testing 3 times to get average
    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=i)
        gnb.fit(X_train, y_train)
        predictions = gnb.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average = 'micro')

        print("GNB Accuracy score: " + str(acc))
        print("GNB F1-score: " + str(f1) + '\n')
        gnb_accs.append(acc)
        gnb_f1.append(f1)
        
    print('Avg GNB Accuracy score: ' + str(numpy.mean(gnb_accs)))
    print('Avg GNB F1-score: ' + str(numpy.mean(gnb_f1)) + '\n')

filenames2 = ['train_name_vec.npz', 'train_steps_vec.npz', 'train_ingr_vec.npz']
desc2 = ['Name - CountVectorizer', 'Steps - CountVectorizer', 'Ingredients - CountVectorizer']

mnb = MultinomialNB()
bnb = BernoulliNB()

# Using a MultinomialNB and a BernoulliNB model for the CountVectorizer data
for title, file in zip(desc2, filenames2):
    matrix = scipy.sparse.load_npz(file)
    X = matrix.toarray()

    print(title)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("MNB Accuracy score: " + str(acc))
    print("MNB F1-score: " + str(f1))

    bnb.fit(X_train, y_train)
    predictions = bnb.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average = 'micro')
    print("BNB Accuracy score: " + str(acc))
    print("BNB F1-score: " + str(f1) + '\n')

# Picking the MultinomialNB model and the CountVectorizer steps data
# to make the predictions for the test data

matrix = scipy.sparse.load_npz('train_steps_vec.npz')
test_matrix = scipy.sparse.load_npz('test_steps_vec.npz')

X = matrix.toarray()
test_data = test_matrix.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
mnb.fit(X_train, y_train)
predictions = list(mnb.predict(test_data))

# creating csv file of predictions
id = list(range(1,10001))
df = pandas.DataFrame(list(zip(id, predictions)), columns = ['id', 'duration_label'])
df.to_csv('naiveBayes.csv', index = False)