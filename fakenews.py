# Include Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
import numpy as np  # Importing NumPy

# Importing dataset using pandas dataframe
df = pd.read_csv("fake_or_real_news.csv")

# Inspect shape of `df`
print("Shape of dataframe:", df.shape)

# Print first lines of `df`
print("First lines of dataframe:")
print(df.head())

# Set index
df = df.set_index("Unnamed: 0")

# Print first lines of `df`
print("First lines of dataframe after setting index:")
print(df.head())

# Separate the labels and set up training and test datasets
y = df.label

# Drop the `label` column
df.drop("label", axis=1, inplace=True)

# Make training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Building the Count and Tfidf Vectors
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test set
count_test = count_vectorizer.transform(X_test)

# Initialize the `tfidf_vectorizer`
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test set
tfidf_test = tfidf_vectorizer.transform(X_test)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Naive Bayes classifier for Multinomial model
clf = MultinomialNB()
alpha = 1.0  # Laplace smoothing parameter
clf = MultinomialNB(alpha=alpha)

clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("TF-IDF Vectorizer Accuracy: %.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print("Confusion Matrix for TF-IDF Vectorizer:")
print(cm)


clf = MultinomialNB()

clf.fit(count_train, y_train)

pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("Count Vectorizer Accuracy: %.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print("Confusion Matrix for Count Vectorizer:")
print(cm)

# Applying Passive Aggressive Classifier
linear_clf = PassiveAggressiveClassifier(n_iter_no_change=50)

linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("TF-IDF Vectorizer Accuracy (Passive Aggressive Classifier): %.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print("Confusion Matrix for TF-IDF Vectorizer (Passive Aggressive Classifier):")
print(cm)

clf = MultinomialNB(alpha=0.1)

last_score = 0
for alpha in np.arange(0, 1, .1):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(tfidf_train, y_train)
    pred = nb_classifier.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    if score > last_score:
        clf = nb_classifier
    print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))

# Function to get most informative features for binary classification
def most_informative_features(vectorizer, classifier, n=10):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:n]

    print("Top %d features for class %s:" % (n, class_labels[0]))
    for coef, feat in topn_class1:
        print(coef, feat)

    print("\nTop %d features for class %s:" % (n, class_labels[1]))
    for coef, feat in topn_class2:
        print(coef, feat)

most_informative_features(tfidf_vectorizer, linear_clf, n=30)

# HashingVectorizer
hash_vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False)
hash_train = hash_vectorizer.fit_transform(X_train)
hash_test = hash_vectorizer.transform(X_test)

clf = MultinomialNB(alpha=.01)

clf.fit(hash_train, y_train)
pred = clf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
print("Hashing Vectorizer Accuracy: %.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
