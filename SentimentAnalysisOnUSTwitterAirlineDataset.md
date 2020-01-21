# Sentiment Analysis on US Twitter Airline Dataset

Name: Ya Liu
Section: MSMA
Email: Ya.liu1@simon.rochester.edu

In this project, I trained a sentiment analysis model with the US Twitter Airline Dataset which contains 1700 Tweets on complaint about Airlines and 1700 Tweets not complaining about Airlines. We can tell the sentiment of tweets with model we developed. The precision of our model on validation set is 0.58.

# Data Preparation

Import all necessary packages


```python
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import classification_report
import pandas as pd
```


```python
import warnings
warnings.filterwarnings("ignore")
```


```python
import xgboost, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
```

    Using TensorFlow backend.


Load data and label all training data
merge two training sets


```python
negative = pd.read_csv("complaint1700.csv")
nonnegative = pd.read_csv("complaint1700.csv")

negative["label"] = "negative"
nonnegative["label"] = "nonnegative"

df = pd.concat([negative,nonnegative])
testData = pd.read_csv("test.csv")
```

Split data into training and validation set and factorize labels


```python

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['tweet'], df['label'])

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
```

Transform text into vectors by TFIDF. 
TF-IDF score represents the relative importance of a term in the document and the entire corpus. 
TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), 
the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.

Sinec there are a lot of terms in the whole corpus, 
I choose a large value for min_df argument so that model won't be affected by too much noise.


```python
vectorizer = TfidfVectorizer(min_df = 600,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(train_x)
valid_vectors = vectorizer.transform(valid_x)
test_vectors = vectorizer.transform(testData["tweet"])
```


```python
train_vectors
```




    <2550x9 sparse matrix of type '<class 'numpy.float64'>'
    	with 6768 stored elements in Compressed Sparse Row format>



# Model Selection

To ease workload, here I define a function which enables me to get a geneneral idea of performance of different models.


```python
def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return (metrics.precision_score(predictions, valid_y),metrics.recall_score(predictions, valid_y))
```


```python
# Naive Bayes 
PrecisionAndRecall = train_model(naive_bayes.MultinomialNB(), train_vectors, train_y, valid_vectors)
print ("NB: ", PrecisionAndRecall)
```

    NB:  (0.6327014218009479, 0.4776386404293381)



```python
# Linear Classifier 
PrecisionAndRecall = train_model(linear_model.LogisticRegression(), train_vectors, train_y, valid_vectors)
print ("LR: ", PrecisionAndRecall)
```

    LR:  (0.556872037914692, 0.4786150712830957)



```python
# SVM on Ngram Level TF IDF Vectors
PrecisionAndRecall = train_model(svm.SVC(), train_vectors, train_y, valid_vectors)
print ("SVM: ", PrecisionAndRecall)
```

    SVM:  (0.6303317535545023, 0.4741532976827095)



```python
# RF
PrecisionAndRecall = train_model(ensemble.RandomForestClassifier(), train_vectors, train_y, valid_vectors)
print ("RF: ", PrecisionAndRecall)
```

    RF:  (0.35308056872037913, 0.3170212765957447)



```python
# Extereme Gradient Boosting
PrecisionAndRecall = train_model(xgboost.XGBClassifier(), train_vectors, train_y, valid_vectors)
print ("Extereme Gradient Boosting: ", PrecisionAndRecall)
```

    Extereme Gradient Boosting:  (0.26066350710900477, 0.2981029810298103)


From result above, I can tell that SVM performs the best compared with other models in general. 
So I choose SVM as traning model for further parameter tunning

# Hyperparameter tunning

I got to know Gridsearch for hyperparameter tunning from another course called predictive analytics I took this semester.
Here I will use it for hyper parameter tuning.
Since it takes a long time to try different combinations, I set a small value to cv.


```python
from sklearn.model_selection import GridSearchCV
param_grid = [{'kernel': ['rbf'], 'gamma': [0.5,1,1.5,2,3],
                     'C': [1,5,10,15,20,25],"probability":[True]},
              {'kernel': ['linear'], 'C': [1,5,10,15,20,25],"probability":[True]},
              {'kernel': ['poly'], 'gamma': [0.5,1,1.5,2,3],'C': [1,5,10,15,20,25],"probability":[True]}    
             ]
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=2, scoring='accuracy')
grid_search.fit(train_vectors, train_y)
```




    GridSearchCV(cv=2, error_score='raise-deprecating',
                 estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                               decision_function_shape='ovr', degree=3,
                               gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                               probability=False, random_state=None, shrinking=True,
                               tol=0.001, verbose=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'C': [1, 5, 10, 15, 20, 25],
                              'gamma': [0.5, 1, 1.5, 2, 3], 'kernel': ['rbf'],
                              'probability': [True]},
                             {'C': [1, 5, 10, 15, 20, 25], 'kernel': ['linear'],
                              'probability': [True]},
                             {'C': [1, 5, 10, 15, 20, 25],
                              'gamma': [0.5, 1, 1.5, 2, 3], 'kernel': ['poly'],
                              'probability': [True]}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)



After grid search, we can build a model with best parameters.


```python
model = grid_search.best_estimator_
print("best parameters：",grid_search.best_params_)
```

    best parameters： {'C': 5, 'kernel': 'linear', 'probability': True}


Let's see how many observations we can get.


```python
predict_value = model.predict(test_vectors)
```

Retry other hyperparameters


```python
param_grid = [{'kernel': ['rbf'], 'gamma': [5,10],
                     'C': [100,150],"probability":[False]},
              {'kernel': ['linear'], 'C': [100,150],"probability":[False]},
             ]
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=2, scoring='accuracy')
grid_search.fit(train_vectors, train_y)
```




    GridSearchCV(cv=2, error_score='raise-deprecating',
                 estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                               decision_function_shape='ovr', degree=3,
                               gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                               probability=False, random_state=None, shrinking=True,
                               tol=0.001, verbose=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'C': [100, 150], 'gamma': [5, 10], 'kernel': ['rbf'],
                              'probability': [False]},
                             {'C': [100, 150], 'kernel': ['linear'],
                              'probability': [False]}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
model = grid_search.best_estimator_
print("best parameters：",grid_search.best_params_)
```

    best parameters： {'C': 100, 'kernel': 'linear', 'probability': False}



```python
predict_value = model.predict(test_vectors)
```

final precision: 157/267=0.58
