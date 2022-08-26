import pandas as pd
from time import  time
import warnings
import seaborn as sns
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')  
nltk.download('wordnet')
scaler = StandardScaler()
selector = VarianceThreshold()
pca = PCA()
label_encoder = preprocessing.LabelEncoder()




# read text file into pandas DataFrame
df = pd.read_csv("spam-no-spam.txt", sep=",")

#tokenize,remove stop words/closed tags and pos_tag the df
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(str)
df['text'].dropna(inplace=True)
df['text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
df['text'] = df['text'].apply(lambda x: [i for i in x if i[0].isalpha()])


def join(text):
    return " ".join(text)
df['text'] = df['text'].apply(join)

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x = v.fit_transform(df['text']).toarray()
X = x
y1 = df['spam']

def train_model_clf(model,param_grid,X,y):
    # Splitting the dataset into the Training set and Test set
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #Ορίζω το Pipe του RandomForest 
    pipeline = Pipeline(steps=[('scaler', scaler), ('selector', selector), ('pca', pca), ("clf",model)])
    # initialize
    grid_pipeline = GridSearchCV(pipeline,param_grid=param_grid, cv=3, n_jobs=-1,verbose=1)
    # fit
    grid_pipeline.fit(x_train,y_train)
    best_model = grid_pipeline.best_estimator_
    best_params = grid_pipeline.best_params_
    # make the predictions
    print()
    print("Training the model for Testing")
    start = time()
    y_pred = grid_pipeline.predict(x_test)
    print('The best params for this model were :',best_params)
    print('Our best estimator is ', best_model)
    metrics_eval_clas(y_test, y_pred,grid_pipeline)
    end = time()
    tr_time = round(end - start, 2)
    print("The training completed in : {} seconds.".format(tr_time))
    print()


def metrics_eval_clas(y_test, y_pred,pipeline):
    print('score of the model =', round(pipeline.best_score_*100,2),'%')
    print('accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)*100,2))
    print('f1-score: {:.3f}'.format(f1_score(y_test, y_pred, average='macro')*100,2))
    print('precision: {:.3f}'.format(precision_score(y_test, y_pred, average='macro')*100,2))
    print('recall: {:.3f}'.format(recall_score(y_test, y_pred, average='macro')*100,2))
    print(classification_report(y_test, y_pred))


param_grid1 = {'clf__n_estimators': [500],
               'clf__max_features': ['sqrt'],
               'clf__max_depth': [40],
               'clf__min_samples_split': [3],
               'clf__min_samples_leaf': [1],
               'clf__bootstrap': [True]}

param_grid2 = {'clf__max_depth': [2,3,6],
               'clf__n_estimators': range(60, 220, 40),
                'clf__learning_rate': [0.1,0.01,1]}


param_grid3 = {'clf__C': [10],
              'selector__threshold' : [1e-2],
              'pca__n_components'   : [0.95],
              'clf__gamma': [0.5],
              'clf__kernel': ['rbf']}



################################################### Classifiers ################################################################################################

#Rf Classifier
random_forrest_clf = train_model_clf(RandomForestClassifier(),param_grid1,X,y1)

#XGBClassifier
XGBClassifier = train_model_clf(XGBClassifier(),param_grid2,X,y1)

#SVMClassifier
SVMClassifier = train_model_clf(SVC(),param_grid3,X,y1)