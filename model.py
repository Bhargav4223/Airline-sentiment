#Import the necessary libraries such as the numpy,pandas and joblib
import numpy as np
import pandas as pd
import joblib
#Reading the airline-tweets dataset.
df = pd.read_csv('Downloads/31640102-airline-tweets.csv')
data = df[['airline_sentiment','text']]
#Seperating out the required features from the dataset.
X = df['text']
y = df['airline_sentiment']
#Split the data into training data and the testing data for further use.
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
#The most important part of developing model is the feature extraction.
# Tfidf Vectorizer is which is used to do the feature extraction like removing the stop words.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
#Fit the training and testing data to this tfidf model and generate the feature extracted data.
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#Now test out for various models and choose the one which is better predicting at this one.

#First we have taken out the Naive Bayes model.
from sklearn.naive_bayes  import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train_tfidf,y_train)

#We have tried out the logisticregression model.
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf,y_train)

# We have tried out the SVC and Linear SVC.
from sklearn.svm import SVC,LinearSVC 
rbf_svc = SVC()
rbf_svc.fit(X_train_tfidf,y_train)

linear_svc = LinearSVC()
linear_svc.fit(X_train,y_train)

#After finding out the best model(SVC here) Import the pipe line and pass over the Linear SVC model and TfidfVectorizer model to the pipeline.
from sklearn.pipeline import Pipeline
pipe = Pipeline([('tfidf',TfidfVectorizer()),('svc',LinearSVC())])
#Fit the model
pipe.fit(X,y)
# Taking the input from the user.
feedback = st.text_input("Kindly Enter your feedback here:")
#Try to predict the output based on the input passed by the user.
review = pipe.predict([feedback])

#Required to save the model using the joblib library as mentioned below.
joblib.dump(pipe,'rf_model.sav')
